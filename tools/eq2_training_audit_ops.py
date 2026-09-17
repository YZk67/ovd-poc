"""Detached-feature Eq.2 probes. These are NOT decoder/optimizer gradients or AP."""
from __future__ import annotations

from contextlib import contextmanager
import math
from statistics import mean

import torch
import torch.nn.functional as F


VARIANTS = ("calibrated", "calibrated_plus_logK", "legacy")


def matcher_settings(matcher):
    keys = ("cost_class", "cost_bbox", "cost_giou", "alpha", "gamma", "cost_class_type")
    settings = {k: getattr(matcher, k) for k in keys}
    if settings["cost_class_type"] != "focal_loss_cost":
        raise ValueError("Expected native focal Hungarian matcher")
    return settings


def detached_matcher(settings):
    """Native detrex matcher arithmetic, usable without importing CUDA extensions.

    Both IoU denominators retain this repository's 1e-6, not torchvision's
    default. Every calibrated assignment is checked against the native capture.
    """
    from scipy.optimize import linear_sum_assignment

    if settings["cost_class_type"] != "focal_loss_cost":
        raise ValueError("Only focal matching is supported")

    def xyxy(boxes):
        x, y, w, h = boxes.unbind(-1)
        if (w < 0).any() or (h < 0).any():
            raise ValueError("Negative box extent")
        return torch.stack((x-.5*w, y-.5*h, x+.5*w, y+.5*h), -1)

    @torch.no_grad()
    def match(outputs, targets):
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        finite(logits, "matching logits")
        finite(boxes, "matching boxes")
        bs, nq = logits.shape[:2]
        sizes = [len(t["labels"]) for t in targets]
        if not sum(sizes):
            return [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)) for _ in targets]
        prob = logits.flatten(0, 1).sigmoid()
        ids = torch.cat([t["labels"] for t in targets])
        tb = torch.cat([t["boxes"] for t in targets])
        b = boxes.flatten(0, 1)
        a, g = settings["alpha"], settings["gamma"]
        neg = (1-a)*prob**g * -(1-prob+1e-8).log()
        pos = a*(1-prob)**g * -(prob+1e-8).log()
        u, v = xyxy(b), xyxy(tb)
        wh = (torch.minimum(u[:, None, 2:], v[:, 2:]) - torch.maximum(u[:, None, :2], v[:, :2])).clamp(min=0)
        inter = wh[..., 0]*wh[..., 1]
        area_u = (u[:, 2]-u[:, 0])*(u[:, 3]-u[:, 1])
        area_v = (v[:, 2]-v[:, 0])*(v[:, 3]-v[:, 1])
        union = area_u[:, None]+area_v-inter
        wh = (torch.maximum(u[:, None, 2:], v[:, 2:]) - torch.minimum(u[:, None, :2], v[:, :2])).clamp(min=0)
        area = wh[..., 0]*wh[..., 1]
        giou = inter/(union+1e-6) - (area-union)/(area+1e-6)
        cost = settings["cost_bbox"]*torch.cdist(b, tb, p=1) + settings["cost_class"]*(pos-neg)[:, ids] - settings["cost_giou"]*giou
        finite(cost, "Hungarian cost")
        cost = cost.view(bs, nq, -1).cpu()
        return [(torch.as_tensor(q, dtype=torch.long), torch.as_tensor(t, dtype=torch.long))
                for q, t in (linear_sum_assignment(c[i].numpy()) for i, c in enumerate(cost.split(sizes, -1)))]
    return match


def finite(tensor, label):
    if not torch.is_tensor(tensor) or not torch.isfinite(tensor).all():
        raise ValueError(f"Missing/nonfinite {label}")


def close_tensor(actual, expected, label, atol=2e-4, rtol=2e-4):
    finite(actual, label)
    finite(expected, label)
    if actual.shape != expected.shape or not torch.allclose(actual, expected.to(actual), atol=atol, rtol=rtol):
        raise ValueError(f"Native replay failed: {label}")


def clone_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: clone_cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clone_cpu(v) for v in value]
    return value


def formula(similarity, variant, scale, tau, bias=0.):
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(tau) or tau <= 0:
        raise ValueError("Expected finite positive classification scale/temperature")
    k = similarity.shape[-1]
    if k < 1:
        raise ValueError("Empty prototype bank")
    if variant == "legacy":
        logits = torch.logsumexp(scale * similarity, dim=-1)
    elif variant in ("calibrated", "calibrated_plus_logK"):
        logits = scale * tau * (torch.logsumexp(similarity / tau, dim=-1) - math.log(k))
        if variant == "calibrated_plus_logK":
            logits = logits + math.log(k)
    else:
        raise ValueError(f"Unknown formula: {variant}")
    return logits + bias


def assignment_signature(indices):
    return [sorted(zip(t.detach().cpu().tolist(), q.detach().cpu().tolist())) for q, t in indices]


def assignment_change(reference, candidate):
    a, b = assignment_signature(reference), assignment_signature(candidate)
    if len(a) != len(b):
        raise ValueError("Assignment batch mismatch")
    changed = total = 0
    for old, new in zip(a, b):
        if [x[0] for x in old] != [x[0] for x in new]:
            raise ValueError("Matching changed the set of covered GT; unsupported probe")
        total += len(old)
        changed += sum(x != y for x, y in zip(old, new))
    return {"gt_assignments": total, "changed": changed,
            "changed_fraction": changed/total if total else None}


def targets_for(logits, targets, indices):
    result = torch.zeros_like(logits)
    if len(targets) != logits.shape[0] or len(indices) != len(targets):
        raise ValueError("Label/matching batch shape mismatch")
    for i, (queries, gt) in enumerate(indices):
        queries, gt = queries.to(logits.device).long(), gt.to(logits.device).long()
        labels = targets[i]["labels"].to(logits.device).long()
        if len(queries) != len(gt) or len(queries.unique()) != len(queries):
            raise ValueError("Invalid/repeated matched query")
        if len(queries) and (queries.min() < 0 or queries.max() >= logits.shape[1]
                            or gt.min() < 0 or gt.max() >= len(labels)):
            raise ValueError("Assignment index outside tensors")
        if len(labels) and (labels.min() < 0 or labels.max() >= logits.shape[-1]):
            raise ValueError("GT label outside FedLoss subset")
        result[i, queries, labels[gt]] = 1.
    return result


def focal_cells(logits, labels, alpha, gamma, normalizer, weight):
    if not 0 <= alpha <= 1 or gamma < 0 or normalizer <= 0 or weight <= 0:
        raise ValueError("Invalid native focal settings")
    p = logits.sigmoid()
    pt = p*labels + (1-p)*(1-labels)
    # Native SetCriterion's mean(query)*num_queries equals sum(cells).
    return F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * (
        1-pt)**gamma * (alpha*labels + (1-alpha)*(1-labels)) * weight/normalizer


def direction_cosine(a, b):
    an, bn = a.double().norm(), b.double().norm()
    return float((a.double()*b.double()).sum()/(an*bn)) if an > 1e-12 and bn > 1e-12 else None


def gradient_probe(x, p, logits, labels, options):
    """Partial derivatives to detached head inputs and raw output slots ONLY.

    Includes normalization of raw features/prototypes. Not gradients to TPA
    parameters, decoder weights, APR, RPSA, or an AdamW/clipped update.
    """
    cell_loss = focal_cells(logits, labels, **options)
    result, vectors = {}, {}
    for label, mask in (("positive", labels.bool()), ("negative", ~labels.bool())):
        loss = cell_loss[mask].sum()
        gz, gx, gp = torch.autograd.grad(loss, (logits, x, p), retain_graph=True)
        for g in (gz, gx, gp):
            finite(g, label + " gradient")
        count = int(mask.sum())
        slot_norm = gp.double().norm(dim=-1)  # [sampled categories, K]
        denominator = slot_norm.sum(dim=-1, keepdim=True)
        shares = slot_norm/denominator.clamp_min(1e-30)
        result[label] = {
            "cells": count, "weighted_loss": float(loss.detach()),
            "mean_probability": float(logits.detach().sigmoid()[mask].mean()) if count else None,
            "mean_abs_logit_gradient": float(gz.detach()[mask].abs().mean()) if count else None,
            "logit_gradient_l2": float(gz.double().norm()),
            "feature_gradient_l2": float(gx.double().norm()),
            "prototype_output_gradient_l2": float(gp.double().norm()),
            "per_category_slot_gradient_l2": slot_norm.detach().cpu().tolist(),
            "per_category_slot_share": shares.detach().cpu().tolist(),
            "category_has_nonzero_slot_gradient": (denominator[:, 0] > 1e-30).cpu().tolist(),
        }
        vectors[label] = (gx.detach(), gp.detach())
    gx = vectors["positive"][0] + vectors["negative"][0]
    gp = vectors["positive"][1] + vectors["negative"][1]
    result["total"] = {"feature_gradient_l2": float(gx.double().norm()),
                       "prototype_output_gradient_l2": float(gp.double().norm()),
                       "weighted_loss": sum(result[s]["weighted_loss"] for s in ("positive", "negative"))}
    vectors["total"] = (gx, gp)
    return result, vectors


def audit_branch(record, matcher, device="cpu"):
    x = record["features"].to(device=device, dtype=torch.float32).detach().requires_grad_(True)
    p = record["prototypes"].to(device=device, dtype=torch.float32).detach().requires_grad_(True)
    if x.ndim != 3 or p.ndim != 3 or x.shape[-1] != p.shape[-1]:
        raise ValueError("Invalid feature/prototype shape")
    finite(x, "features")
    finite(p, "prototypes")
    sim = torch.einsum("bqd,ckd->bqck", F.normalize(x, dim=-1), F.normalize(p, dim=-1))
    targets = [{k: v.to(device) for k, v in t.items()} for t in record["targets"]]
    boxes = record["boxes"].to(device)
    indices = [(q.to(device), t.to(device)) for q, t in record["indices"]]
    logits = {v: formula(sim, v, record["scale"], record["tau"], record["bias"]) for v in VARIANTS}
    close_tensor(logits["calibrated"].detach(), record["logits"].to(device), record["name"] + " logits")
    native = targets_for(logits["calibrated"], targets, indices)
    options = {k: record[k] for k in ("alpha", "gamma", "normalizer", "weight")}
    expected_loss = float(focal_cells(logits["calibrated"], native, **options).detach().sum())
    if not math.isclose(expected_loss, record["weighted_native_loss"], rel_tol=3e-4, abs_tol=3e-4):
        raise ValueError("Native focal loss replay failed: " + record["name"])
    rematches = {}
    with torch.no_grad():
        for variant in VARIANTS:
            rematches[variant] = (matcher({"pred_logits": logits[variant].detach(), "pred_boxes": boxes}, targets)
                                  if record["hungarian"] else indices)
    if assignment_change(indices, rematches["calibrated"])["changed"]:
        raise ValueError("Native Hungarian replay mismatch; stopping instead of attributing numerical ties")
    result = {"name": record["name"], "group": record["group"], "hungarian": record["hungarian"],
              "queries": x.shape[1], "categories": p.shape[0], "slots": p.shape[1],
              "normalizer": record["normalizer"], "weight": record["weight"], "variants": {},
              "native_replay": {"logit_max_abs_error": float((logits["calibrated"].detach()-record["logits"].to(device)).abs().max()),
                                "weighted_loss_abs_error": abs(expected_loss-record["weighted_native_loss"]),
                                "matching_identical": True}}
    reference_vectors = {}
    for variant in VARIANTS:
        fixed, vectors = gradient_probe(x, p, logits[variant], native, options)
        change = assignment_change(indices, rematches[variant])
        if variant == "calibrated":
            reference_vectors = vectors
        alignment = {s: {"feature": direction_cosine(reference_vectors[s][0], vectors[s][0]),
                         "prototype_output": direction_cosine(reference_vectors[s][1], vectors[s][1])}
                     for s in vectors}
        if change["changed"]:
            labels = targets_for(logits[variant], targets, rematches[variant])
            rematched, _ = gradient_probe(x, p, logits[variant], labels, options)
        else:
            rematched = fixed
        result["variants"][variant] = {"fixed_assignment": fixed, "rematched": rematched,
                                       "matching": change, "fixed_gradient_cosine_vs_calibrated": alignment}
    return result


@contextmanager
def capture_head_inputs(model):
    """Observe native forward without altering values, matching, RNG or losses."""
    features, calls, sampled = {}, [], {}
    originals = []
    layers = model.transformer.decoder.num_layers
    original_filter = model.filter_content_info
    original_labels = model.criterion.loss_labels

    def record_filter(data):
        indices, mapped = original_filter(data)
        sampled["category_indices"] = indices.detach().cpu().tolist()
        return indices, mapped

    def record_labels(outputs, targets, indices, num_boxes):
        result = original_labels(outputs, targets, indices, num_boxes)
        calls.append({"logits": clone_cpu(outputs["pred_logits"]), "boxes": clone_cpu(outputs["pred_boxes"]),
                      "targets": clone_cpu(targets), "indices": clone_cpu(indices),
                      "normalizer": float(num_boxes), "raw_native_loss": float(result["loss_class"].detach())})
        return result

    def wrap(index, classifier, original):
        def compute(x, *, content_inds, additional_class):
            output = original(x, content_inds=content_inds, additional_class=additional_class)
            # The encoder's first call covers spatial tokens, not criterion queries.
            # Retain only its selected-proposal call to avoid a huge CPU cache.
            if index != layers or x.shape[1] == model.num_queries:
                if additional_class is not None or not classifier.norm_weight or classifier._external_prototypes is None:
                    raise ValueError("Expected normalized shared-TPA classifier without extra classes")
                if content_inds is None or content_inds.detach().cpu().tolist() != sampled["category_indices"]:
                    raise ValueError("Classifier and FedLoss categories differ")
                features[index] = {"features": clone_cpu(x),
                                   "prototypes": clone_cpu(classifier._external_prototypes),
                                   "scale": float(classifier.norm_temperature), "tau": float(classifier.tpa_cls_tau),
                                   "bias": float(classifier.cls_bias.detach()) if classifier.use_bias else 0.}
            return output
        return compute

    try:
        for index, classifier in enumerate(model.class_embed):
            original = classifier._compute_tpa_logits
            originals.append((classifier, original))
            classifier._compute_tpa_logits = wrap(index, classifier, original)
        model.filter_content_info = record_filter
        model.criterion.loss_labels = record_labels
        yield (features, calls, sampled)
    finally:
        model.filter_content_info = original_filter
        model.criterion.loss_labels = original_labels
        for classifier, original in originals:
            classifier._compute_tpa_logits = original


def assemble_records(model, captured, losses):
    features, calls, sampled = captured
    n = model.transformer.decoder.num_layers
    # Actual DINOCriterion order: final, auxiliaries, encoder, DN final, DN auxiliaries.
    specs = [("loss_class", n-1, "final", False)]
    specs += [(f"loss_class_{i}", i, "auxiliary", False) for i in range(n-1)]
    specs += [("loss_class_enc", n, "encoder", False)]
    if len(calls) == 2*n+1:
        specs += [("loss_class_dn", n-1, "denoising", True)]
        specs += [(f"loss_class_dn_{i}", i, "denoising", True) for i in range(n-1)]
    elif len(calls) != n+1:
        raise ValueError("Unexpected native classification call count")
    records = []
    for (name, layer, group, dn), call in zip(specs, calls):
        head = features[layer]
        q = call["logits"].shape[1]
        x = head["features"]
        if layer == n:
            if x.shape[1] != q:
                raise ValueError("Encoder criterion features not captured")
        else:
            padding = x.shape[1] - model.num_queries
            if padding < 0 or q != (padding if dn else model.num_queries):
                raise ValueError("DN/normal query slices differ from native layout")
            x = x[:, :padding] if dn else x[:, padding:]
        weight = float(model.criterion.weight_dict[name])
        actual = float(losses[name].detach())
        if not math.isclose(actual, call["raw_native_loss"]*weight, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("Native classification branch/weight order changed")
        records.append({**head, **call, "features": x.contiguous(), "name": name, "group": group,
                        "hungarian": not dn, "weight": weight, "weighted_native_loss": actual,
                        "alpha": float(model.criterion.alpha), "gamma": float(model.criterion.gamma)})
    if not records or not sampled.get("category_indices"):
        raise ValueError("Missing native classification/FedLoss records")
    return records


def stage_summary(batches):
    """Equal batch/branch averages, NOT a norm of the joint model gradient."""
    result = {}
    for group in ("all", "final", "auxiliary", "encoder", "denoising"):
        entries = [r for b in batches for r in b["branches"] if group == "all" or r["group"] == group]
        if not entries:
            result[group] = None
            continue
        variants = {}
        for v in VARIANTS:
            values = {}
            for assignment in ("fixed_assignment", "rematched"):
                for label in ("positive", "negative"):
                    for metric in ("mean_probability", "mean_abs_logit_gradient", "prototype_output_gradient_l2", "feature_gradient_l2"):
                        numbers = [r["variants"][v][assignment][label][metric] for r in entries
                                   if r["variants"][v][assignment][label][metric] is not None]
                        values[f"{assignment}/{label}/{metric}"] = mean(numbers) if numbers else None
            eligible = [r["variants"][v]["matching"] for r in entries if r["hungarian"]]
            denominator = sum(r["gt_assignments"] for r in eligible)
            values["rematched_GT_fraction"] = sum(r["changed"] for r in eligible)/denominator if denominator else None
            variants[v] = values
        result[group] = {"branch_batches": len(entries), "variants": variants}
    return result


def compare_stages(stages):
    summaries = {s: stage_summary(stages[s]) for s in ("8ep", "12ep")}
    effects = {}
    for group, a in summaries["8ep"].items():
        b = summaries["12ep"][group]
        if a is None or b is None:
            effects[group] = None
            continue
        effects[group] = {}
        for variant in VARIANTS[1:]:
            metrics = {}
            for key in a["variants"][variant]:
                x0, x1 = (a["variants"][v][key] for v in ("calibrated", variant))
                y0, y1 = (b["variants"][v][key] for v in ("calibrated", variant))
                metrics[key] = None if any(v is None for v in (x0, x1, y0, y1)) else {
                    "effect_at_8ep": x1-x0, "effect_at_12ep": y1-y0,
                    "stage_difference_of_formula_effect": (y1-y0)-(x1-x0)}
            effects[group][variant] = metrics
    return {"stage_summaries": summaries, "formula_stage_interactions": effects,
            "summary_weighting": "Equal captured batch/branch; not the norm of summed parameter gradients."}
