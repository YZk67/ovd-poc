"""Endpoint module interventions, not loss attribution or optimizer replay."""

from __future__ import annotations

from contextlib import contextmanager
import math
import re

import torch
import torch.nn.functional as F

from tools.rare_stage_update_ops import grouped_margins, selected_logits


GROUPS = ("visual_adapter", "encoder_memory", "proposal_head", "query_content",
          "decoder_core", "decoder_box", "final_projection", "upstream_tpa")


def canonical_name(name):
    if name.startswith("module."):
        name = name[7:]
    name = re.sub(r"^transformer\.decoder\.(class_embed|bbox_embed)\.", r"\1.", name)
    return re.sub(r"^class_embed\.\d+\.tpa\.", "class_embed.0.tpa.", name)


def canonical_state(state):
    result, aliases = {}, {}
    for name, value in state.items():
        if not torch.is_tensor(value):
            raise ValueError(f"Non-tensor model state: {name}")
        key = canonical_name(name)
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite checkpoint tensor: {name}")
        if key in result:
            previous = result[key]
            if (value.shape != previous.shape or value.dtype != previous.dtype
                    or not torch.equal(value, previous)):
                raise ValueError(f"Conflicting shared aliases: {name}")
        else:
            result[key] = value
        aliases.setdefault(key, []).append(name)
    return result, aliases


def key_group(key, decoder_layers=6):
    if ".tpa." in key:
        if key.endswith("._step"):
            return "training_only"
        if key.endswith((".slot_prior_strength", ".prototype_mode_strength")):
            return "fixed_protocol"
        return "upstream_tpa"
    if key.startswith(("backbone.downsample_layers.", "backbone.stages.", "identical.", "thead.", "head.")):
        return "frozen_clip"
    if key.startswith(("backbone.", "neck.", "position_embedding.")):
        return "visual_adapter"
    if key.startswith(("transformer.encoder.", "transformer.enc_output.", "transformer.enc_output_norm.")) or key == "transformer.level_embeds":
        return "encoder_memory"
    if key.startswith("content_layer."):
        return "query_content"
    if key.startswith(("transformer.decoder.layers.", "transformer.decoder.norm.", "transformer.decoder.ref_point_head.")):
        return "decoder_core"
    match = re.fullmatch(r"(class_embed|bbox_embed)\.(\d+)\.(.+)", key)
    if match:
        head, index, tail = match.groups()
        index = int(index)
        if index > decoder_layers:
            raise ValueError(f"Unexpected head index: {key}")
        if index == decoder_layers:
            return "proposal_head"
        if head == "bbox_embed":
            return "decoder_box"
        if index == decoder_layers - 1:
            if tail.startswith("linear."):
                return "final_projection"
            if tail == "cls_bias":
                return "final_bias"
        elif tail.startswith("linear.") or tail == "cls_bias":
            return "auxiliary_classifier"
    if key.startswith(("criterion.", "transformer.rpsa.", "label_enc.")) or key == "freq_weight":
        return "training_only"
    return "unclassified"


def inventory(old, new):
    """Assign every CHANGED tensor; norms count shared parameters only once."""
    a, _ = canonical_state(old)
    b, _ = canonical_state(new)
    if a.keys() != b.keys():
        raise ValueError("Endpoint state keys differ")
    groups = {}
    for key in sorted(a):
        x, y = a[key], b[key]
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError(f"Endpoint shape/dtype differs: {key}")
        changed = not torch.equal(x, y)
        group = key_group(key)
        if changed and group in ("unclassified", "fixed_protocol", "frozen_clip"):
            raise ValueError(f"Unsupported changed {group} tensor: {key}")
        row = groups.setdefault(group, {"keys": [], "changed_keys": [], "numel": 0,
                                        "old_sq_norm": 0., "delta_sq_norm": 0.})
        row["keys"].append(key)
        row["numel"] += x.numel()
        if changed:
            row["changed_keys"].append(key)
        row["old_sq_norm"] += float(x.double().square().sum())
        row["delta_sq_norm"] += float((y.double() - x.double()).square().sum())
    for row in groups.values():
        row["delta_l2"] = math.sqrt(row.pop("delta_sq_norm"))
        norm = math.sqrt(row.pop("old_sq_norm"))
        row["relative_delta_l2"] = row["delta_l2"] / norm if norm else None
    return groups


def clear_eval_caches(model):
    for module in model.modules():
        for name in ("_cached_eval", "_external_prototypes", "_external_apr_loss"):
            if hasattr(module, name):
                setattr(module, name, None)


@contextmanager
def swap_group(model, donor, group):
    """In-memory endpoint replacement, including every shared-state alias; undo on failure."""
    chosen = set(GROUPS) if group == "all_query" else {group}
    source, _ = canonical_state({k: v for k, v in donor.items() if key_group(canonical_name(k)) in chosen})
    live = model.state_dict()
    keys = [k for k in live if key_group(canonical_name(k)) in chosen]
    if not keys:
        raise ValueError(f"Model has no group: {group}")
    saved = {k: live[k].detach().clone() for k in keys}
    try:
        with torch.no_grad():
            for key in keys:
                value = source[canonical_name(key)]
                if value.shape != live[key].shape or value.dtype != live[key].dtype:
                    raise ValueError(f"Donor shape/dtype differs: {key}")
                live[key].copy_(value)
        clear_eval_caches(model)
        yield
    finally:
        with torch.no_grad():
            for key, value in saved.items():
                live[key].copy_(value)
        clear_eval_caches(model)


def box_iou(a, b):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 4 or b.shape[1] != 4:
        raise ValueError("Expected Nx4 boxes")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Nonfinite boxes")
    wh = (torch.minimum(a[:, None, 2:], b[None, :, 2:])
          - torch.maximum(a[:, None, :2], b[None, :, :2])).clamp_min(0)
    inter = wh.prod(-1)
    aa = (a[:, 2:] - a[:, :2]).clamp_min(0).prod(-1)
    bb = (b[:, 2:] - b[:, :2]).clamp_min(0).prod(-1)
    return inter / (aa[:, None] + bb[None] - inter).clamp_min(1e-12)


def match_regions(records, candidate_boxes, threshold=.5):
    """Geometry-only, maximum-cardinality then maximum-IoU one-to-one matching.

    Repeated anchors with the SAME native query reuse its match; distinct native
    queries cannot silently share a hybrid query. No class scores or GT boxes.
    """
    from scipy.optimize import linear_sum_assignment

    if not 0 < threshold <= 1:
        raise ValueError("Invalid matching threshold")
    anchors = {}
    for row in records:
        q = row["query_id"]
        if q in anchors and anchors[q] != row["box_xyxy"]:
            raise ValueError("Same native query has inconsistent boxes")
        anchors[q] = row["box_xyxy"]
    if not anchors:
        return []
    ids = list(anchors)
    overlaps = box_iou(torch.tensor(list(anchors.values()), dtype=torch.float64), candidate_boxes.double())
    n, m = overlaps.shape
    # One dummy per anchor. n+1 ensures cardinality wins over any IoU tradeoff.
    utility = torch.zeros(n, m + n, dtype=torch.float64)
    utility[:, :m] = torch.where(overlaps >= threshold, n + 1 + overlaps, -torch.ones_like(overlaps))
    rr, cc = linear_sum_assignment(-utility.numpy())
    matches = {ids[i]: ({"query_id": int(j), "iou": float(overlaps[i, j])} if j < m else None)
               for i, j in zip(rr, cc)}
    # Do not turn indistinguishable candidate/anchor ties into score evidence.
    # A deterministic Hungarian tie break is not a verified query identity.
    anchor_boxes = torch.tensor(list(anchors.values()), dtype=torch.float64)
    for i, key in enumerate(ids):
        tied_anchors = (anchor_boxes - anchor_boxes[i]).abs().amax(-1) <= 1e-7
        tied_candidates = False
        if m > 1:
            top = overlaps[i].topk(2).values
            tied_candidates = bool(top[0] >= threshold and (top[0] - top[1]).abs() <= 1e-7)
        if int(tied_anchors.sum()) > 1 or tied_candidates:
            matches[key] = None
    return [matches[row["query_id"]] for row in records]


def region_effects(records, native, hybrid, bank, indices, *, threshold=.5, beta=.3):
    """Frozen final bank/bias and frozen CLIP: only matched query features change."""
    matches = match_regions(records, hybrid["query_boxes"], threshold)
    selected = [i for i, match in enumerate(matches) if match is not None]
    old_q = torch.tensor([r["query_id"] for r in records], dtype=torch.long)
    z0 = selected_logits(native["features"][old_q], bank["prototypes"], indices, bank)
    changes = torch.zeros(len(records), dtype=torch.float64)
    feature_cos = [None] * len(records)
    if selected:
        hybrid_q = torch.tensor([matches[i]["query_id"] for i in selected])
        x = hybrid["features"][hybrid_q]
        z1 = selected_logits(x, bank["prototypes"], indices[selected], bank)
        changes[selected] = (1 - beta) * (F.logsigmoid(z1) - F.logsigmoid(z0[selected]))
        cos = F.cosine_similarity(native["features"][old_q[selected]].double(), x.double(), dim=-1)
        for i, v in zip(selected, cos.tolist()):
            feature_cos[i] = v
    panel = [records[i] for i in selected]
    margins = grouped_margins(changes[selected], panel)
    counts = grouped_margins(torch.zeros(len(records)), records)
    for name, count in counts.items():
        value = margins.setdefault(name, {"tp_regions": 0, "fp_regions": 0, "pairs": 0, "mean_tp_minus_fp": None})
        value["all_regions_matched"] = (value["tp_regions"] == count["tp_regions"] and value["fp_regions"] == count["fp_regions"])
        value["full_panel_margin_change"] = value["mean_tp_minus_fp"] if value["all_regions_matched"] else None
    return {"margin_change": margins, "matched": len(selected), "total": len(records),
            "regions": [{**r, "match": m, "feature_cosine": c,
                         "delta_fused_log_score": float(changes[i]) if m is not None else None}
                        for i, (r, m, c) in enumerate(zip(records, matches, feature_cos))]}
