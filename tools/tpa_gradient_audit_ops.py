"""Local gradient directions and text-side JVPs; never applies optimizer updates."""

from __future__ import annotations

import math
from statistics import mean

import torch
import torch.nn.functional as F

from lami_dino.prototype_ops import route_conflicting_task_gradient
from tools.tpa_geometry_audit_ops import WEIGHTS


SCALARS = ("pre_spread_ratio", "center_shift_ratio", "pre_unit_mean_norm",
           "post_unit_mean_norm", "post_pairwise_cos")
CENTERS = ("value_center", "pre_slot_center", "post_slot_center")
LOGITS = ("center_response", "mean_slot_response", "dispersion_uplift", "native_logit")


def loss_gradients(loss_dict, parameters):
    """Differentiate weighted native losses once per group, on the SAME graph.

    RPSA remains part of task for trainer-equivalent routing, but is also
    reported separately. Diagnostic non-loss keys must never enter the sum.
    """
    if not isinstance(loss_dict, dict) or "loss_apr" not in loss_dict:
        raise ValueError("Native forward must return loss_apr and detector losses")
    losses = {k: v for k, v in loss_dict.items() if k.startswith("loss")}
    if any(not torch.is_tensor(v) or v.numel() != 1 or not torch.isfinite(v).all() for v in losses.values()):
        raise ValueError("All native losses must be finite scalar tensors")
    detector_keys = [k for k in losses if k not in ("loss_apr", "loss_rpsa")]
    if not detector_keys:
        raise ValueError("Missing detector losses")
    zero = losses["loss_apr"].new_zeros(())
    components = {"detector": sum(losses[k] for k in detector_keys),
                  "rpsa": losses.get("loss_rpsa", zero), "apr": losses["loss_apr"]}
    gradients, unused = {}, {}
    for index, (name, loss) in enumerate(components.items()):
        grad = (torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=index < 2)
                if loss.requires_grad else (None,) * len(parameters))
        unused[name] = [i for i, g in enumerate(grad) if g is None]
        gradients[name] = torch.cat([(g.detach() if g is not None else torch.zeros_like(p)).flatten()
                                     for g, p in zip(grad, parameters)])
        if not torch.isfinite(gradients[name]).all():
            raise ValueError(f"Nonfinite {name} gradient")
    if unused["apr"] == list(range(len(parameters))):
        raise ValueError("APR is disconnected from TPA")
    return gradients, {k: float(v.detach()) for k, v in losses.items()}, unused


def gradient_directions(gradients, max_norm):
    """Route ONLY after accumulating/averaging micro-batch gradients."""
    for g in gradients.values():
        if g.ndim != 1 or not torch.isfinite(g).all():
            raise ValueError("Expected finite flat gradients")
    if len({g.shape for g in gradients.values()}) != 1:
        raise ValueError("Component gradient shapes differ")
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("Clip max norm must be positive")
    task = gradients["detector"] + gradients["rpsa"]
    total = task + gradients["apr"]
    routed, stats = route_conflicting_task_gradient(total, gradients["apr"])
    directions = {**gradients, "task": task, "unprojected_total": total,
                  "projected_task": routed - gradients["apr"],
                  "routed_total": routed, "projection_added": routed - total}
    summaries = {}
    for name, g in directions.items():
        norm = float(g.double().norm())
        unit_descent = -g.double() / norm if norm > 1e-12 else torch.zeros_like(g, dtype=torch.float64)
        summaries[name] = {"norm": norm, "zero_direction": norm <= 1e-12,
                           "clip_coefficient_if_used_alone": min(1., max_norm / (norm + 1e-6)),
                           "loss_derivative_per_unit_descent": {
                               k: float(torch.dot(v.double(), unit_descent))
                               for k, v in {**gradients, "task": task}.items()}}
    return directions, {"routing": {k: float(v) for k, v in stats.items()}, "directions": summaries}


def text_observables(weights, buffers, prompts, tau, features, class_indices, cls_tau, logit_scale, bias,
                     *, query_only=False):
    """Differentiable eval TPA, including the existing fixed-radius transform.

    No detector graph here. Validation-query features are constants and do NOT
    contribute to the training gradient. Float64 supports stable small JVPs.
    """
    q, kw, kb, vw, vb = weights
    values, keys = F.linear(prompts, vw, vb), F.linear(prompts, kw, kb)
    logits = torch.einsum("kh,cnh->ckn", q, keys)
    positions = torch.linspace(0, prompts.shape[1] - 1, steps=len(q), device=prompts.device).round().long()
    prior = logits.new_zeros((len(q), prompts.shape[1]))
    prior.scatter_(1, positions[:, None], buffers["slot_prior_strength"].expand(len(q), 1))
    attention = ((logits + prior[None]) / (math.sqrt(q.shape[1]) * tau)).softmax(-1)
    before = torch.einsum("ckn,cnd->ckd", attention, values)
    c = values.mean(1, keepdim=True)
    strength = buffers["prototype_mode_strength"]
    radius_enabled = bool(strength > 0)
    # Strength is a fixed buffer, NOT a differentiated parameter. Avoid the
    # disabled branch: 0 * an undefined norm double-backward is still NaN.
    if radius_enabled:
        residual = before - c
        unit_residual = residual / residual.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        fixed = c + strength * c.norm(dim=-1, keepdim=True).clamp_min(1e-6) * unit_residual
        after = before + (fixed - before)
    else:
        after = before
    centers = torch.stack((c[:, 0], before.mean(1), after.mean(1)), dim=1)
    post_unit = F.normalize(after, dim=-1)
    k = after.shape[1]
    x = F.normalize(features, dim=-1)
    slots = post_unit[class_indices]
    cosines = (slots * x[:, None]).sum(-1)
    center_score = logit_scale * (F.normalize(centers[class_indices, 2], dim=-1) * x).sum(-1)
    mean_score = logit_scale * cosines.mean(-1)
    native = logit_scale * cls_tau * (torch.logsumexp(cosines / cls_tau, dim=-1) - math.log(k)) + bias
    query_logits = torch.stack((center_score, mean_score, native - bias - mean_score, native), dim=-1)
    if query_only:
        # Ranking audits need logits, not geometric norm derivatives. Do not
        # let an unrelated, nonsmooth geometry diagnostic poison their JVP.
        return query_logits
    pre_unit = F.normalize(before, dim=-1)
    gram = post_unit @ post_unit.transpose(-1, -2)
    cosine = ((gram.sum((-1, -2)) - gram.diagonal(dim1=-2, dim2=-1).sum(-1)) / (k * (k - 1))
              if k > 1 else after.sum((-1, -2)) * 0)
    c_norm = c[:, 0].norm(dim=-1).clamp_min(1e-6)
    # For no-radius the shift is identically zero for ALL parameter values.
    # Its derivative is exactly zero; differentiating norm(0) with PyTorch
    # 1.12's divide-then-mask norm backward needlessly creates 0/0 in JVP.
    shift = ((centers[:, 2] - centers[:, 1]).norm(dim=-1) / c_norm if radius_enabled
             else before.new_zeros(before.shape[0]))
    scalars = torch.stack(((before - c).flatten(1).norm(dim=-1) / math.sqrt(k) / c_norm,
                           shift, pre_unit.mean(1).norm(dim=-1), post_unit.mean(1).norm(dim=-1), cosine), dim=-1)
    return centers, scalars, query_logits


def parameter_block_norms(flat, state):
    output, offset = {}, 0
    for name in WEIGHTS:
        n = state[name].numel()
        output[name] = float(flat[offset:offset + n].double().norm())
        offset += n
    if offset != flat.numel():
        raise ValueError("Flat gradient size differs from TPA parameter layout")
    return output


def direction_jvp(state, prompts, bank, query_features, query_classes, gradient, *, query_only=False):
    """J[ -g/||g|| ] at the unchanged checkpoint; no parameter assignment."""
    weights = tuple(state[name].detach().cpu().double() for name in WEIGHTS)
    buffers = {k: state[k].detach().cpu().double() for k in ("slot_prior_strength", "prototype_mode_strength")}
    gradient = gradient.detach().cpu().double()
    if not torch.isfinite(gradient).all():
        raise ValueError("Nonfinite direction")
    norm = float(gradient.norm())
    unit = -gradient / norm if norm > 1e-12 else gradient * 0
    tangents, offset = [], 0
    for weight in weights:
        n = weight.numel()
        tangents.append(unit[offset:offset + n].view_as(weight))
        offset += n
    if offset != len(gradient):
        raise ValueError("Gradient length differs from TPA weights")
    def function(*args):
        result = text_observables(args, buffers, prompts.cpu().double(), bank["tpa_tau"],
                                  query_features.cpu().double(), query_classes.cpu(),
                                  bank["temperature"], bank["logit_scale"], bank["cls_bias"],
                                  query_only=query_only)
        return (result,) if query_only else result
    if norm <= 1e-12:
        with torch.no_grad():
            values = function(*weights)
        derivatives = tuple(torch.zeros_like(v) for v in values)
    else:
        values, derivatives = torch.autograd.functional.jvp(function, weights, tuple(tangents), create_graph=False)
    labels = ("query_logits",) if query_only else ("centers", "scalars", "query_logits")
    invalid = [f"{kind}.{label}={int((~torch.isfinite(x)).sum())}/{x.numel()}"
               for kind, tensors in (("values", values), ("derivatives", derivatives))
               for label, x in zip(labels, tensors) if not torch.isfinite(x).all()]
    if invalid:
        raise ValueError("Nonfinite text-side JVP: " + ", ".join(invalid))
    if query_only:
        return {"query_logits": values[0].tolist(), "query_logit_derivatives": derivatives[0].tolist()}
    centers, scalars, query = values
    dc, ds, dq = derivatives
    center_norms = centers.norm(dim=-1)
    unit_centers = F.normalize(centers, dim=-1)
    radial = (dc * unit_centers).sum(-1)
    angular = ((dc - radial[..., None] * unit_centers).norm(dim=-1)
               / center_norms.clamp_min(1e-12)) * (180 / math.pi)
    return {"values": scalars.tolist(), "scalar_derivatives": ds.tolist(),
            "center_norm_derivatives": radial.tolist(),
            "center_angular_speed_deg": angular.tolist(),
            "center_direction_defined": (center_norms > 1e-12).tolist(),
            "query_logits": query.tolist(), "query_logit_derivatives": dq.tolist()}


def split_jvp_summary(result, categories, query_records):
    summary = {}
    for split in ("all", "r", "c", "f"):
        indices = [i for i, c in enumerate(categories) if split == "all" or c["frequency"] == split]
        summary[split] = {"classes": len(indices)}
        if indices:
            summary[split]["scalar_derivative_mean"] = {
                key: mean(result["scalar_derivatives"][i][j] for i in indices) for j, key in enumerate(SCALARS)}
            summary[split]["center_angular_speed_mean_deg"] = {
                key: mean(vals) if (vals := [result["center_angular_speed_deg"][i][j] for i in indices
                                             if result["center_direction_defined"][i][j]]) else None
                for j, key in enumerate(CENTERS)}
    query_summary = []
    for category, kind in sorted({(r["category"], r["kind"]) for r in query_records}):
        indices = [i for i, r in enumerate(query_records) if (r["category"], r["kind"]) == (category, kind)]
        means = {key: mean(result["query_logit_derivatives"][i][j] for i in indices) for j, key in enumerate(LOGITS)}
        means["center_to_mean_correction"] = means["mean_slot_response"] - means["center_response"]
        query_summary.append({"category": category, "kind": kind, "queries": len(indices), "derivative_mean": means})
    return summary, query_summary
