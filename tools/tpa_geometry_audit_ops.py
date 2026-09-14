"""CPU reconstruction of the existing TPA forward; no detector or training."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math

import torch
import torch.nn.functional as F

from tools.rare_logit_decomposition_ops import TERMS, decompose_entry, decompose_logit


WEIGHTS = ("prototype_queries", "key_proj.weight", "key_proj.bias",
           "value_proj.weight", "value_proj.bias")
BUFFERS = ("slot_prior_strength", "prototype_mode_strength")


def tensor_digest(tensor):
    """Same identity as the native pairing dump, including shape and dtype."""
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def extract_shared_tpa(checkpoint):
    """Reject unequal aliases instead of silently choosing the wrong head."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must contain a state mapping")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    canonical = {}
    for name, tensor in state.items():
        if not isinstance(name, str) or not torch.is_tensor(tensor):
            continue
        while name.startswith(("module.", "model.")):
            name = name.split(".", 1)[1]
        if name in canonical:
            raise ValueError("Duplicate canonical checkpoint key")
        canonical[name] = tensor
    prefixes = sorted(name[:-len("key_proj.weight")] for name in canonical
                      if name.endswith("tpa.key_proj.weight"))
    if not prefixes:
        raise ValueError("No TPA weights found in checkpoint")
    selected = None
    for prefix in prefixes:
        block = {}
        for name in WEIGHTS + BUFFERS:
            value = canonical.get(prefix + name)
            if value is None and name in BUFFERS:
                value = torch.tensor(0.)  # legacy checkpoint compatibility
            if value is None or not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"Missing/nonfinite TPA tensor: {prefix + name}")
            if selected is not None and (value.shape != selected[name].shape
                                         or value.dtype != selected[name].dtype
                                         or not torch.equal(value.cpu(), selected[name])):
                raise ValueError(f"Unequal shared TPA aliases: {prefix + name}")
            block[name] = value.detach().cpu().clone()
        selected = block
    return selected, {"prefixes": prefixes, "aliases_equal": True,
                      "iteration": checkpoint.get("iteration")}


@torch.no_grad()
def reconstruct_tpa(prompts, state, tau):
    """Exactly the eval arithmetic in TextPrototypeAggregator, before/after radius.

    Slot prior, learned projections and queries remain unchanged. No dropout,
    APR, projection of gradients, or initialization is performed.
    """
    if prompts.ndim != 3 or min(prompts.shape) < 1 or not torch.isfinite(prompts).all():
        raise ValueError("Expected finite nonempty prompts [C,N,D]")
    tau = float(tau)
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("TPA tau must be finite and positive")
    prompts = prompts.detach().cpu().float()
    state = {k: v.detach().cpu().float() for k, v in state.items()}
    q = state["prototype_queries"]
    if q.ndim != 2 or min(q.shape) < 1:
        raise ValueError("Expected prototype queries [K,H]")
    for name in BUFFERS:
        if state[name].numel() != 1 or not torch.isfinite(state[name]).all() or state[name] < 0:
            raise ValueError(f"Invalid scalar {name}")
    keys = F.linear(prompts, state["key_proj.weight"], state["key_proj.bias"])
    values = F.linear(prompts, state["value_proj.weight"], state["value_proj.bias"])
    logits = torch.einsum("kh,cnh->ckn", q, keys)
    positions = torch.linspace(0, prompts.shape[1] - 1, steps=len(q)).round().long()
    prior = logits.new_zeros((len(q), prompts.shape[1]))
    prior.scatter_(1, positions[:, None], state["slot_prior_strength"].expand(len(q), 1))
    attention = ((logits + prior[None]) / (math.sqrt(q.shape[1]) * tau)).softmax(-1)
    before = torch.einsum("ckn,cnd->ckd", attention, values)
    center = values.mean(1, keepdim=True)
    residual = before - center
    unit = residual / residual.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    strength = state["prototype_mode_strength"]
    radius = strength * center.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    fixed = center + radius * unit
    after = before + (strength > 0).to(before.dtype) * (fixed - before)
    if not all(torch.isfinite(x).all() for x in (before, after, center, unit, radius)):
        raise ValueError("Nonfinite TPA reconstruction")
    return {"before": before, "after": after, "value_center": center[:, 0],
            "residual": residual, "unit_residual": unit, "radius": radius[:, 0, 0],
            "enabled": bool(strength > 0)}


def angle_degrees(left, right):
    if min(float(left.norm()), float(right.norm())) <= 1e-12:
        return None
    cosine = float(torch.dot(left / left.norm(), right / right.norm()))
    return math.degrees(math.acos(max(-1., min(1., cosine))))


def stage_geometry(prototypes):
    p = prototypes.double()
    raw_center = p.mean(0)
    unit_slots = F.normalize(p, dim=-1)
    unit_center = unit_slots.mean(0)
    k = len(p)
    gram = unit_slots @ unit_slots.T
    singular = torch.linalg.eigvalsh(gram).clamp_min(0).sqrt()
    fractions = singular / singular.sum().clamp_min(1e-12)
    return {"center_norm": float(raw_center.norm()),
            "unit_slot_mean_norm": float(unit_center.norm()),
            "normalization_center_angle_deg": angle_degrees(raw_center, unit_center),
            "slot_norms": p.norm(dim=-1).tolist(),
            "pairwise_cos": float((gram.sum() - gram.trace()) / (k * (k - 1))) if k > 1 else None,
            "effective_rank": float((-(fractions * fractions.clamp_min(1e-12).log()).sum()).exp())
            if float(singular.sum()) > 1e-12 else None}


def class_geometry(reconstruction, index):
    before, after, center = (reconstruction[k][index].double()
                             for k in ("before", "after", "value_center"))
    a, b = before.mean(0), after.mean(0)
    unit = reconstruction["unit_residual"][index].double()
    radius = float(reconstruction["radius"][index])
    residual_norms = reconstruction["residual"][index].double().norm(dim=-1)
    denominator = float(center.norm())
    expected = center + radius * unit.mean(0) if reconstruction["enabled"] else a
    closure = float((b - expected).abs().max())
    if closure > 2e-5 * max(1., float(after.abs().max())):
        raise ValueError("Fixed-radius center identity failed")
    def relative(vector):
        return float(vector.norm()) / denominator if denominator > 1e-12 else None
    return {"value_center_norm": denominator, "target_radius": radius,
            "pre_mean_offset_over_value_center_norm": relative(a - center),
            "post_mean_offset_over_value_center_norm": relative(b - center),
            "mean_shift_over_value_center_norm": relative(b - a),
            "pre_value_center_angle_deg": angle_degrees(a, center),
            "post_value_center_angle_deg": angle_degrees(b, center),
            "pre_post_center_angle_deg": angle_degrees(a, b),
            "pre_post_unit_mean_angle_deg": angle_degrees(F.normalize(before, dim=-1).mean(0),
                                                          F.normalize(after, dim=-1).mean(0)),
            "mean_unit_residual_norm": float(unit.mean(0).norm()),
            "residual_norms": residual_norms.tolist(),
            "clamped_residual_slots": int((residual_norms < 1e-6).sum()),
            "value_center_radius_clamped": denominator < 1e-6,
            "realized_radii": (after - center).norm(dim=-1).tolist(),
            "center_identity_max_abs_error": closure,
            "before": stage_geometry(before), "after": stage_geometry(after)}


def validate_reconstructed_bank(reconstruction, bank):
    """CPU/GPU roundoff allowed, but a different forward/protocol fails closed."""
    after = reconstruction["after"]
    cached = bank["prototypes"].cpu().float()
    if after.shape != cached.shape or not torch.isfinite(cached).all():
        raise ValueError("Reconstructed prototype bank shape/values differ")
    error = float((after - cached).abs().max())
    direction_error = float((F.normalize(after, dim=-1) - F.normalize(cached, dim=-1)).norm(dim=-1).max())
    checks = {"raw_max_abs_error": error, "max_unit_slot_l2_error": direction_error,
              "raw_atol": 2e-4, "raw_rtol": 2e-4, "unit_slot_l2_tolerance": 1e-3}
    if not torch.allclose(after, cached, atol=2e-4, rtol=2e-4) or direction_error > 1e-3:
        raise ValueError(f"Reconstructed prototypes differ from native cache: {checks}")
    return checks


def fixed_query_audit(entry, sample, bank, class_index, protocol, reconstruction):
    validated = decompose_entry(entry, sample, bank, class_index, protocol)
    variants = {}
    for stage in ("before", "after"):
        variants[stage] = decompose_logit(
            sample["features"][entry["query_id"]], reconstruction[stage][class_index],
            temperature=bank["temperature"], logit_scale=bank["logit_scale"], cls_bias=bank["cls_bias"])
    replay_error = abs(variants["after"]["native_logit"] - validated["logit_terms"]["native_logit"])
    if replay_error > 2e-3:
        raise ValueError(f"Reconstructed query logit differs from native cache: {replay_error}")
    weight = float(protocol["beta"] if bank["novel_mask"][class_index] else protocol["alpha"])
    offset = math.log(protocol["novel_scale"]) if bank["novel_mask"][class_index] else 0.
    scores = {}
    for stage, terms in variants.items():
        logp = float(F.logsigmoid(torch.tensor(terms["native_logit"], dtype=torch.float64)))
        fused = (1 - weight) * logp + weight * entry["clip_log_probability"] + offset
        scores[stage] = {"detector_probability": math.exp(logp), "fused_log_score": fused,
                         "fused_score": math.exp(fused)}
    feature = F.normalize(sample["features"][entry["query_id"]].cpu().double(), dim=0)
    center = reconstruction["value_center"][class_index].double()
    anchor_response = float(bank["logit_scale"]) * float(torch.dot(feature, F.normalize(center, dim=0)))
    return {"query_id": entry["query_id"], "box_xyxy": entry["box_xyxy"],
            "clip_log_probability_fixed": entry["clip_log_probability"],
            "value_center_response_fixed": anchor_response,
            "value_center_direction_defined": bool(center.norm() > 1e-12),
            "before": variants["before"], "after": variants["after"],
            "after_minus_before": {k: variants["after"][k] - variants["before"][k] for k in TERMS},
            "scores": scores, "native_cache_logit": validated["logit_terms"]["native_logit"],
            "native_cache_checks": validated["score_replay_checks"],
            "reconstructed_logit_abs_error": replay_error,
            "reconstructed_logit_tolerance": 2e-3}
