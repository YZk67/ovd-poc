"""Pure helpers for optimizer-aware query/bank loss-source screening.

The GPU runner keeps model/optimizer state unchanged.  These helpers emulate
one finite, finite-gradient AdamW step and summarize how removing one loss
source from one parameter path changes that step.
"""

from __future__ import annotations

import math
import re

import torch


SOURCES = (
    "class_final",
    "class_aux",
    "class_dn",
    "class_encoder",
    "box",
    "apr",
    "rpsa",
)
TARGETS = ("query", "bank")


def split_loss_keys(losses):
    """Partition every scalar ``loss*`` entry into a declared source."""
    groups = {name: [] for name in SOURCES}
    for key, value in losses.items():
        if not key.startswith("loss"):
            continue
        if not torch.is_tensor(value) or value.numel() != 1 or not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite/non-scalar loss: {key}")
        if key == "loss_class":
            group = "class_final"
        elif re.fullmatch(r"loss_class_\d+", key):
            group = "class_aux"
        elif re.fullmatch(r"loss_class_dn(?:_\d+)?", key):
            group = "class_dn"
        elif key == "loss_class_enc":
            group = "class_encoder"
        elif key == "loss_apr":
            group = "apr"
        elif key == "loss_rpsa":
            group = "rpsa"
        elif key == "loss_bbox" or key.startswith("loss_bbox_") or key == "loss_giou" or key.startswith("loss_giou_"):
            group = "box"
        else:
            raise ValueError(f"Unclassified training loss: {key}")
        groups[group].append(key)
    missing = [name for name, keys in groups.items() if not keys]
    if missing:
        raise ValueError(f"Expected every native loss source, missing: {missing}")
    return groups


def clip_coefficient(norm, max_norm, eps=1e-6):
    norm, max_norm = float(norm), float(max_norm)
    if not math.isfinite(norm) or norm < 0 or not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("Invalid clipping norm/budget")
    return min(1.0, max_norm / (norm + float(eps)))


def counterfactual_norm(full_norm, full_parts, counterfactual_parts):
    """Replace a subset of a full vector and return its resulting L2 norm."""
    if len(full_parts) != len(counterfactual_parts):
        raise ValueError("Counterfactual part layout differs")
    value = float(full_norm) ** 2
    for before, after in zip(full_parts, counterfactual_parts):
        if before.shape != after.shape or not torch.isfinite(before).all() or not torch.isfinite(after).all():
            raise ValueError("Invalid counterfactual gradient part")
        value += float(after.double().square().sum() - before.double().square().sum())
    if value < -1e-6 * max(float(full_norm) ** 2, 1.0):
        raise ValueError("Counterfactual norm square became negative")
    return math.sqrt(max(value, 0.0))


def _step_number(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("AdamW step must be scalar")
        value = value.item()
    value = float(value)
    if not math.isfinite(value) or value < 0 or value != int(value):
        raise ValueError("Invalid AdamW step")
    return int(value)


def adamw_delta(parameter, gradient, state, group):
    """Return the next PyTorch-style AdamW parameter delta without mutation.

    This covers the non-capturable, non-differentiable AdamW configuration used
    by the project.  The operation is performed in the parameter's native dtype
    and device, just as the live optimizer would do after gradients are unscaled.
    """
    if parameter.shape != gradient.shape or not torch.isfinite(parameter).all() or not torch.isfinite(gradient).all():
        raise ValueError("Invalid AdamW parameter/gradient")
    required = ("exp_avg", "exp_avg_sq", "step")
    if any(key not in state for key in required):
        raise ValueError("AdamW state is incomplete")
    if group.get("capturable", False) or group.get("differentiable", False):
        raise ValueError("Capturable/differentiable AdamW is unsupported")
    if group.get("maximize", False):
        gradient = -gradient
    beta1, beta2 = tuple(group.get("betas", ()))
    lr, eps, decay = (float(group[k]) for k in ("lr", "eps", "weight_decay"))
    if not (0 <= beta1 < 1 and 0 <= beta2 < 1 and lr > 0 and eps > 0 and decay >= 0):
        raise ValueError("Invalid AdamW hyperparameters")
    exp_avg = state["exp_avg"].detach().clone()
    exp_avg_sq = state["exp_avg_sq"].detach().clone()
    if exp_avg.shape != parameter.shape or exp_avg_sq.shape != parameter.shape:
        raise ValueError("AdamW moment layout differs from parameter")
    step = _step_number(state["step"]) + 1
    exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
    if group.get("amsgrad", False):
        if "max_exp_avg_sq" not in state:
            raise ValueError("AMSGrad state is incomplete")
        maximum = torch.maximum(state["max_exp_avg_sq"].detach(), exp_avg_sq)
        denom = maximum.sqrt().div_(math.sqrt(1.0 - beta2 ** step)).add_(eps)
    else:
        denom = exp_avg_sq.sqrt().div_(math.sqrt(1.0 - beta2 ** step)).add_(eps)
    updated = parameter.detach().clone()
    updated.mul_(1.0 - lr * decay)
    updated.addcdiv_(exp_avg, denom, value=-lr / (1.0 - beta1 ** step))
    delta = updated - parameter.detach()
    if not torch.isfinite(delta).all():
        raise FloatingPointError("Nonfinite AdamW counterfactual delta")
    return delta


def vector_metrics(normal, counterfactual, historical):
    """Aggregate a list of parameter-shaped updates without concatenating."""
    if not (len(normal) == len(counterfactual) == len(historical)) or not normal:
        raise ValueError("Update vector layouts differ or are empty")
    sums = {key: 0.0 for key in (
        "normal_sq", "counterfactual_sq", "effect_sq", "historical_sq",
        "normal_historical", "counterfactual_historical", "effect_historical",
        "effect_normal",
    )}
    for a, b, h in zip(normal, counterfactual, historical):
        if a.shape != b.shape or a.shape != h.shape:
            raise ValueError("Update tensor layouts differ")
        a, b, h = a.double(), b.double(), h.double()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all() or not torch.isfinite(h).all():
            raise ValueError("Nonfinite update metric input")
        e = a - b
        sums["normal_sq"] += float(a.square().sum())
        sums["counterfactual_sq"] += float(b.square().sum())
        sums["effect_sq"] += float(e.square().sum())
        sums["historical_sq"] += float(h.square().sum())
        sums["normal_historical"] += float((a * h).sum())
        sums["counterfactual_historical"] += float((b * h).sum())
        sums["effect_historical"] += float((e * h).sum())
        sums["effect_normal"] += float((e * a).sum())
    norms = {name: math.sqrt(max(sums[name + "_sq"], 0.0))
             for name in ("normal", "counterfactual", "effect", "historical")}
    def cosine(dot, left, right):
        denom = norms[left] * norms[right]
        return sums[dot] / denom if denom > 1e-30 else None
    return {
        **{name + "_update_l2": value for name, value in norms.items() if name != "historical"},
        "historical_delta_l2": norms["historical"],
        "normal_historical_cosine": cosine("normal_historical", "normal", "historical"),
        "counterfactual_historical_cosine": cosine("counterfactual_historical", "counterfactual", "historical"),
        "source_effect_historical_cosine": cosine("effect_historical", "effect", "historical"),
        "source_effect_normal_cosine": cosine("effect_normal", "effect", "normal"),
        "source_effect_historical_projection": (
            sums["effect_historical"] / sums["historical_sq"]
            if sums["historical_sq"] > 1e-30 else None
        ),
    }


def summarize_screen(windows):
    """Deterministic descriptive ranking; never labels a loss as causal."""
    rows = []
    for source in SOURCES:
        for target in TARGETS:
            values = [w["sources"][source][target] for w in windows]
            cosines = [v["source_effect_historical_cosine"] for v in values
                       if v["source_effect_historical_cosine"] is not None]
            projections = [v["source_effect_historical_projection"] for v in values
                           if v["source_effect_historical_projection"] is not None]
            connected = sum(v["source_gradient_l2"] > 1e-12 for v in values)
            rows.append({
                "source": source,
                "target": target,
                "connected_windows": connected,
                "window_count": len(values),
                "positive_alignment_windows": sum(v > 0 for v in cosines),
                "mean_effect_historical_cosine": sum(cosines) / len(cosines) if cosines else None,
                "mean_effect_historical_projection": sum(projections) / len(projections) if projections else None,
                "effect_historical_cosines": cosines,
            })
    rows.sort(key=lambda row: (
        row["mean_effect_historical_cosine"] is not None,
        (row["mean_effect_historical_cosine"]
         if row["mean_effect_historical_cosine"] is not None else -float("inf")),
    ), reverse=True)
    return rows
