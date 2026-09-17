"""Conditional objective audit, not a training fix or an AdamW update."""
from __future__ import annotations

from contextlib import contextmanager
import math
import re

import torch


def normalization_plan(counts, world_size=4, reference_world_size=8):
    """Counts are global GT totals for each physical microbatch.

    DDP averaging makes the effective denominator max(global_GT, world_size).
    Include the accumulation mean and the criterion's empty-batch clamp.
    The reference pools GT over all microbatches, with a declared world size.
    DN group counts remain native: only their GT denominator is changed.
    """
    if (not counts or any(int(n) != n or n < 0 for n in counts)
            or world_size < 1 or reference_world_size < 1):
        raise ValueError("Expected nonnegative GT counts and positive world sizes")
    denominators = [max(int(n), world_size) for n in counts]
    pooled = max(sum(counts), reference_world_size)
    return {
        "global_gt_counts": list(counts),
        "micro_global_denominators": denominators,
        "pooled_global_denominator": pooled,
        "detection_loss_multipliers": [len(counts) * n / pooled for n in denominators],
        "criterion_normalizers": [n / world_size for n in denominators],
        "reference_world_size": reference_world_size,
    }


def split_objective(losses):
    detection, other = [], []
    for key, value in losses.items():
        if not key.startswith("loss"):
            continue
        if not torch.is_tensor(value) or value.numel() != 1 or not torch.isfinite(value).all():
            raise ValueError("Nonfinite/non-scalar loss: " + key)
        if re.fullmatch(r"loss_(class|bbox|giou)(?:_\d+|_enc|_dn(?:_\d+)?)?", key):
            detection.append(key)
        elif key in ("loss_apr", "loss_rpsa"):
            other.append(key)
        else:
            raise ValueError("Unknown loss normalization: " + key)
    if not all(key in detection for key in ("loss_class", "loss_bbox", "loss_giou")):
        raise ValueError("Missing native detection losses")
    return detection, other


def objective_gradients(losses, parameters, multiplier, accumulation=2):
    """Two scalar objectives on exactly the SAME forward/matching graph.

    Return CPU gradients before APR routing, clipping, or optimizer moments.
    No backward into .grad, no step; APR/RPSA retain their native weighting.
    """
    detection, other = split_objective(losses)
    if not math.isfinite(multiplier) or multiplier <= 0 or accumulation < 1:
        raise ValueError("Invalid objective multiplier")
    det = sum(losses[k] for k in detection)
    rest = sum(losses[k] for k in other)
    scalars = {"micro": (det + rest) / accumulation,
               "pooled_gt": (multiplier * det + rest) / accumulation}
    result = {}
    for i, (name, loss) in enumerate(scalars.items()):
        values = torch.autograd.grad(loss, parameters, retain_graph=(i == 0), allow_unused=True)
        parts = []
        for value, p in zip(values, parameters):
            if value is not None and not torch.isfinite(value).all():
                raise ValueError("Nonfinite objective gradient")
            parts.append(torch.zeros_like(p, device="cpu", dtype=torch.float32) if value is None
                         else value.detach().float().cpu())
        result[name] = parts
    return result, {name: float(loss.detach()) for name, loss in scalars.items()}


@contextmanager
def paired_fedloss(model, shared_indices=None):
    """Consume the native sampler RNG in BOTH policies, then remap from global IDs.

    Do not reset RNG *after* the native sample: doing so would change the native
    dropout/DN stream. The caller pairs the entire forward RNG and input copies.
    """
    original = model.filter_content_info
    record = {}

    def filtered(data):
        global_labels = [x["instances"].gt_classes.clone() for x in data]
        native, mapped = original(data)
        selected = native if shared_indices is None else shared_indices.to(native.device)
        if selected.numel() != native.numel() or selected.unique().numel() != selected.numel():
            raise ValueError("FedLoss policies must have equal, unique vocabulary size")
        if selected.numel() == 0 or selected.min() < 0 or selected.max() >= model.num_classes:
            raise ValueError("Invalid FedLoss category index")
        if len(mapped) != len(global_labels):
            raise ValueError("Native sampler changed the batch layout")
        lookup = torch.full((model.num_classes,), -1, dtype=torch.long, device=selected.device)
        lookup[selected] = torch.arange(selected.numel(), device=selected.device)
        for item, labels in zip(mapped, global_labels):
            if labels.numel() and (labels.min() < 0 or labels.max() >= model.num_classes):
                raise ValueError("Invalid global training label")
            local = lookup[labels.to(selected.device)]
            if (local < 0).any():
                raise ValueError("Shared FedLoss vocabulary dropped a GT category")
            item["instances"].gt_classes = local
        record.update(native_indices=native.detach().cpu().tolist(),
                      selected_indices=selected.detach().cpu().tolist())
        return selected, mapped

    model.filter_content_info = filtered
    try:
        yield record
        if not record:
            raise ValueError("FedLoss was not called")
    finally:
        model.filter_content_info = original


def compare_gradients(reference, candidate, groups):
    """Groupwise raw-gradient differences. Zero cosine is undefined, not agreement."""
    if len(reference) != len(candidate) or len(reference) != len(groups):
        raise ValueError("Gradient inventory mismatch")
    sums = {}
    for a, b, group in zip(reference, candidate, groups):
        if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("Invalid gradient pair")
        a, b = a.double(), b.double()
        values = (float(a.square().sum()), float(b.square().sum()),
                  float((b-a).square().sum()), float((a*b).sum()))
        for name in ("all_trainable", group):
            row = sums.setdefault(name, [0., 0., 0., 0.])
            for i, v in enumerate(values):
                row[i] += v
    result = {}
    for group, (aa, bb, dd, ab) in sums.items():
        an, bn, dn = math.sqrt(aa), math.sqrt(bb), math.sqrt(dd)
        result[group] = {"reference_l2": an, "candidate_l2": bn, "difference_l2": dn,
                         "relative_difference_l2": dn/an if an else None,
                         "cosine": max(-1., min(1., ab/(an*bn))) if an and bn else None}
    return result


def category_overlap(a, b):
    a, b = set(a), set(b)
    return {"intersection": len(a & b), "union": len(a | b),
            "jaccard": len(a & b)/len(a | b) if a | b else 1.}


def dn_layout(gt_counts, dn_number):
    maximum = max(gt_counts, default=0)
    if maximum == 0 or dn_number <= 0:
        return {"dn_num": 0, "single_padding": 0}
    return {"dn_num": max(dn_number // maximum, 1), "single_padding": 2 * maximum}
