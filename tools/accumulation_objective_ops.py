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


def sample_global_categories(model, global_gt, sampler, *, rank=0, broadcast=None):
    """Native sampler semantics with a size known identically on every rank.

    FedLoss's target is a minimum: never truncate GT to fit it. The audit has
    already gathered the complete GT union, so no fixed-size receive buffer is
    necessary. This does not modify DINO's production distributed sampler.
    """
    gt = torch.as_tensor(sorted(set(global_gt)), dtype=torch.long, device=model.device)
    if gt.numel() and (gt.min() < 0 or gt.max() >= model.num_classes):
        raise ValueError("Invalid global GT union")
    size = max(gt.numel(), int(model.fed_loss_num_cat))
    if not 0 < size <= model.num_classes:
        raise ValueError("Invalid FedLoss sample target")
    if rank == 0:
        indices = sampler(gt, model.fed_loss_num_cat, model.num_classes, model.freq_weight)
        if indices.numel() != size:
            raise ValueError("Native sampler returned unexpected vocabulary size")
    else:
        indices = torch.empty(size, dtype=torch.long, device=model.device)
    if broadcast is not None:
        broadcast(indices)
    elif rank != 0:
        raise ValueError("Nonzero rank requires category broadcast")
    return indices


@contextmanager
def paired_fedloss(model, shared_indices=None, native_draw=None):
    """Consume the native sampler RNG in BOTH policies, then remap from global IDs.

    Do not reset RNG *after* the native sample: doing so would change the native
    dropout/DN stream. The caller pairs the entire forward RNG and input copies.
    """
    original = model.filter_content_info
    record = {}

    def filtered(data):
        global_labels = [x["instances"].gt_classes.clone() for x in data]
        if native_draw is None:
            native, mapped = original(data)
        else:
            native, mapped = native_draw(), data
        selected = native if shared_indices is None else shared_indices.to(native.device)
        if selected.numel() < native.numel():
            raise ValueError("Shared FedLoss vocabulary cannot shrink below the native sample")
        if selected.unique().numel() != selected.numel():
            raise ValueError("FedLoss vocabulary must be unique")
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
                      selected_indices=selected.detach().cpu().tolist(),
                      native_category_count=native.numel(), selected_category_count=selected.numel())
        return selected, mapped

    model.filter_content_info = filtered
    try:
        yield record
        if not record:
            raise ValueError("FedLoss was not called")
    finally:
        model.filter_content_info = original


@contextmanager
def paired_tpa_dropout(dropout, fed_record, *, reference=None, seed=0):
    """Keep native dropout unchanged; share its masks by GLOBAL category ID.

    For shared-only categories draw independent masks with a private generator.
    Restore the native post-dropout RNG so a longer bank cannot shift DN's
    random stream. The cache is ephemeral (contains tensors), not a JSON row.
    """
    original = dropout.forward
    cache, metadata = {}, {}

    def states(x):
        return (torch.get_rng_state().clone(),
                torch.cuda.get_rng_state(x.device).clone() if x.is_cuda else None)

    def restore(x, state):
        torch.set_rng_state(state[0])
        if x.is_cuda:
            torch.cuda.set_rng_state(state[1], x.device)

    def forward(x):
        if metadata or not dropout.training or dropout.inplace or not 0 <= dropout.p < 1:
            raise ValueError("Expected one non-inplace training TPA dropout per forward")
        ids = fed_record["selected_indices"]
        if x.ndim != 3 or x.shape[0] != len(ids):
            raise ValueError("TPA dropout bank does not match FedLoss categories")
        before = states(x)
        if reference is None:
            result = original(x)
            after = states(x)
            # Recover the exact mask, including positions where input is zero,
            # without consuming an extra RNG draw in the native forward.
            devices = [x.device.index] if x.is_cuda else []
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                restore(x, before)
                mask = original(torch.ones_like(x))
            if not torch.allclose(result.detach(), x.detach()*mask, rtol=1e-6, atol=1e-7):
                raise ValueError("Cannot reconstruct native TPA dropout mask")
            cache.update(ids=list(ids), mask=mask, before=before, after=after, p=dropout.p)
            common = len(ids)
        else:
            if dropout.p != reference["p"] or tuple(x.shape[1:]) != tuple(reference["mask"].shape[1:]):
                raise ValueError("TPA dropout layout/probability changed")
            for a, b in zip(before, reference["before"]):
                if (a is None) != (b is None) or (a is not None and not torch.equal(a, b)):
                    raise ValueError("Unpaired RNG before TPA dropout")
            generator = torch.Generator(device=x.device)
            generator.manual_seed(seed)
            mask = torch.empty_like(x).bernoulli_(1-dropout.p, generator=generator).div_(1-dropout.p)
            native_map = {c: i for i, c in enumerate(reference["ids"])}
            dst = [i for i, c in enumerate(ids) if c in native_map]
            src = [native_map[ids[i]] for i in dst]
            mask[dst] = reference["mask"][src]
            common = len(dst)
            result = x * mask
            restore(x, reference["after"])
        metadata.update(mode="native" if reference is None else "category_keyed_replay",
                        category_count=len(ids), shared_masks=common,
                        fresh_masks=len(ids)-common, native_downstream_rng_preserved=True)
        return result

    dropout.forward = forward
    try:
        yield cache, metadata
        if not metadata:
            raise ValueError("TPA dropout was not observed")
    finally:
        dropout.forward = original


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
