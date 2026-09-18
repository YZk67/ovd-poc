"""One-graph GT-weighting probes and disposable, native AdamW steps.

No Detectron2 dependency. No live optimizer step or model weight write.
"""
from __future__ import annotations

from copy import deepcopy
import math

import torch

from tools.accumulation_objective_ops import split_objective
from tools.decoder_aux_ablation_ops import state_digest


ARMS = ("native_micro", "pooled_gt")


def capture_gradients(losses, parameters, tpa_parameters, multiplier, amp_scale, accumulation=2):
    """Both objectives share one graph, dropout, FedLoss set and assignments.

    Total gradients remain AMP-scaled until after accumulation/rank averaging.
    APR is differentiated unscaled, exactly as in Trainer._compute_apr_gradients.
    None is recorded separately: an unused parameter is NOT a zero-gradient
    AdamW update (momentum and weight decay would make those different).
    """
    detection, other = split_objective(losses)
    if (not math.isfinite(multiplier) or multiplier <= 0
            or not math.isfinite(amp_scale) or amp_scale <= 0 or accumulation < 1):
        raise ValueError("Invalid normalization/AMP scale/accumulation")
    if "loss_apr" not in other or not tpa_parameters:
        raise ValueError("Native APR and trainable TPA are required")
    apr = torch.autograd.grad(losses["loss_apr"] / accumulation, tpa_parameters,
                              retain_graph=True, allow_unused=True)
    apr = [None if g is None else g.detach().cpu().clone() for g in apr]
    if any(g is not None and not torch.isfinite(g).all() for g in apr):
        raise FloatingPointError("Nonfinite APR gradient")
    det, rest = sum(losses[k] for k in detection), sum(losses[k] for k in other)
    scalars = {"native_micro": (det + rest) / accumulation,
               "pooled_gt": (multiplier * det + rest) / accumulation}
    gradients, present = {}, {}
    for index, arm in enumerate(ARMS):
        values = torch.autograd.grad(scalars[arm] * amp_scale, parameters,
                                     retain_graph=index == 0, allow_unused=True)
        present[arm] = [g is not None for g in values]
        gradients[arm] = []
        for p, g in zip(parameters, values):
            if g is not None and not torch.isfinite(g).all():
                raise FloatingPointError("Nonfinite scaled gradient; no finite-step conclusion")
            gradients[arm].append(torch.zeros_like(p, device="cpu") if g is None
                                  else g.detach().cpu().clone())
    if present[ARMS[0]] != present[ARMS[1]]:
        raise ValueError("Normalization unexpectedly changed graph connectivity")
    return gradients, present[ARMS[0]], apr, {k: float(v.detach()) for k, v in scalars.items()}


def add_optional(accumulated, current):
    if accumulated is None:
        return [None if g is None else g.clone() for g in current]
    if len(accumulated) != len(current):
        raise ValueError("Gradient layout changed between microbatches")
    for index, value in enumerate(current):
        if value is not None:
            if accumulated[index] is None:
                accumulated[index] = value.clone()
            else:
                accumulated[index].add_(value)
    return accumulated


def assign_gradients(parameters, parts, present):
    if not (len(parameters) == len(parts) == len(present)):
        raise ValueError("Gradient/presence layout differs")
    for p, g, active in zip(parameters, parts, present):
        if p.shape != g.shape or p.dtype != g.dtype or not torch.isfinite(g).all():
            raise ValueError("Invalid synchronized gradient")
        if not active and torch.count_nonzero(g):
            raise ValueError("Unused parameter has a nonzero gradient")
        p.grad = g.to(p.device).clone() if active else None


def snapshot_gradients(parameters):
    result = []
    for p in parameters:
        if p.grad is not None and not torch.isfinite(p.grad).all():
            raise FloatingPointError("Nonfinite unscaled/routed/clipped gradient")
        result.append(torch.zeros_like(p, device="cpu") if p.grad is None
                      else p.grad.detach().cpu().clone())
    return result


def verify_forward(observed, reference):
    """Replay receipt checks; A/B themselves use the exact same forward graph."""
    for key in ("micro", "inputs", "seed", "rng_after", "native_indices",
                "selected_indices", "normalizers", "matches", "dn"):
        if observed[key] != reference[key]:
            raise ValueError("Native forward differs from the objective audit: " + key)
    losses, expected = observed["weighted_losses"], reference["weighted_losses"]
    if losses.keys() != expected.keys() or any(
            not math.isclose(v, expected[k], rel_tol=2e-5, abs_tol=2e-6) for k, v in losses.items()):
        raise ValueError("Native loss values differ from the objective audit")


def shadow_adamw_step(optimizer, ordered_parameters, expected_state_digest):
    """Run ONE actual AdamW step on disjoint clones, on the native device.

    The live .grad values are read, never replaced. Every arm independently
    copies the same live moments, LR, per-parameter step and weight decay.
    Unused parameters keep grad=None. This includes rounding to parameter dtype,
    not merely an idealized analytic delta. No clone/checkpoint is saved.
    """
    if type(optimizer) is not torch.optim.AdamW:
        raise ValueError("Require native torch.optim.AdamW")
    if state_digest(optimizer.state_dict()) != expected_state_digest:
        raise ValueError("Live optimizer state changed before shadow step")
    if len({id(p) for p in ordered_parameters}) != len(ordered_parameters):
        raise ValueError("Duplicate parameter in audit inventory")
    clones, shadow_groups = {}, []
    for group in optimizer.param_groups:
        if any(group.get(k, False) for k in ("capturable", "differentiable", "fused")):
            raise ValueError("Unsupported native AdamW implementation")
        copied = {k: deepcopy(v) for k, v in group.items() if k != "params"}
        copied["params"] = []
        for p in group["params"]:
            if id(p) in clones:
                raise ValueError("Duplicate parameter in optimizer groups")
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    raise FloatingPointError("Nonfinite shadow gradient")
                if not all(k in optimizer.state.get(p, {}) for k in ("step", "exp_avg", "exp_avg_sq")):
                    raise ValueError("Active parameter lacks restored AdamW moments")
            q = torch.nn.Parameter(p.detach().clone(), requires_grad=p.requires_grad)
            q.grad = None if p.grad is None else p.grad.detach().clone()
            clones[id(p)] = q
            copied["params"].append(q)
        shadow_groups.append(copied)
    if set(clones) != {id(p) for p in ordered_parameters}:
        raise ValueError("Optimizer and trainable audit inventories differ")
    # All effective options are already in each copied parameter group.
    # Optimizer.defaults can contain internal, non-constructor keys (e.g.
    # decoupled_weight_decay on recent PyTorch); restore it as data, not kwargs.
    shadow = torch.optim.AdamW(shadow_groups)
    shadow.defaults = deepcopy(optimizer.defaults)
    shadow.load_state_dict(deepcopy(optimizer.state_dict()))
    if state_digest(shadow.state_dict()) != expected_state_digest:
        raise ValueError("Shadow optimizer did not restore exact state")
    before_steps = [float(shadow.state[q]["step"]) if q in shadow.state else None
                    for q in (clones[id(p)] for p in ordered_parameters)]
    shadow.step()
    deltas, active_count = [], 0
    for p, old_step in zip(ordered_parameters, before_steps):
        q = clones[id(p)]
        if q.grad is not None:
            active_count += 1
            if float(shadow.state[q]["step"]) != old_step + 1:
                raise ValueError("Shadow parameter did not take exactly one update")
        elif old_step is not None and float(shadow.state[q]["step"]) != old_step:
            raise ValueError("Unused parameter optimizer state advanced")
        delta = q.detach() - p.detach()
        if not torch.isfinite(delta).all():
            raise FloatingPointError("Nonfinite shadow AdamW update")
        if q.grad is None and torch.count_nonzero(delta):
            raise ValueError("Unused parameter changed in shadow step")
        deltas.append(delta.cpu())
    if state_digest(optimizer.state_dict()) != expected_state_digest:
        raise ValueError("Shadow step mutated live optimizer state")
    return deltas, {"optimizer_steps_on_copies": 1, "updated_parameter_tensors": active_count,
                    "unused_parameter_tensors": len(deltas)-active_count,
                    "initial_optimizer_digest": expected_state_digest,
                    "device": str(ordered_parameters[0].device),
                    "live_optimizer_unchanged": True}
