"""Finite, reversible optimizer interventions; pure PyTorch, no detector imports.

Remove a coarse source from ALL optimizer gradients, then rerun native APR
routing and both clipping budgets. Counterfactuals deliberately include these
mediated effects; they are not additive loss attributions or training recipes.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import math

import torch

from tools.pairing_optimizer_source_ops import split_loss_keys
from tools.query_path_update_ops import clear_eval_caches

START = 71000
SOURCES = ("classification", "box", "apr", "rpsa")
ARMS = ("native", *("minus_" + s for s in SOURCES), "history_only")


def grouped_losses(losses):
    fine = split_loss_keys(losses)
    return {"classification": sum((fine[k] for k in fine if k.startswith("class_")), []),
            **{k: fine[k] for k in SOURCES[1:]}}


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def model_snapshot(model):
    # Includes nonpersistent buffers, which state_dict alone does not restore.
    return {"state": cpu_copy(model.state_dict()),
            "buffers": {k: cpu_copy(v) for k, v in model.named_buffers()}}


def restore_model(model, snapshot, buffer_changes=None):
    model.load_state_dict(snapshot["state"], strict=True)
    buffers = dict(model.named_buffers())
    if buffers.keys() != snapshot["buffers"].keys():
        raise ValueError("Named buffer layout changed during the audit")
    with torch.no_grad():
        for name, value in snapshot["buffers"].items():
            buffers[name].copy_(value)
        for name, value in (buffer_changes or {}).items():
            if name not in buffers or value.shape != buffers[name].shape or value.dtype != buffers[name].dtype:
                raise ValueError("Captured training buffer does not match this model")
            buffers[name].copy_(value)
    clear_eval_caches(model)


@contextmanager
def restored_branch(model, optimizer, snapshot, optimizer_state, *, buffer_changes=None):
    """Always restore moments, parameters and buffers, including on exceptions."""
    modes = [(m, m.training) for m in model.modules()]
    restore_model(model, snapshot, buffer_changes)
    optimizer.load_state_dict(cpu_copy(optimizer_state))
    optimizer.zero_grad(set_to_none=True)
    try:
        yield
    finally:
        optimizer.zero_grad(set_to_none=True)
        restore_model(model, snapshot)
        optimizer.load_state_dict(cpu_copy(optimizer_state))
        for module, mode in modes:
            module.training = mode


def validate_gradients(payload, named):
    names, parameters = zip(*named)
    if list(names) != payload["names"] or tuple(payload["components"]) != SOURCES:
        raise ValueError("Gradient inventory/source ordering differs")
    if len(payload["present"]) != len(named) or len(payload["full"]) != len(named):
        raise ValueError("Gradient/presence layout differs")
    for values in [payload["full"], payload["apr_reference"], *payload["components"].values()]:
        if len(values) != len(named):
            raise ValueError("Component layout differs")
        for p, v in zip(parameters, values):
            if p.shape != v.shape or p.dtype != v.dtype or not torch.isfinite(v).all():
                raise ValueError("Invalid gradient shape/dtype or nonfinite gradient")
    numerator = denominator = 0.
    for i, full in enumerate(payload["full"]):
        reconstructed = sum((payload["components"][s][i] for s in SOURCES), torch.zeros_like(full))
        if not payload["present"][i] and (torch.count_nonzero(full) or torch.count_nonzero(reconstructed)):
            raise ValueError("Nonzero gradient marked globally unused")
        numerator += float((full.double() - reconstructed.double()).square().sum())
        denominator += float(full.double().square().sum())
    error = math.sqrt(numerator) / max(math.sqrt(denominator), 1e-30)
    if error > 2e-3:
        raise ValueError(f"AMP source gradients do not reconstruct native total: relative L2={error}")
    return error


def assign_gradients(named, payload, arm, scale):
    if arm not in ARMS or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid arm/AMP scale")
    source = arm.removeprefix("minus_") if arm.startswith("minus_") else None
    for i, (_, parameter) in enumerate(named):
        if not payload["present"][i]:
            parameter.grad = None
            continue
        full = payload["full"][i]
        value = torch.zeros_like(full) if arm == "history_only" else (
            full - payload["components"][source][i] if source else full)
        parameter.grad = (value.to(parameter.device) * scale).clone()
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("Intervention overflows restored AMP scale; stop, do not compare skipped steps")


def actual_step(trainer, named, payload, arm):
    """Use actual native routing/clip/GradScaler/AdamW; NOT an SGD direction/JVP.

    The caller creates a fresh GradScaler loaded from the complete checkpoint
    for EVERY branch, preventing stale per-optimizer unscale/overflow state.
    """
    scale = float(trainer.grad_scaler.get_scale())
    assign_gradients(named, payload, arm, scale)
    # Initialize GradScaler lazily without adding any extra model backward.
    trainer.grad_scaler.scale(next(trainer.model.parameters()).new_zeros(()))
    trainer.grad_scaler.unscale_(trainer.optimizer)
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for _, p in named):
        raise FloatingPointError("Nonfinite native unscaled gradient")
    tpa = trainer._get_tpa()
    index = {id(p): i for i, (_, p) in enumerate(named)}
    apr = []
    for p in tpa.parameters():
        ref = payload["apr_reference"][index[id(p)]].to(p.device)
        apr.append(torch.zeros_like(ref) if arm in ("minus_apr", "history_only") else ref)
    trainer._route_tpa_gradients(tuple(apr))
    norms = trainer.clip_model_grads()
    if any(not torch.isfinite(n) for n in norms):
        raise FloatingPointError("Nonfinite post-routing clipping norm")
    if any(p.grad is not None and not trainer.optimizer.state.get(p) for _, p in named):
        raise ValueError("A stepped parameter has no restored AdamW history; refuse fresh optimizer state")
    before = [float(trainer.optimizer.state[p].get("step", 0))
              for _, p in named if p.grad is not None]
    trainer.grad_scaler.step(trainer.optimizer)
    trainer.grad_scaler.update()
    after = [float(trainer.optimizer.state[p]["step"]) for _, p in named if p.grad is not None]
    if not before or any(b + 1 != a for b, a in zip(before, after)):
        raise ValueError("Actual AdamW step skipped or advanced unexpectedly")
    if any(not torch.isfinite(p).all() for _, p in named):
        raise FloatingPointError("Nonfinite actual updated parameter")
    return {"optimizer_steps": 1, "parameters_stepped": len(after),
            "amp_scale_before": scale, "amp_scale_after": float(trainer.grad_scaler.get_scale()),
            "preclip_norms": dict(zip(("detector", "tpa"), map(float, norms))),
            "routing": trainer._last_tpa_projection_metrics,
            "momentum_retained": True, "weight_decay_retained": True,
            "clip_recomputed_for_all_parameters": True}


def observable(analysis):
    row = analysis["by_iou"]["0.50"]
    best = row["best_fused_true_class_eligible"]
    return {"eligible": row["raw_eligible_queries"], "kept": row["retained_true_class_queries"],
            "log_score_margin": None if best is None else math.log(max(best["score_threshold_ratio"], 1e-30)),
            "true_score": None if best is None else best["fused_score"],
            "cutoff": analysis["image_cutoff"]}


def summarize(windows, tolerance=1e-4):
    """Require actual native worsening; never turn local agreement into AP proof."""
    rows = []
    for window in windows:
        reference = observable(window["buffer_only"])
        arms = {k: observable(v["analysis"]) for k, v in window["arms"].items()}
        base, native = reference["log_score_margin"], arms["native"]["log_score_margin"]
        # Box disappearance is reported separately, not folded into a score claim.
        delta = None if base is None or native is None else native - base
        candidates = []
        if delta is not None and delta < -tolerance:
            candidates = [s for s in SOURCES if arms["minus_"+s]["log_score_margin"] is not None
                          and arms["minus_"+s]["log_score_margin"] > native + tolerance]
        rows.append({"window": window["window"], "native_minus_buffer_control": delta,
                     "native_degradation_reproduced": delta is not None and delta < -tolerance,
                     "removals_improving_over_native": candidates, "observables": arms})
    consistent = [s for s in SOURCES if len(rows) >= 2 and all(
        r["native_degradation_reproduced"] and s in r["removals_improving_over_native"] for r in rows)]
    return {"primary": "IoU>=0.50 best native true-class log(score/top300 cutoff)",
            "numerical_tolerance": tolerance, "windows": rows, "consistent_local_candidates": consistent,
            "verdict": "LOCAL_CANDIDATE_ONLY" if consistent else "NO_CONSISTENT_LOCAL_SOURCE",
            "historical_causality_proven": False, "global_AP_measured": False,
            "next_training_recommended": False}
