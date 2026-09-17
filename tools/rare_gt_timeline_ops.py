"""Chronological single-GT selection coverage, not AP or training causality."""

from __future__ import annotations

import math

EPOCH_ITERATIONS = {epoch: epoch * 7100 - 1 for epoch in range(8, 13)}
IOU_KEYS = ("0.50", "0.75", "0.90")


def stage_row(epoch, analysis, iou_key):
    if epoch not in EPOCH_ITERATIONS or analysis["iteration"] != EPOCH_ITERATIONS[epoch]:
        raise ValueError("Epoch/iteration mismatch in no-radius timeline")
    data = analysis["by_iou"][iou_key]
    best = data["best_fused_true_class_eligible"]
    eligible, retained = data["raw_eligible_queries"], data["retained_true_class_queries"]
    if not 0 <= retained <= eligible or (best is None) != (eligible == 0):
        raise ValueError("Invalid eligible/retained counts")
    cutoff = analysis["image_cutoff"]
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Need a positive native top300 cutoff")
    if best and (not math.isfinite(best["fused_score"]) or best["fused_score"] < 0):
        raise ValueError("Nonfinite/negative best eligible score")
    state = "retained" if retained else "no_eligible_box" if not eligible else "excluded"
    return {
        "epoch": epoch, "iteration": analysis["iteration"], "state": state,
        "eligible_queries": eligible, "retained_true_class_queries": retained,
        "maximum_query_iou": analysis["best_iou_query"]["region_iou"],
        "image_cutoff": cutoff, "best_fused_true_class_eligible": best,
        # A numerically marginal selection cannot establish a robust transition.
        "boundary_sensitive": bool(best and abs(best["fused_score"] - cutoff) <= 1e-5),
        "reason": data["reason"],
    }


def summarize_timeline(stages, missing_epochs):
    """Report ALL sampled loss/recovery transitions; never assume monotonicity.

    Brackets bound observed endpoint changes only. Unobserved steps can contain
    any number of losses/recoveries. Best queries are GT-anchored independently.
    """
    epochs = sorted(stages)
    if not epochs or epochs[0] != 8 or epochs[-1] != 12:
        raise ValueError("Both authenticated 8ep/12ep endpoints are required")
    if set(epochs) | set(missing_epochs) != set(EPOCH_ITERATIONS) or set(epochs) & set(missing_epochs):
        raise ValueError("Account for every intermediate checkpoint, present OR missing")
    result = {}
    for key in IOU_KEYS:
        rows = [stage_row(epoch, stages[epoch], key) for epoch in epochs]
        changes = []
        for left, right in zip(rows, rows[1:]):
            was_hit, is_hit = left["state"] == "retained", right["state"] == "retained"
            if was_hit == is_hit:
                continue
            changes.append({
                "kind": "loss" if was_hit else "recovery",
                "from_epoch": left["epoch"], "to_epoch": right["epoch"],
                "from_checkpoint_iteration": left["iteration"],
                "to_checkpoint_iteration": right["iteration"],
                "update_interval_inclusive": [left["iteration"] + 1, right["iteration"]],
                "missing_epochs_inside": [e for e in missing_epochs if left["epoch"] < e < right["epoch"]],
                "from_state": left["state"], "to_state": right["state"],
                "boundary_sensitive": left["boundary_sensitive"] or right["boundary_sensitive"],
            })
        losses = [c for c in changes if c["kind"] == "loss"]
        last_loss = losses[-1] if losses and rows[-1]["state"] != "retained" else None
        result[key] = {
            "rows": rows, "transitions": changes,
            "first_observed_loss_bracket": losses[0] if losses else None,
            "last_observed_loss_bracket_before_final_miss": last_loss,
            "resolution": "intermediate_checkpoints_observed" if len(epochs) > 2 else "endpoints_only_not_narrowed",
            "has_observed_recovery": any(c["kind"] == "recovery" for c in changes),
            "scope": "Sampled selection coverage, not official TP/AP, continuous-time persistence, or loss-source attribution.",
        }
    return result
