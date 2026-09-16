#!/usr/bin/env python3
"""One full LVIS evaluation of the 50:50 no-radius 8ep/12ep model average.

Every floating model-state tensor is averaged jointly. Non-floating counters
come from 12ep. No training, optimizer, EMA replay, ratio sweep, or source write.
The existing query-path audit locks the 8ep endpoint, config, data, and assets;
the 12ep endpoint must be model_final.pth in that same trajectory directory.
"""

from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from lami_dino.checkpoint_init import load_trusted_torch_file
from tools.compare_rare_pr_reports import file_identity, save_json
from tools.evaluate_decoder_rollback import (
    PROTOCOL, endpoint_state, evaluation_command, run_evaluation, validate_audit,
)
from tools.query_path_update_ops import canonical_name, canonical_state, key_group
from tools.summarize_eq2_counterfactual import METRICS, read_last_result


EARLY_ITERATION = 56799
LATE_ITERATION = 85199
BASELINE = dict(zip(METRICS, (
    44.6979, 57.5725, 47.1734, 33.8265, 55.0475, 61.9948, 42.4229, 43.2172, 47.3503,
)))
SUCCESS_THRESHOLDS = {"AP": 44.4, "APr": 42.9}


def mean_tensor(early, late):
    if early.dtype in (torch.float16, torch.bfloat16):
        return ((early.float() + late.float()) * .5).to(early.dtype)
    return (early + late) * .5


def average_state(early, late):
    """Jointly average all floating tensors; accept only training-only counter drift."""
    canonical_state(early)
    canonical_state(late)
    if early.keys() != late.keys():
        raise ValueError("8ep/12ep raw model-state keys differ")
    result = {}
    floating, changed, counters = [], [], []
    for key, end in late.items():
        start = early[key]
        if start.shape != end.shape or start.dtype != end.dtype:
            raise ValueError(f"8ep/12ep shape or dtype differs: {key}")
        group = key_group(canonical_name(key))
        if group in ("frozen_clip", "fixed_protocol") and not torch.equal(start, end):
            raise ValueError(f"Frozen CLIP or fixed protocol changed across the trajectory: {key}")
        if start.is_floating_point():
            value = mean_tensor(start, end)
            floating.append(key)
            if not torch.equal(start, end):
                changed.append(key)
        else:
            if not torch.equal(start, end):
                if group != "training_only":
                    raise ValueError(f"Non-training integer/bool state changed: {key} ({group})")
                counters.append(key)
            value = end.clone()
        result[key] = value
    canonical_state(result)  # Shared aliases must remain bit-identical after averaging.
    if not changed:
        raise ValueError("8ep/12ep floating model states are identical")
    return result, {"floating_raw_keys": floating, "changed_floating_raw_keys": changed,
                    "late_nonfloating_counter_keys": counters}


def verify_average(saved, early, late):
    if set(saved) != {"model", "full_checkpoint_average"}:
        raise ValueError("Average must be weights-only, without iteration/optimizer/trainer/scheduler")
    state = saved["model"]
    canonical_state(state)
    if state.keys() != late.keys():
        raise ValueError("Serialized average model keys differ from 12ep")
    for key, value in state.items():
        expected = mean_tensor(early[key], late[key]) if late[key].is_floating_point() else late[key]
        if (value.shape != expected.shape or value.dtype != expected.dtype
                or not torch.equal(value, expected)):
            raise ValueError(f"Serialized 50:50 average verification failed: {key}")


def validate_baseline(directory):
    log = directory / "log.txt"
    predictions = directory / "lvis_instances_results.json"
    values = read_last_result(log)
    if any(not math.isclose(value, BASELINE[key], rel_tol=0, abs_tol=.0002)
           for key, value in zip(METRICS, values)):
        raise ValueError("Trajectory log does not end with the recorded native 12ep metrics")
    if not predictions.is_file() or predictions.stat().st_size <= 2:
        raise ValueError("Native 12ep full-validation predictions are missing")
    return file_identity(log), file_identity(predictions)


def collect_results(output):
    # train_net catches some evaluator failures and returns success: inspect artifacts too.
    console = (output / "console.log").read_text(errors="replace")
    if "Skipping evaluation" in console or "Traceback (most recent call last)" in console:
        raise RuntimeError("Evaluation failed/skipped; inspect console.log")
    metrics = dict(zip(METRICS, read_last_result(output / "log.txt")))
    if any(not math.isfinite(v) or not 0 <= v <= 100 for v in metrics.values()):
        raise ValueError("Invalid LVIS metrics")
    predictions = output / "lvis_instances_results.json"
    if not predictions.is_file() or predictions.stat().st_size <= 2:
        raise ValueError("No nonempty full-validation prediction JSON")
    delta = {key: metrics[key] - BASELINE[key] for key in METRICS}
    gates = {key: metrics[key] >= threshold for key, threshold in SUCCESS_THRESHOLDS.items()}
    return {"complete": True, "average_ratio_8ep": .5, "average_ratio_12ep": .5,
            "baseline_native_12ep": BASELINE, "average_8ep_12ep": metrics,
            "delta_average_minus_native12": delta,
            "precommitted_success_thresholds": SUCCESS_THRESHOLDS,
            "passes_all_thresholds": all(gates.values()), "threshold_checks": gates,
            "predictions": file_identity(predictions),
            "scope": "One full native LVIS evaluation of a joint whole-model 50:50 endpoint "
                     "average. Not EMA replay, optimizer replay, loss attribution, or a ratio sweep."}


def run(args):
    if args.num_gpus < 1 or args.cpu_threads < 1:
        raise ValueError("Positive GPU count and CPU threads required")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise ValueError("Use a NEW output directory; stale/partial evaluation must not be overwritten")
    torch.set_num_threads(args.cpu_threads)

    audit, _ = validate_audit(args.audit_report)
    sources = audit["inputs"]["sources"]
    trajectory = Path(sources["old_checkpoint"]["path"]).resolve().parent
    late_checkpoint = trajectory / "model_final.pth"
    if output == trajectory or trajectory in output.parents or output in trajectory.parents:
        raise ValueError("Output must be a separate sibling directory, outside the training trajectory")
    if not late_checkpoint.is_file():
        raise FileNotFoundError(f"Expected 12ep endpoint in the same trajectory: {late_checkpoint}")
    baseline_log, baseline_predictions = validate_baseline(trajectory)
    identities = {"early_checkpoint": sources["old_checkpoint"],
                  "late_checkpoint": file_identity(late_checkpoint)}

    states = {}
    for side, path, iteration in (
        ("early", sources["old_checkpoint"]["path"], EARLY_ITERATION),
        ("late", late_checkpoint, LATE_ITERATION),
    ):
        print(f"[load CPU] {side}: iteration={iteration}, {path}", flush=True)
        checkpoint = load_trusted_torch_file(path)
        states[side] = endpoint_state(checkpoint, iteration)
        del checkpoint
    state, checks = average_state(states["early"], states["late"])

    output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output / "average_8ep_12ep_50_50_eval_only.pth"
    provenance = {"eval_only": True, "ratio_early": .5, "ratio_late": .5,
                  "early_iteration": EARLY_ITERATION, "late_iteration": LATE_ITERATION,
                  "sources": identities, **checks}
    with checkpoint_path.open("xb") as handle:
        torch.save({"model": state, "full_checkpoint_average": provenance}, handle)
    saved = load_trusted_torch_file(checkpoint_path)
    verify_average(saved, states["early"], states["late"])
    print(f"[verified] averaged {len(checks['floating_raw_keys'])} floating tensors jointly; "
          f"{len(checks['changed_floating_raw_keys'])} changed across endpoints; "
          f"kept {len(checks['late_nonfloating_counter_keys'])} late counters", flush=True)
    del states, state, saved
    gc.collect()

    command = evaluation_command(sources["config_file"]["path"], checkpoint_path,
                                 output, args.num_gpus)
    manifest = {"audit_report": file_identity(args.audit_report), "sources": identities,
                "baseline_log": baseline_log, "baseline_predictions": baseline_predictions,
                "average_checkpoint": file_identity(checkpoint_path), "provenance": provenance,
                "protocol": PROTOCOL, "command": command, "runner": file_identity(__file__),
                "training_updates": 0, "evaluations_requested": 0 if args.prepare_only else 1}
    save_json(output / "manifest.json", manifest)
    if args.prepare_only:
        print(f"[prepared only] {checkpoint_path}; evaluation has NOT run", flush=True)
        return manifest

    print("[evaluate] ONE full LVIS evaluation, native boxes / CLIP ROI / top-300", flush=True)
    run_evaluation(command, output)
    result = collect_results(output)
    result["manifest"] = file_identity(output / "manifest.json")
    save_json(output / "summary.json", result)
    print("\n=== Native 12ep vs full-model average(8ep,12ep) ===")
    print(f"{'variant':>24} " + " ".join(f"{key:>8}" for key in METRICS))
    for name in ("baseline_native_12ep", "average_8ep_12ep", "delta_average_minus_native12"):
        print(f"{name:>24} " + " ".join(f"{result[name][key]:8.4f}" for key in METRICS))
    print(f"success_gate={result['passes_all_thresholds']} "
          f"(AP>={SUCCESS_THRESHOLDS['AP']}, APr>={SUCCESS_THRESHOLDS['APr']})")
    print(f"[save] {output / 'summary.json'}", flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-report", required=True, help="Complete query-path audit locking 8ep/config/assets")
    parser.add_argument("--output-dir", required=True, help="NEW sibling directory; do not create/tee into it first")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true", help="Build and verify only; no GPU evaluation")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
