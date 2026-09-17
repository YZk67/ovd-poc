#!/usr/bin/env python3
"""One full LVIS evaluation of frozen-TPA formula arm B with its log(5) bias.

Reuse the completed screen's checkpoint and inference protocol. The only model
override changed is the evaluation logit bias. Compare with saved B (calibrated)
and A results, keeping calibration and joint train/eval effects separate.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.evaluate_decoder_rollback import evaluation_command, run_evaluation, verified_identity
from tools.run_tpa_formula_screen import collect_evaluation
from tools.summarize_eq2_counterfactual import METRICS, read_last_result
from tools.train_tpa_formula_arm import read_manifest


LOG5 = math.log(5)
BIAS_KEY = "model.classifier.tpa_eval_logit_bias="


def log5_command(config, checkpoint, output, num_gpus):
    command = evaluation_command(config, checkpoint, output, num_gpus)
    entries = [i for i, token in enumerate(command) if token.startswith(BIAS_KEY)]
    if len(entries) != 1 or command[entries[0]] != BIAS_KEY + "0.0":
        raise ValueError("Expected exactly one zero-bias baseline override")
    command[entries[0]] = BIAS_KEY + repr(LOG5)
    return command


def validate_screen(screen_dir):
    manifest = read_manifest(screen_dir / "manifest.json")
    report = load_json(screen_dir / "summary.json")
    if Path(manifest["output_dir"]).resolve() != screen_dir:
        raise ValueError("Use the original screen directory on the training server")
    if (
        report.get("complete") is not True
        or report.get("all_tpa_geometry_unchanged") is not True
        or report.get("manifest_fingerprint") != manifest["fingerprint"]
        or report.get("updates_per_arm") != manifest["updates"]
        or manifest["arms"].get("B") != "calibrated_plus_logK"
    ):
        raise ValueError("Need a completed frozen-TPA formula screen with arm B")

    for arm in ("A", "B"):
        stage = report["training"][arm]
        receipts = stage["receipts"]
        if (
            stage.get("tpa_geometry_unchanged") is not True
            or len(receipts) != manifest["num_gpus"]
            or {row["rank"] for row in receipts} != set(range(manifest["num_gpus"]))
            or any(
                row.get("complete") is not True
                or row["arm"] != arm
                or row["aggregation"] != manifest["arms"][arm]
                or row["updates"] != manifest["updates"]
                or row["manifest_fingerprint"] != manifest["fingerprint"]
                or row["tpa_geometry_digest"] != manifest["tpa_geometry_digest"]
                for row in receipts
            )
        ):
            raise ValueError(f"Missing/inconsistent frozen-TPA receipts for {arm}")
        directory = screen_dir / f"eval_{arm}"
        receipt = load_json(directory / "verified_result.json")
        if (
            receipt["checkpoint"] != stage["checkpoint"]
            or receipt["manifest_fingerprint"] != manifest["fingerprint"]
            or receipt["result"] != report["evaluations"][arm]
        ):
            raise ValueError(f"Evaluation receipt disagrees with summary for {arm}")
        metrics = report["evaluations"][arm]["metrics"]
        values = read_last_result(directory / "log.txt")
        if set(metrics) != set(METRICS) or any(
            not math.isfinite(metrics[key])
            or not 0 <= metrics[key] <= 100
            or not math.isclose(value, metrics[key], rel_tol=0, abs_tol=0.00005)
            for key, value in zip(METRICS, values)
        ):
            raise ValueError(f"Baseline log disagrees with summary for {arm}")

    checkpoint = report["training"]["B"]["checkpoint"]
    if Path(checkpoint["path"]).resolve() != screen_dir / "B" / "model_final.pth":
        raise ValueError("Expected this screen's B/model_final.pth")
    verified_identity(checkpoint, "arm B checkpoint")
    expected = evaluation_command(
        manifest["config"]["path"], checkpoint["path"], screen_dir / "eval_B", manifest["num_gpus"]
    )
    previous = load_json(screen_dir / "eval_B" / "command.json")
    # Use this environment's interpreter, but require every baseline argument to
    # match. Never execute arbitrary command text from an uploaded JSON.
    if previous[1:] != expected[1:]:
        raise ValueError("Saved B inference command differs from the current evaluation protocol")
    return manifest, report, checkpoint


def comparison(report, result):
    a = report["evaluations"]["A"]["metrics"]
    b = report["evaluations"]["B"]["metrics"]
    plus = result["metrics"]
    return {
        "A_calibrated": a,
        "B_calibrated": b,
        "B_plus_log5": plus,
        "delta_B_plus_log5_minus_B_calibrated": {key: plus[key] - b[key] for key in METRICS},
        "delta_B_plus_log5_minus_A_calibrated": {key: plus[key] - a[key] for key in METRICS},
    }


def run(args):
    screen_dir = Path(args.screen_dir).expanduser().resolve()
    output = (Path(args.output_dir).expanduser().resolve() if args.output_dir
              else screen_dir / "eval_B_plus_log5")
    if output.exists():
        raise ValueError("Use a NEW evaluation output directory; existing results will not be overwritten")
    if args.num_gpus is not None and args.num_gpus < 1:
        raise ValueError("num-gpus must be positive")
    manifest, report, checkpoint = validate_screen(screen_dir)
    num_gpus = args.num_gpus if args.num_gpus is not None else manifest["num_gpus"]
    command = log5_command(manifest["config"]["path"], checkpoint["path"], output, num_gpus)
    provenance = {
        "screen_summary": file_identity(screen_dir / "summary.json"),
        "screen_manifest": file_identity(screen_dir / "manifest.json"),
        "baseline_command": file_identity(screen_dir / "eval_B" / "command.json"),
        "checkpoint": checkpoint,
        "config": manifest["config"],
        "tpa_geometry_digest": manifest["tpa_geometry_digest"],
        "eval_logit_bias": LOG5,
        "command": command,
        "runner": file_identity(__file__),
        "training_updates": 0,
    }
    print(f"[B same checkpoint] eval logit bias: 0 -> {LOG5:.16g}", flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        print("[dry-run] No output written; no training or GPU inference started", flush=True)
        return provenance

    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "manifest.json", provenance)
    save_json(output / "command.json", command)
    run_evaluation(command, output)
    evaluated = collect_evaluation(output)
    verified_identity(checkpoint, "arm B checkpoint unchanged after evaluation")
    verified_identity(provenance["screen_summary"], "source screen summary unchanged")
    result = {
        "complete": True,
        "training_updates": 0,
        "checkpoint_unchanged": True,
        "source_tpa_geometry_unchanged": True,
        "manifest": file_identity(output / "manifest.json"),
        "evaluation": evaluated,
        **comparison(report, evaluated),
        "scope": [
            "B_plus_log5 vs B_calibrated is a same-checkpoint inference-bias intervention.",
            "B_plus_log5 vs A_calibrated changes both the training formula and evaluation bias; it is not a training-only gain.",
            "Native inference is rerun, including encoder selection, boxes, CLIP ROI and top-300; candidates are not held fixed.",
            "No training or checkpoint weights are changed; preserved TPA geometry does not imply preserved mode usage.",
        ],
    }
    save_json(output / "summary.json", result)
    print("\n=== Arm B: evaluation log(5) counterfactual ===")
    print(f"{'variant':>42} " + " ".join(f"{key:>8}" for key in METRICS))
    for name, metrics in comparison(report, evaluated).items():
        print(f"{name:>42} " + " ".join(f"{metrics[key]:8.4f}" for key in METRICS))
    print(f"[save] {output / 'summary.json'}", flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir", required=True, help="Completed three-arm screen directory")
    parser.add_argument("--output-dir", help="New directory; defaults to SCREEN/eval_B_plus_log5")
    parser.add_argument("--num-gpus", type=int, help="Defaults to the screen's GPU count")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print command only")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
