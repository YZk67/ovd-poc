#!/usr/bin/env python3
"""CPU-only postmortem of completed A/B predictions; no checkpoint/model loading.

Uses each arm's certified prediction JSON, saves all rare/all-IoU PR streams
once, then checks exact full-AP decomposition and the pre-existing focus classes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.decoder_aux_postmortem_ops import FOCUS, analyze, display


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def verified(expected, label):
    print(f"[verify CPU] {label}: {expected['path']}", flush=True)
    actual = file_identity(expected["path"])
    if actual["sha256"] != expected["sha256"] or actual["bytes"] != expected["bytes"]:
        raise ValueError(f"{label} changed; cannot mix stale reports/predictions")
    return actual


def inputs_from_trial(args):
    summary_path = Path(args.summary).expanduser().resolve()
    summary = load_json(summary_path)
    manifest_path = summary_path.parent / "manifest.json"
    m = load_json(manifest_path)
    signature = digest({k: v for k, v in m.items() if k != "fingerprint"})
    if (not summary.get("complete") or summary.get("updates_per_arm") != 500 or m.get("updates") != 500
            or summary.get("manifest_fingerprint") != signature or m.get("fingerprint") != signature):
        raise ValueError("Need the completed, fingerprint-matched 500-update A/B trial")
    verified(m["classification_audit"], "source classification audit (metadata only)")
    classification = load_json(m["classification_audit"]["path"])
    original_ann = classification["inputs"]["sources"]["annotations"]
    annotation_path = Path(args.annotations or original_ann["path"]).expanduser().resolve()
    ann = verified({**original_ann, "path": str(annotation_path)}, "original full LVIS annotations")
    predictions, expected = {}, {}
    for arm in ("A", "B"):
        evaluation = summary["evaluations"][arm]
        expected[arm] = evaluation["metrics"]["APr"]
        if not math.isfinite(expected[arm]) or not 0 <= expected[arm] <= 100:
            raise ValueError("Invalid summary APr")
        predictions[arm] = verified(evaluation["predictions"], f"{arm} all-class predictions")
    if os.path.samefile(predictions["A"]["path"], predictions["B"]["path"]):
        raise ValueError("A/B predictions must be different files")
    if not math.isclose(expected["B"]-expected["A"], summary["delta_B_minus_A"]["APr"], abs_tol=1e-5):
        raise ValueError("Summary B-A APr arithmetic mismatch")
    return {"summary": file_identity(summary_path), "trial_manifest": file_identity(manifest_path),
            "classification_audit": m["classification_audit"], "annotations": ann,
            "predictions": predictions, "expected_apr": expected,
            "code": {relative: file_identity(ROOT/relative) for relative in (
                "tools/review_decoder_aux_ablation.py", "tools/decoder_aux_postmortem_ops.py",
                "tools/report_lvis_rare_pr.py", "tools/rare_pr_comparison_ops.py")}}


def report_command(inputs, arm, output):
    return [sys.executable, "-u", str(ROOT/"tools/report_lvis_rare_pr.py"),
            "--predictions", inputs["predictions"][arm]["path"], "--annotations", inputs["annotations"]["path"],
            "--expected-apr", str(inputs["expected_apr"][arm]), "--apr-tolerance", "0.0002",
            "--all-curves", "--all-iou-curves", "--focus", *FOCUS, "--output", str(output)]


def run_cpu(command, log_path):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    with log_path.open("x") as log:
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            status = process.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)


def verify_report(path, inputs, arm):
    report = load_json(path)
    pred = inputs["predictions"][arm]
    if (Path(report["predictions"]).resolve() != Path(pred["path"]).resolve()
            or report["prediction_bytes"] != pred["bytes"]
            or Path(report["annotations"]).resolve() != Path(inputs["annotations"]["path"]).resolve()
            or not math.isclose(report["official_apr"], inputs["expected_apr"][arm], rel_tol=0, abs_tol=.0002)):
        raise ValueError(f"{arm} PR report differs from certified trial predictions/annotations/APr")
    # Self comparison validates ALL ten-IoU streams and raw/official AP closure.
    analyze(report, report)
    return report


def run(args):
    inputs = inputs_from_trial(args)  # Both files checked BEFORE any CPU evaluation.
    output = Path(args.output_dir).expanduser().resolve()
    protected = [Path(inputs[k]["path"]).resolve() for k in ("summary", "trial_manifest", "classification_audit", "annotations")]
    protected += [Path(v["path"]).resolve() for v in inputs["predictions"].values()]
    trial = Path(inputs["summary"]["path"]).parent
    if output == trial or trial in output.parents or output in trial.parents or any(
            output == p or output in p.parents or p in output.parents for p in protected):
        raise ValueError("Use a separate output directory; trial and source files stay read-only")
    signature = digest(inputs)
    manifest = {"inputs": inputs, "fingerprint": signature, "gpu_inference": False, "training_updates": 0}
    if args.resume:
        if load_json(output/"manifest.json") != manifest:
            raise ValueError("Postmortem inputs/code changed; cannot reuse prior CPU reports")
    else:
        if output.exists():
            raise ValueError("Use a NEW directory; do not pre-create it or tee inside it. --resume reuses verified reports.")
        output.mkdir(parents=True, exist_ok=False)
        save_json(output/"manifest.json", manifest)
    if args.prepare_only:
        print("[prepare-only] CPU identities checked; no evaluation, GPU or training", flush=True)
        return manifest
    reports = {}
    for arm in ("A", "B"):
        path, receipt = output/f"{arm}_report.json", output/f"{arm}_verified.json"
        if receipt.exists():
            prior = load_json(receipt)
            if prior["fingerprint"] != signature or prior["report"] != file_identity(path):
                raise ValueError(f"Stale/tampered {arm} CPU report")
            reports[arm] = verify_report(path, inputs, arm)
            print(f"[reuse CPU] {arm} complete all-IoU report", flush=True)
            continue
        if path.exists() or (output/f"{arm}.log").exists():
            raise ValueError(f"Incomplete {arm} CPU output: retain for diagnosis; no overwrite or GPU fallback")
        print(f"[CPU {arm}] all validation images; all rare categories; all ten IoUs", flush=True)
        run_cpu(report_command(inputs, arm, path), output/f"{arm}.log")
        reports[arm] = verify_report(path, inputs, arm)
        # Catch source changes during evaluation, including same-size replacement.
        verified(inputs["predictions"][arm], f"{arm} predictions still unchanged")
        verified(inputs["annotations"], "annotations still unchanged")
        save_json(receipt, {"fingerprint": signature, "report": file_identity(path)})
    result = analyze(reports["A"], reports["B"])
    result.update(manifest_fingerprint=signature, inputs=inputs,
                  source_reports={arm: file_identity(output/f"{arm}_report.json") for arm in ("A", "B")},
                  gpu_inference=False, training_updates=0)
    save_json(output/"report.json", result)
    display(result)
    print(f"[save] {output/'report.json'}", flush=True)
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True, help="Completed run_decoder_aux_ablation.py summary.json")
    p.add_argument("--output-dir", required=True, help="NEW directory, separate from the A/B trial")
    p.add_argument("--annotations", help="Optional relocated copy, hash must match original LVIS annotation JSON")
    p.add_argument("--resume", action="store_true", help="Reuse only hash-verified completed CPU reports")
    p.add_argument("--prepare-only", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
