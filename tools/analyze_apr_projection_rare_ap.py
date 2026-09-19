#!/usr/bin/env python3
"""CPU-only rare AP concentration from completed 4000-update A/P predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Project-local tools must precede Detectron2's unrelated tools package.
sys.path.insert(0, str(ROOT))
from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.rare_ap_concentration_ops import analyze, format_results

# Metadata only: do not import training helpers, torch, or load checkpoints.
POLICIES = {"A": {"projection": True, "barrier_weight": .1, "balance_weight": .03},
            "P": {"projection": False, "barrier_weight": .1, "balance_weight": .03}}
CODE_FILES = ("tools/analyze_apr_projection_rare_ap.py", "tools/rare_ap_concentration_ops.py",
              "tools/report_lvis_rare_pr.py", "tools/rare_pr_comparison_ops.py", "tools/compare_rare_pr_reports.py")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trial-dir", required=True, help="Completed extension directory containing manifest.json and summary.json")
    p.add_argument("--output-dir", required=True, help="New CPU report directory, separate from the trial")
    p.add_argument("--resume", action="store_true", help="Reuse this pipeline's verified CPU reports after interruption")
    return p.parse_args(argv)


def verified_identity(record, label):
    path = Path(record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}; NO inference/training will be started")
    print(f"[verify] {label}: {path}", flush=True)
    actual = file_identity(path)
    if actual != record:
        raise ValueError(f"Identity mismatch for {label}; do not substitute another evaluation's JSON")
    return actual


def preflight(args):
    trial, output = Path(args.trial_dir).resolve(), Path(args.output_dir).resolve()
    if output == trial or output in trial.parents or trial in output.parents:
        raise ValueError("Use a separate output directory outside the source trial")
    inputs = {k: file_identity(trial/(k+".json")) for k in ("manifest", "summary")}
    manifest, summary = (load_json(inputs[k]["path"]) for k in ("manifest", "summary"))
    digest = hashlib.sha256(json.dumps({k: v for k, v in manifest.items() if k != "fingerprint"}, sort_keys=True).encode()).hexdigest()
    if (manifest.get("fingerprint") != digest or summary.get("manifest_fingerprint") != digest
            or manifest.get("schema") != "apr_projection_extension_v1"
            or summary.get("complete") is not True or summary.get("arms") != POLICIES
            or manifest.get("arms") != POLICIES or manifest.get("total_updates") != 4000
            or summary.get("total_updates_per_arm") != 4000 or set(summary.get("evaluations", {})) != {"A", "P"}):
        raise ValueError("Require the completed, matching 4000-update A/P extension manifest and summary")
    expected = {}
    for arm in ("A", "P"):
        value = summary["evaluations"][arm]["metrics"]["APr"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f"Invalid {arm} expected APr")
        expected[arm] = value
    if not math.isclose(summary["delta_P_minus_A"]["APr"], expected["P"]-expected["A"], abs_tol=1e-8, rel_tol=0):
        raise ValueError("Summary P-A delta disagrees with its arm metrics")
    # Check BOTH saved predictions and the original annotation identity before
    # spending any evaluator time. No dependency on original GPU assets.
    records = {"annotations": manifest["val_annotations"],
               **{arm: summary["evaluations"][arm]["predictions"] for arm in ("A", "P")}}
    for label, record in records.items():
        if output == Path(record["path"]).resolve() or output in Path(record["path"]).resolve().parents:
            raise ValueError("Output must not contain/overwrite a source input")
        inputs[label] = verified_identity(record, label)
    if os.path.samefile(inputs["A"]["path"], inputs["P"]["path"]):
        raise ValueError("A/P must be different saved prediction files")
    for label in ("manifest", "summary"):
        if file_identity(inputs[label]["path"]) != inputs[label]:
            raise ValueError("Source metadata changed during preflight")
    provenance = {"schema": "apr_projection_rare_ap_v1", "inputs": inputs,
                  "expected_apr": expected, "trial_fingerprint": digest,
                  "code": {name: file_identity(ROOT/name) for name in CODE_FILES}}
    managed = ["inputs.json", "STATUS.json", "report.json", "results.txt", "per_class.csv"]
    managed += [f"{arm}_{name}.json" for arm in ("A", "P") for name in ("report", "receipt")]
    if any((output/name).is_symlink() for name in managed):
        raise ValueError("Refusing symlinked output artifacts")
    if args.resume:
        if not (output/"inputs.json").is_file() or load_json(output/"inputs.json") != provenance:
            raise ValueError("Resume inputs/code differ; use a new output directory")
    elif any((output/name).exists() for name in managed):
        raise ValueError("Refusing to overwrite existing reports; use --resume or a new directory")
    return output, provenance


def validate_report(report, arm, provenance):
    prediction = provenance["inputs"][arm]
    if (Path(report["predictions"]).resolve() != Path(prediction["path"])
            or report["prediction_bytes"] != prediction["bytes"]
            or Path(report["annotations"]).resolve() != Path(provenance["inputs"]["annotations"]["path"])
            or not isinstance(report.get("official_apr"), (int, float))
            or not math.isclose(report["official_apr"], provenance["expected_apr"][arm], abs_tol=.0002, rel_tol=0)):
        raise ValueError(f"{arm} report provenance/APr mismatch")
    analyze(report, report)  # All categories, AP validity, macro closure, GT count.


def run(args):
    output, provenance = preflight(args)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output/"inputs.json", provenance)
    save_json(output/"STATUS.json", {"status": "RUNNING", "stage": "preflight_complete"})
    try:
        reports, identities = {}, {}
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        for arm in ("A", "P"):
            target, receipt = output/f"{arm}_report.json", output/f"{arm}_receipt.json"
            save_json(output/"STATUS.json", {"status": "RUNNING", "stage": "official_CPU_"+arm})
            if args.resume and receipt.is_file():
                saved = load_json(receipt)
                if saved != {"report": file_identity(target), "inputs": provenance}:
                    raise ValueError(f"Cached {arm} report/receipt changed")
                print(f"[reuse] verified {arm} official CPU report", flush=True)
            else:
                # Retain all classes in the input to LVISResults/top300. The
                # reporter then selects rare category precision tensors.
                command = [sys.executable, "-u", str(ROOT/"tools/report_lvis_rare_pr.py"),
                           "--predictions", provenance["inputs"][arm]["path"],
                           "--annotations", provenance["inputs"]["annotations"]["path"],
                           "--expected-apr", str(provenance["expected_apr"][arm]),
                           "--apr-tolerance", "0.0002", "--output", str(target)]
                print(f"[CPU {arm}] official full-validation rare AP; no GPU, may take several minutes", flush=True)
                subprocess.run(command, check=True, cwd=str(ROOT), env=env)
            report = load_json(target)
            validate_report(report, arm, provenance)
            identities[arm] = file_identity(target)
            save_json(receipt, {"report": identities[arm], "inputs": provenance})
            reports[arm] = report
        result = analyze(reports["A"], reports["P"])
        for record in [*provenance["inputs"].values(), *provenance["code"].values(), *identities.values()]:
            if file_identity(record["path"]) != record:
                raise ValueError("Source/report/code changed during analysis: "+record["path"])
        result.update(complete=True, provenance=provenance, official_reports=identities)
        text = format_results(result)
        save_json(output/"report.json", result)
        (output/"results.txt").write_text(text, encoding="utf-8")
        with (output/"per_class.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result["per_class"][0]))
            writer.writeheader()
            writer.writerows(result["per_class"])
        save_json(output/"STATUS.json", {"status": "COMPLETE", "report": file_identity(output/"report.json")})
        print(text, flush=True)
        print(f"[save] {output/'report.json'}\n[save] {output/'results.txt'}", flush=True)
        return result
    except Exception as error:
        save_json(output/"STATUS.json", {"status": "FAILED", "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    run(parse_args())
