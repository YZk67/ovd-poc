#!/usr/bin/env python3
"""One CPU-only pipeline: two saved prediction JSONs -> official rare PR -> stage comparison.

No checkpoint loading, inference, or training. Both prediction inputs must exist
before any evaluation starts. Use a new output directory to preserve past reports.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-predictions", required=True, help="Earlier checkpoint, e.g. saved 8ep JSON")
    parser.add_argument("--new-predictions", required=True, help="Later checkpoint, e.g. saved 12ep JSON")
    parser.add_argument("--expected-old-apr", required=True, type=float)
    parser.add_argument("--expected-new-apr", required=True, type=float)
    parser.add_argument("--annotations", default=str(ROOT / "dataset/lvis/lvis_v1_val.json"))
    parser.add_argument("--top-declines", type=int, default=20)
    parser.add_argument("--full-iou", action="store_true",
                        help="All ten IoUs; exact full-APr recall-support/precision decomposition, not only AP50/AP75")
    parser.add_argument("--old-label", default="earlier checkpoint")
    parser.add_argument("--new-label", default="later checkpoint")
    parser.add_argument("--old-report", help="Optional existing complete rare PR report; read-only reuse")
    parser.add_argument("--new-report", help="Optional existing complete rare PR report; read-only reuse")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def run(args):
    import math
    from tools.compare_rare_pr_reports import file_identity, load_json, save_json

    inputs = {name: Path(getattr(args, name)).expanduser().resolve()
              for name in ("old_predictions", "new_predictions", "annotations")}
    missing = [f"{name}: {path}" for name, path in inputs.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing inputs; NO evaluation or GPU inference started:\n" + "\n".join(missing)
            + "\nA log/APr number cannot reconstruct PR curves. Supply the earlier checkpoint's "
            "saved all-class prediction JSON; do not substitute the training directory's "
            "JSON if a later evaluation overwrote it."
        )
    if os.path.samefile(inputs["old_predictions"], inputs["new_predictions"]):
        raise ValueError("Old/new predictions must be different saved files")
    if args.top_declines < 1:
        raise ValueError("--top-declines must be positive")
    if not args.old_label.strip() or not args.new_label.strip() or args.old_label == args.new_label:
        raise ValueError("Stage labels must be nonempty and different")
    for label in ("old", "new"):
        value = getattr(args, "expected_" + label + "_apr")
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f"expected {label} APr must be finite AP points in [0, 100]")
    output = Path(args.output_dir).expanduser().resolve()
    supplied = {s: Path(getattr(args, s+"_report")).expanduser().resolve() if getattr(args, s+"_report") else None
                for s in ("old", "new")}
    # Validate BOTH reusable reports before launching either evaluator.
    for side, path in supplied.items():
        if path is None:
            continue
        report = load_json(path)
        prediction = inputs[side+"_predictions"]
        if (Path(report.get("predictions", "")).resolve() != prediction
                or report.get("prediction_bytes") != prediction.stat().st_size
                or Path(report.get("annotations", "")).resolve() != inputs["annotations"]):
            raise ValueError(f"{side} reusable report source path/size differs from supplied predictions/annotations")
        if args.full_iou:
            from tools.compare_rare_stage_full_pr import validate_report
            validate_report(report, getattr(args, "expected_"+side+"_apr"), side)
        else:
            from tools.rare_pr_comparison_ops import compare_reports
            if not math.isclose(report["official_apr"], getattr(args, "expected_"+side+"_apr"), rel_tol=0, abs_tol=.02):
                raise ValueError(f"{side} reusable report has wrong APr")
            if not compare_reports(report, report)["complete"]:
                raise ValueError(f"{side} reusable report lacks all rare .50/.75 curves")
    outputs = {label: supplied[label] or output / (label + "_report.json") for label in ("old", "new")}
    outputs["comparison"] = output / "comparison.json"
    if args.full_iou:
        outputs["full_pr"] = output / "report.json"
    manifest_path = output/"inputs.json"
    complete_path = output/"COMPLETE.json"
    protected = [*inputs.values(), *[p for p in supplied.values() if p is not None]]
    if any(output == p or output in p.parents or p in output.parents or output == p.parent for p in protected):
        raise ValueError("Use a separate output directory; inputs remain read-only")
    created = [manifest_path, complete_path,
               *[p for k, p in outputs.items() if k not in supplied or supplied[k] is None]]
    for path in created:
        if path.resolve() in protected or path.exists():
            raise ValueError(f"Refusing to overwrite {path}; choose a new output directory. "
                             "Existing reports can be compared with compare_rare_pr_reports.py.")
    source_ids = {k: file_identity(v) for k, v in inputs.items()}
    report_ids = {k: file_identity(v) for k, v in supplied.items() if v is not None}
    code_ids = {name: file_identity(ROOT/"tools"/name) for name in (
        "run_rare_stage_comparison.py", "report_lvis_rare_pr.py",
        "compare_rare_pr_reports.py", "rare_pr_comparison_ops.py",
        "compare_rare_stage_full_pr.py", "decoder_aux_postmortem_ops.py")}
    output.mkdir(parents=True, exist_ok=True)
    save_json(manifest_path, {"inputs": source_ids, "reused_reports": report_ids, "code": code_ids,
                             "expected_apr": {s: getattr(args, "expected_"+s+"_apr") for s in ("old", "new")},
                             "stage_labels": {s: getattr(args, s+"_label") for s in ("old", "new")},
                             "full_iou": args.full_iou, "gpu_inference": False, "training_updates": 0,
                             "provenance_limit": "Supplied stage labels and expected APr; no independent checkpoint/config authentication. "
                             "Reused legacy report path/size/AP checks cannot authenticate its original prediction hash."})
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    for label in ("old", "new"):
        if supplied[label] is not None:
            print(f"[reuse {label}] {supplied[label]}; no evaluator pass", flush=True)
            continue
        expected = getattr(args, "expected_" + label + "_apr")
        print(f"[stage {label}] CPU LVIS evaluation, expected APr={expected:.4f}; "
              "save all rare curves once", flush=True)
        subprocess.run([
            sys.executable, "-u", str(ROOT / "tools/report_lvis_rare_pr.py"),
            "--predictions", str(inputs[label + "_predictions"]),
            "--annotations", str(inputs["annotations"]),
            "--expected-apr", str(expected), "--all-curves",
            *(["--all-iou-curves", "--apr-tolerance", "0.0002"] if args.full_iou else []),
            "--output", str(outputs[label]),
        ], cwd=ROOT, env=env, check=True)
    subprocess.run([
        sys.executable, "-u", str(ROOT / "tools/compare_rare_pr_reports.py"),
        "--old-report", str(outputs["old"]), "--new-report", str(outputs["new"]),
        "--expected-old-apr", str(args.expected_old_apr),
        "--expected-new-apr", str(args.expected_new_apr),
        "--top-declines", str(args.top_declines), "--output", str(outputs["comparison"]),
    ], cwd=ROOT, env=env, check=True)
    if args.full_iou:
        subprocess.run([
            sys.executable, "-u", str(ROOT / "tools/compare_rare_stage_full_pr.py"),
            "--old-report", str(outputs["old"]), "--new-report", str(outputs["new"]),
            "--expected-old-apr", str(args.expected_old_apr), "--expected-new-apr", str(args.expected_new_apr),
            "--old-label", args.old_label, "--new-label", args.new_label,
            "--top-classes", str(args.top_declines), "--output", str(outputs["full_pr"]),
        ], cwd=ROOT, env=env, check=True)
    for source in [*source_ids.values(), *report_ids.values(), *code_ids.values()]:
        if file_identity(source["path"]) != source:
            raise ValueError("Source changed during comparison; do not interpret the outputs")
    save_json(complete_path, {"source_files_unchanged": True, "gpu_inference": False,
                              "training_updates": 0, "full_iou": args.full_iou})
    print(f"[done] {outputs.get('full_pr', outputs['comparison'])}; CPU only, source predictions unchanged", flush=True)
    return outputs


if __name__ == "__main__":
    run(parse_args())
