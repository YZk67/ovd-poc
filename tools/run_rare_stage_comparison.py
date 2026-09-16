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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-predictions", required=True, help="Earlier checkpoint, e.g. saved 8ep JSON")
    parser.add_argument("--new-predictions", required=True, help="Later checkpoint, e.g. saved 10ep JSON")
    parser.add_argument("--expected-old-apr", required=True, type=float)
    parser.add_argument("--expected-new-apr", required=True, type=float)
    parser.add_argument("--annotations", default=str(ROOT / "dataset/lvis/lvis_v1_val.json"))
    parser.add_argument("--top-declines", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def run(args):
    import math

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
    for label in ("old", "new"):
        value = getattr(args, "expected_" + label + "_apr")
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f"expected {label} APr must be finite AP points in [0, 100]")
    output = Path(args.output_dir).expanduser().resolve()
    outputs = {label: output / (label + "_report.json") for label in ("old", "new")}
    outputs["comparison"] = output / "comparison.json"
    for path in outputs.values():
        if path.resolve() in inputs.values() or path.exists():
            raise ValueError(f"Refusing to overwrite {path}; choose a new output directory. "
                             "Existing reports can be compared with compare_rare_pr_reports.py.")
    output.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    for label in ("old", "new"):
        expected = getattr(args, "expected_" + label + "_apr")
        print(f"[stage {label}] CPU LVIS evaluation, expected APr={expected:.4f}; "
              "save all rare curves once", flush=True)
        subprocess.run([
            sys.executable, "-u", str(ROOT / "tools/report_lvis_rare_pr.py"),
            "--predictions", str(inputs[label + "_predictions"]),
            "--annotations", str(inputs["annotations"]),
            "--expected-apr", str(expected), "--all-curves",
            "--output", str(outputs[label]),
        ], cwd=ROOT, env=env, check=True)
    subprocess.run([
        sys.executable, "-u", str(ROOT / "tools/compare_rare_pr_reports.py"),
        "--old-report", str(outputs["old"]), "--new-report", str(outputs["new"]),
        "--expected-old-apr", str(args.expected_old_apr),
        "--expected-new-apr", str(args.expected_new_apr),
        "--top-declines", str(args.top_declines), "--output", str(outputs["comparison"]),
    ], cwd=ROOT, env=env, check=True)
    print(f"[done] {outputs['comparison']}; CPU only, source predictions unchanged", flush=True)
    return outputs


if __name__ == "__main__":
    run(parse_args())
