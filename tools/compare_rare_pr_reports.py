#!/usr/bin/env python3
"""Compare full-validation rare AP and false positives preceding each TP.

Default mode reads existing reports only. --fill-missing-curves optionally uses
saved all-class predictions and official CPU LVIS matching for missing classes;
it never runs a detector, reads a checkpoint, or changes the source reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
# Detectron2 also exports ``tools``; always prioritize this repository.
sys.path.insert(0, str(ROOT))

DEFAULT_FOCUS = [
    "corkboard", "koala", "cocoa_(beverage)", "joystick", "roller_skate",
    "shepherd_dog", "trench_coat", "crape", "sparkler_(fireworks)", "cornbread",
]


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def file_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def display_ap(comparison):
    def number(value):
        return "n/a" if value is None else f"{value:.4f}"

    print("\n=== Full-validation AP comparison (old -> new) ===", flush=True)
    print(f"old APr={comparison['old_apr']:.4f} new APr={comparison['new_apr']:.4f} "
          f"delta={comparison['delta_apr']:+.4f}", flush=True)
    print(f"{'class':25} {'GT':>4} {'AP old/new':>19} {'dAP':>9} "
          f"{'AP50 old/new':>19} {'dAP50':>9} {'AP75 old/new':>19} {'dAP75':>9}", flush=True)
    for row in comparison["per_class"]:
        parts = []
        for metric in ("AP", "AP50", "AP75"):
            pair = number(row["old_" + metric]) + "/" + number(row["new_" + metric])
            delta = row["delta_" + metric]
            parts.append(f"{pair:>19} " + (f"{delta:+9.4f}" if delta is not None else f"{'n/a':>9}"))
        print(f"{row['name']:25} {row['gt_annotations']:4d} " + " ".join(parts), flush=True)


def display_curves(comparison):
    print("\n=== Full-validation ranked TP/FP comparison ===", flush=True)
    print("Ranks are 1-based, within one category across ALL validation images. "
          "Ignored predictions are excluded; ties retain the source's stable order.", flush=True)
    print(f"{'class':25} {'IoU':>5} {'side':>4} {'TP/FP':>12} {'recall%':>8} "
          f"{'first TP rank':>13} {'FP before first TP':>18} {'median FP before TP':>20}", flush=True)
    for row in comparison["per_class"]:
        for iou, sides in row["curves"].items():
            for label in ("old", "new"):
                summary = sides[label]
                if summary is None:
                    print(f"{row['name']:25} {iou:>5} {label:>4} MISSING_CURVE", flush=True)
                    continue
                total = f"{summary['true_positives']}/{summary['false_positives']}"
                first = summary["first_tp_rank"]
                fp = summary["fp_before_first_tp"]
                median = summary["fp_before_tp_median"]
                recall = summary["max_recall"]
                print(f"{row['name']:25} {iou:>5} {label:>4} {total:>12} "
                      f"{(f'{100 * recall:.2f}' if recall is not None else 'n/a'):>8} "
                      f"{(first if first is not None else 'NO_TP'):>13} "
                      f"{(fp if fp is not None else 'n/a'):>18} "
                      f"{(f'{median:.2f}' if median is not None else 'n/a'):>20}", flush=True)
                events = summary["every_tp"]
                print("  TP entries (TP-index, rank, FP-before, score): " + (
                    "; ".join(f"{e['tp_index']}, {e['rank']}, {e['fp_before']}, {e['score']:.6g}" for e in events)
                    if events else "none"
                ), flush=True)
    print("\nInterpretation: same TP count does not imply the same PR/AP. Compare "
          "FP preceding TP at the SAME IoU, as well as recall and AP50/AP75. "
          "TP #k is a score-order index, NOT a paired GT identity across checkpoints. "
          "This is diagnostic evidence, not a training-cause attribution or a promise of AP gain.", flush=True)


def run(args):
    from tools.rare_pr_comparison_ops import compare_reports

    paths = {label: Path(getattr(args, label + "_report")) for label in ("old", "new")}
    output = Path(args.output)
    input_paths = [*paths.values(), Path(args.annotations)]
    reports = {label: load_json(path) for label, path in paths.items()}
    prediction_paths = {}
    for label, report in reports.items():
        # Protect both recorded originals and relocated/overridden inputs.
        for key in ("predictions", "annotations"):
            if report.get(key):
                input_paths.append(Path(report[key]))
        prediction = getattr(args, label + "_predictions") or report.get("predictions")
        prediction_paths[label] = Path(prediction) if prediction else None
        if prediction:
            input_paths.append(Path(prediction))
        expected = getattr(args, "expected_" + label + "_apr")
        if expected is not None:
            actual = report.get("official_apr")
            if (not math.isfinite(expected) or actual is None
                    or not math.isclose(actual, expected, rel_tol=0, abs_tol=.02)):
                raise ValueError(f"{label} report APr={actual} differs from expected {expected}")
    if any(output.resolve() == path.resolve() for path in input_paths):
        raise ValueError("Output must not overwrite source reports, annotations or predictions")
    print(f"[load] old={paths['old']} new={paths['new']}; no model/GPU", flush=True)
    comparison = compare_reports(reports["old"], reports["new"], args.focus)
    identities = {label: file_identity(path) for label, path in paths.items()}
    comparison["source_reports"] = identities
    comparison["curve_replay"] = {}
    display_ap(comparison)
    if comparison["missing_curves"]:
        print("[missing]", json.dumps(comparison["missing_curves"], ensure_ascii=False), flush=True)
    # Preserve already-available AP results even if an optional replay is interrupted.
    save_json(output, comparison)
    if args.fill_missing_curves and comparison["missing_curves"]:
        from tools.rare_pr_curve_replay import fill_report_curves

        replay = {}
        for label in ("old", "new"):
            needed = [r["name"] for r in comparison["missing_curves"] if r["side"] == label]
            if not needed:
                continue
            if prediction_paths[label] is None:
                raise ValueError(f"Missing {label} predictions path; pass --{label}-predictions")
            print(f"[fill {label}] missing full-validation curves only; "
                  f"predictions={prediction_paths[label]}", flush=True)
            reports[label], replay[label] = fill_report_curves(
                reports[label], args.focus,
                predictions=prediction_paths[label], annotations=args.annotations,
            )
            comparison = compare_reports(reports["old"], reports["new"], args.focus)
            comparison.update(source_reports=identities, curve_replay=replay)
            save_json(output, comparison)
            print(f"[fill {label}] selected per-class AP replay: PASS", flush=True)
            print(f"[provenance {label}] Ranks describe the supplied prediction file. "
                  "AP agreement and hashes recorded now do not authenticate the original "
                  "report's checkpoint/protocol or original FP ordering.", flush=True)
    display_curves(comparison)
    print(f"[save] {output} complete={comparison['complete']}", flush=True)
    if not comparison["complete"]:
        print("[incomplete] AP comparison is available; missing PR is NOT zero FP. "
              "Use --fill-missing-curves with saved full-validation predictions, "
              "or supply reports generated with the requested --focus categories.", flush=True)
    return comparison


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-report", required=True)
    parser.add_argument("--new-report", required=True)
    parser.add_argument("--focus", nargs="+", default=DEFAULT_FOCUS)
    parser.add_argument("--expected-old-apr", type=float)
    parser.add_argument("--expected-new-apr", type=float)
    parser.add_argument("--fill-missing-curves", action="store_true")
    parser.add_argument("--old-predictions", help="Default: old report's predictions path")
    parser.add_argument("--new-predictions", help="Default: new report's predictions path")
    parser.add_argument("--annotations", default="dataset/lvis/lvis_v1_val.json")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    result = run(parse_args())
    sys.exit(0 if result["complete"] else 2)
