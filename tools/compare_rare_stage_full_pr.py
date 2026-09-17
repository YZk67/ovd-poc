#!/usr/bin/env python3
"""CPU/report-only full-IoU decomposition of a training-stage APr change.

This explains the measured PR change, not which historical loss/LR/update caused
it. No model loading, training, GPU inference or validation-based tuning.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.decoder_aux_postmortem_ops import analyze, sign


def validate_report(report, expected, label):
    actual = report.get("official_apr")
    if (not math.isfinite(expected) or not 0 <= expected <= 100 or actual is None
            or not math.isfinite(actual) or abs(actual-expected) > .0002):
        raise ValueError(f"{label} APr={actual}, expected {expected}; do not substitute a different stage")
    # Check every valid category/IoU and official/raw precision closure.
    # Missing intermediate IoUs must NOT be extrapolated from .50/.75.
    analyze(report, report, focus=(), labels={"A": label, "B": label})


def category_sensitivity(rows, delta):
    """All leave-one-category-out values; sensitivity, NOT significance testing."""
    n = len(rows)
    if n < 2:
        return {"defined": False, "reason": "Fewer than two valid classes"}
    total = math.fsum(r["delta_AP"] for r in rows)
    records = [{"excluded_category": r["name"], "gt_annotations": r["gt_annotations"],
                "excluded_apr_contribution": r["apr_contribution"],
                "remaining_mean_delta_AP": (total-r["delta_AP"])/(n-1)}
               for r in rows]
    records.sort(key=lambda r: (r["remaining_mean_delta_AP"], r["excluded_category"]))
    return {"defined": True, "remaining_classes": n-1, "min": records[0], "max": records[-1],
            "opposite_sign_exclusions": [r for r in records if sign(r["remaining_mean_delta_AP"]) != sign(delta)
                                         and sign(r["remaining_mean_delta_AP"]) != "unchanged" and sign(delta) != "unchanged"],
            "all_exclusions": records,
            "scope": "Diagnostic only: all valid classes stay in official APr. Not a CI, p-value or seed stability test."}


def display(result, limit):
    labels = result["endpoint_labels"]
    print(f"\n=== Full-IoU stage APr: {labels['A']} -> {labels['B']} ===", flush=True)
    print(f"valid={result['valid_classes']} old={result['A_apr']:.4f} new={result['B_apr']:.4f} "
          f"delta_APr={result['delta_apr']:+.4f}")
    print(f"class gains={result['positive_apr_contribution']:+.4f} "
          f"class declines={result['negative_apr_contribution']:+.4f}")
    print("\n=== Exact PR decomposition; all rare classes, all ten IoUs ===")
    for key, value in result["global"]["partition_contribution"].items():
        print(f"{key}: {value:+.6f}")
    print("These three terms sum to the official delta_APr. Shared precision is NOT FP-only causation.")
    print("\nIoU  delta_APr_at_IoU  lost-recall  gained-recall  shared-precision")
    for iou, values in result["by_iou"].items():
        p = values["macro_partition"]
        print(f"{iou} {values['macro_delta_AP']:+17.5f} {p['lost_recall_support']:+12.5f} "
              f"{p['gained_recall_support']:+14.5f} {p['shared_recall_precision']:+17.5f}")
    print("\n=== GT strata; official denominator unchanged ===")
    for row in result["gt_strata"]:
        print(row)
    for outcome, reverse in (("decline", False), ("gain", True)):
        print(f"\n=== Largest {outcome}s: class GT old_AP new_AP delta_AP APr_contribution ===")
        rows = [r for r in result["per_class"] if r["outcome"] == outcome]
        rows.sort(key=lambda r: ((-r['delta_AP'] if reverse else r['delta_AP']), r['category_id']))
        for row in rows[:limit]:
            print(f"{row['name']:25} {row['gt_annotations']:4d} {row['old_AP']:8.3f} {row['new_AP']:8.3f} "
                  f"{row['delta_AP']:+9.3f} {row['apr_contribution']:+10.5f}")
    sensitivity = result["category_sensitivity"]
    print("\n=== Leave-one-category-out sensitivity; NOT significance ===")
    if sensitivity["defined"]:
        print("min:", sensitivity["min"])
        print("max:", sensitivity["max"])
        print("opposite_sign_exclusions:", sensitivity["opposite_sign_exclusions"])
    else:
        print(sensitivity["reason"])
    print("\nNo historical training source identified by this PR decomposition. "
          "No loss/formula change or new training is selected automatically.", flush=True)


def run(args):
    if (args.top_classes < 1 or not args.old_label.strip() or not args.new_label.strip()
            or args.old_label == args.new_label):
        raise ValueError("Need positive top-classes and nonempty distinct endpoint labels")
    paths = {s: Path(getattr(args, s+"_report")).resolve() for s in ("old", "new")}
    if paths["old"] == paths["new"] or paths["old"].samefile(paths["new"]):
        raise ValueError("Stage reports must be different files")
    source_ids = {s: file_identity(p) for s, p in paths.items()}
    reports = {s: load_json(p) for s, p in paths.items()}
    output = Path(args.output).resolve()
    protected = list(paths.values())
    for r in reports.values():
        protected += [Path(r[k]).resolve() for k in ("predictions", "annotations") if r.get(k)]
    if output in protected or output.exists():
        raise ValueError("Refusing to overwrite an input or existing output")
    for side in ("old", "new"):
        validate_report(reports[side], getattr(args, "expected_"+side+"_apr"), getattr(args, side+"_label"))
    result = analyze(reports["old"], reports["new"], focus=(),
                     labels={"A": args.old_label, "B": args.new_label})
    result.pop("original_focus")  # No inherited auxiliary-ablation focus categories.
    result.update(category_sensitivity=category_sensitivity(result["per_class"], result["delta_apr"]),
                  source_reports=source_ids,
                  gpu_inference=False, training_updates=0,
                  interpretation="PR-accounting decomposition, not attribution to a training loss/LR or proof of statistical significance.")
    result["scope"].append("Stage labels are supplied by the user; matching APr does not independently authenticate checkpoint/config provenance.")
    for side, path in paths.items():
        if file_identity(path) != source_ids[side]:
            raise ValueError("Input report changed during comparison; no result saved")
    save_json(output, result)
    display(result, args.top_classes)
    print(f"[save] {output}", flush=True)
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-report", required=True)
    p.add_argument("--new-report", required=True)
    p.add_argument("--expected-old-apr", required=True, type=float)
    p.add_argument("--expected-new-apr", required=True, type=float)
    p.add_argument("--old-label", default="8ep")
    p.add_argument("--new-label", default="12ep")
    p.add_argument("--top-classes", type=int, default=20)
    p.add_argument("--output", required=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
