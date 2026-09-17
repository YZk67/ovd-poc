#!/usr/bin/env python3
"""CPU-only full rare-class/GT decomposition of the completed decoder A/B trial."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.decoder_aux_outcome_ops import build_top300_transitions, summarize_gradient_logs, summarize_pr
from tools.diagnose_rare_fp_regions import fingerprint
from tools.lvis_gt_matching import match_panel_gt
from tools.rare_pr_comparison_ops import compare_reports
from tools.run_rare_stage_comparison import parse_args as parse_pr_args, run as run_pr
from tools.train_decoder_aux_arm import read_manifest


def verified_identity(expected, label):
    actual = file_identity(expected["path"])
    if actual != expected:
        raise ValueError(f"{label} identity changed")
    return actual


def validate_trial(directory):
    manifest = read_manifest(directory / "manifest.json")
    summary = load_json(directory / "summary.json")
    if (not summary.get("complete") or summary.get("manifest_fingerprint") != manifest["fingerprint"]
            or summary.get("updates_per_arm") != manifest["updates"]):
        raise ValueError("Need a completed paired decoder auxiliary-gradient trial")
    if manifest["updates"] != 500:
        raise ValueError("Formal outcome decomposition requires the completed 500-update trial")
    for arm in ("A", "B"):
        verified_identity(summary["evaluations"][arm]["predictions"], f"{arm} predictions")
        metrics = summary["evaluations"][arm]["metrics"]
        if set(metrics) != {"AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf"}:
            raise ValueError("Incomplete official metrics")
    expected = {k: summary["evaluations"]["B"]["metrics"][k]-summary["evaluations"]["A"]["metrics"][k]
                for k in summary["delta_B_minus_A"]}
    if any(not math.isclose(expected[k], summary["delta_B_minus_A"][k], abs_tol=1e-10)
           for k in expected):
        raise ValueError("Trial metric deltas do not close")
    return manifest, summary


def read_selected_predictions(identity, rare_images, known_images, known_categories):
    verified_identity(identity, "prediction JSON")
    with open(identity["path"], encoding="utf-8") as handle:
        source = json.load(handle)
    if not isinstance(source, list):
        raise ValueError("Predictions must be an LVIS result list")
    selected, counts = [], Counter()
    for row in source:
        if (row.get("image_id") not in known_images or row.get("category_id") not in known_categories
                or not isinstance(row.get("bbox"), list) or len(row["bbox"]) != 4
                or not math.isfinite(float(row.get("score", float("nan"))))):
            raise ValueError("Invalid prediction object")
        counts[row["image_id"]] += 1
        if row["image_id"] in rare_images:
            selected.append(row)
    if any(value > 300 for value in counts.values()):
        raise ValueError("Prediction JSON exceeds global top-300 per image")
    return selected


def load_gradient_logs(directory, updates, start):
    result, identities = {}, {}
    for arm in ("A", "B"):
        ranks = []
        for rank in range(4):
            path = directory / arm / f"updates_rank{rank}.jsonl"
            identities[f"{arm}_rank{rank}"] = file_identity(path)
            rows = [json.loads(line) for line in path.open()]
            if len(rows) != updates or any(
                    row["update"] != i+1 or row["iteration"] != start+i
                    or row["remove"] != (arm == "B") for i, row in enumerate(rows)):
                raise ValueError("Incomplete gradient intervention log")
            ranks.append(rows)
        result[arm] = ranks
    return result, identities


def display(report):
    macro = report["pr_comparison"]["macro_attribution"]
    print("\n=== All-valid rare-class AP attribution: B - A ===")
    print(f"APr {report['metrics']['A']['APr']:.4f} -> {report['metrics']['B']['APr']:.4f} "
          f"({report['metrics']['delta_B_minus_A']['APr']:+.4f}); "
          f"classes improved/declined/unchanged={macro['classes_improved']}/"
          f"{macro['classes_declined']}/{macro['classes_unchanged']}")
    print(f"positive contribution={macro['positive_contribution']:+.4f}, "
          f"negative contribution={macro['negative_contribution']:+.4f}")
    print(f"{'GT':>7} {'classes':>8} {'mean dAP':>11} {'median dAP':>13} {'APr contribution':>18}")
    for row in macro["gt_strata"]:
        print(f"{row['gt_range']:>7} {row['classes']:8d} {row['mean_delta_AP']:+11.4f} "
              f"{row['median_delta_AP']:+13.4f} {row['apr_contribution']:+18.4f}")
    print("\nLargest rare-class declines:")
    transition_rows = {row["category_id"]: row
                       for row in report["top300_transitions"]["per_class_by_iou"]["0.50"]}
    for row in macro["per_class"][:20]:
        transition = transition_rows[row["category_id"]]
        pr = transition["full_validation_pr"]
        print(f"{row['name']:25} GT={row['gt_annotations']:3d} "
              f"A/B={row['old_AP']:7.3f}/{row['new_AP']:7.3f} dAP={row['delta_AP']:+8.3f} "
              f"lost/gained@.50={transition['lost']}/{transition['gained']} "
              f"dRecall={pr['delta_max_recall_points']:+7.2f} "
              f"sharedPR={pr['shared_recall_precision_points']:+7.2f} "
              f"mean-dFP={pr['same_recall_mean_delta_fp_before']}")
    for key, row in report["top300_transitions"]["summary_by_iou"].items():
        print(f"\n=== Official rare GT transitions @IoU={key}, final top-300 only ===")
        print(f"transitions={row['transitions']} TP {row['A_tp']}->{row['B_tp']} ({row['delta_tp']:+d}) "
              f"FP {row['A_fp']}->{row['B_fp']} ({row['delta_fp']:+d})")
        print(f"lost reasons={row['loss_reasons']}")
        print(f"gained reasons={row['gain_reasons']}")
        print("declining-class diagnostics:",
              report["top300_transitions"]["declining_class_summary_by_iou"][key])
        print("PR delta partition:", report["pr_delta_by_iou"][key])
    print("\n[limit] final predictions cannot separate proposal absence from a correct pair dropped before top-300.")
    print("[limit] gradient logs are global per-step norms, so per-class gradient/AP correlation is not identifiable.")


def run(args):
    trial = Path(args.trial_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    annotations = Path(args.annotations).expanduser().resolve()
    if output.exists():
        raise ValueError("Use a NEW output directory; source A/B predictions and prior reports are read-only")
    if not annotations.is_file():
        raise FileNotFoundError(annotations)
    if trial == output or trial in output.parents:
        raise ValueError("Analysis output must be outside the read-only A/B trial directory")
    manifest, summary = validate_trial(trial)
    output.mkdir(parents=True, exist_ok=False)
    pr_dir = output / "rare_pr"
    run_pr(parse_pr_args([
        "--old-predictions", summary["evaluations"]["A"]["predictions"]["path"],
        "--new-predictions", summary["evaluations"]["B"]["predictions"]["path"],
        "--expected-old-apr", str(summary["evaluations"]["A"]["metrics"]["APr"]),
        "--expected-new-apr", str(summary["evaluations"]["B"]["metrics"]["APr"]),
        "--annotations", str(annotations), "--top-declines", str(args.top_declines),
        "--output-dir", str(pr_dir),
    ]))
    old_report, new_report = (load_json(pr_dir/f"{name}_report.json") for name in ("old", "new"))
    comparison = compare_reports(old_report, new_report, None)
    if not comparison["complete"]:
        raise ValueError("All-rare PR comparison is incomplete")
    dataset = load_json(annotations)
    known_images = {row["id"] for row in dataset["images"]}
    known_categories = {row["id"] for row in dataset["categories"]}
    rare_ids = {row["id"] for row in dataset["categories"] if row.get("frequency") == "r"}
    rare_images = {row["image_id"] for row in dataset["annotations"] if row["category_id"] in rare_ids}
    selected, matches = {}, {}
    for arm in ("A", "B"):
        print(f"[match {arm}] load final predictions; rare-GT images={len(rare_images)}", flush=True)
        selected[arm] = read_selected_predictions(summary["evaluations"][arm]["predictions"],
                                                   rare_images, known_images, known_categories)
        matches[arm] = match_panel_gt(dataset, rare_images, selected[arm],
                                      iou_thresholds=(.5, .75), max_dets=300)
    transitions = build_top300_transitions(matches["A"], matches["B"], dataset,
                                           selected["A"], selected["B"], comparison)
    gradient_logs, gradient_identities = load_gradient_logs(trial, manifest["updates"], manifest["start"])
    report = {
        "complete": True,
        "inputs": {"trial_summary": file_identity(trial/"summary.json"),
                   "trial_manifest": file_identity(trial/"manifest.json"),
                   "annotations": file_identity(annotations),
                   "rare_pr_A": file_identity(pr_dir/"old_report.json"),
                   "rare_pr_B": file_identity(pr_dir/"new_report.json"),
                   "rare_pr_comparison_top_declines": file_identity(pr_dir/"comparison.json"),
                   "A_predictions": summary["evaluations"]["A"]["predictions"],
                   "B_predictions": summary["evaluations"]["B"]["predictions"],
                   "gradient_logs": gradient_identities},
        "metrics": {"A": summary["evaluations"]["A"]["metrics"],
                    "B": summary["evaluations"]["B"]["metrics"],
                    "delta_B_minus_A": summary["delta_B_minus_A"]},
        "pr_comparison": comparison,
        "pr_delta_by_iou": summarize_pr(comparison),
        "top300_transitions": transitions,
        "gradient_intervention": summarize_gradient_logs(gradient_logs),
        "identifiability": {
            "proposal_vs_pre_top300_classification": "not_identifiable_from_final_prediction_JSONs",
            "per_class_gradient_correlation": "not_identifiable_from_global_per_step_gradient_logs",
            "supported_reasons": ["correct_class_iou_candidate_unmatched",
                                  "geometry_present_true_class_absent_from_final_top300",
                                  "no_iou_eligible_detection_in_final_top300"],
            "interpretation": "Observed B-A effects of one paired 500-update trajectory; not a proof for all seeds/stages.",
        },
    }
    report["fingerprint"] = fingerprint(report["inputs"])
    save_json(output/"report.json", report)
    display(report)
    print(f"[save] {output/'report.json'}; CPU only, no model inference/training", flush=True)
    return report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trial-dir", required=True)
    p.add_argument("--output-dir", required=True, help="NEW directory; do not pre-create")
    p.add_argument("--annotations", default="/root/autodl-tmp/dataset/lvis/lvis_v1_val.json")
    p.add_argument("--top-declines", type=int, default=20)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
