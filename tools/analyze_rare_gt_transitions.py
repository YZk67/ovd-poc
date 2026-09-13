#!/usr/bin/env python3
"""CPU-only attribution of native old/new rare-GT detection transitions.

Reads saved pairing rows and prediction JSONs, not model weights or tensor dumps.
Official LVIS matching determines TP/FN. Cached all-query coverage distinguishes
missing geometry from missing top-k candidates. Full-validation AP, when supplied,
is joined by category; it is never allocated causally to individual missed GT.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as handle:
            name = handle.name
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(name, path)
    finally:
        if name is not None and os.path.exists(name):
            os.unlink(name)


def unique_ids(rows, key, label):
    ids = [row[key] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate {label} IDs")
    return set(ids)


def iou_key(value):
    return f"{value:.2f}" if value == round(value, 2) else f"{value:.12g}"


def validate_source(source, dataset):
    """Reject incomplete/wrong-panel reports before evaluating any predictions."""
    if source.get("complete") is not True:
        raise ValueError("Pairing source is incomplete; require complete=true")
    image_ids = unique_ids(dataset["images"], "id", "annotation image")
    category_ids = unique_ids(dataset["categories"], "id", "category")
    unique_ids(dataset["annotations"], "id", "GT annotation")
    rare_ids = {c["id"] for c in dataset["categories"] if c.get("frequency") == "r"}
    if not rare_ids:
        raise ValueError("Annotations contain no rare categories")
    for ann in dataset["annotations"]:
        if ann["image_id"] not in image_ids or ann["category_id"] not in category_ids:
            raise ValueError("GT annotation references unknown image/category")
    panel = source["panel"]
    selected = panel["image_ids"]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Panel image IDs are empty or duplicated")
    if not set(selected).issubset(image_ids):
        raise ValueError("Panel contains image IDs absent from annotations")
    if set(panel["rare_category_ids"]) != rare_ids:
        raise ValueError("Panel rare category IDs differ from annotations")
    rare_images = {a["image_id"] for a in dataset["annotations"] if a["category_id"] in rare_ids}
    if set(panel["rare_image_ids"]) != rare_images or not rare_images.issubset(selected):
        raise ValueError("Panel must retain every rare-GT image; cannot join full rare AP otherwise")

    native = [source["panel_pr"][v] for v in ("old_old", "new_new")]
    for report in native:
        if set(report["image_ids"]) != set(selected):
            raise ValueError("Native panel PR image IDs differ from pairing panel")
        if set(report["rare_category_ids"]) != rare_ids:
            raise ValueError("Native panel PR rare category IDs differ")
    max_dets = native[0]["max_dets"]
    if isinstance(max_dets, bool) or not isinstance(max_dets, int) or max_dets <= 0:
        raise ValueError("Invalid source max_dets")
    thresholds = sorted(float(t) for t in native[0]["iou_thresholds"])
    if not thresholds or any(not math.isfinite(t) or not 0 < t <= 1 for t in thresholds):
        raise ValueError("Invalid source IoU thresholds")
    keys = [iou_key(t) for t in thresholds]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate source IoU thresholds")
    for report in native:
        if report["max_dets"] != max_dets or sorted(report["iou_thresholds"]) != thresholds:
            raise ValueError("Old/new native matching protocols differ")
        if set(report["summary_by_iou"]) != set(keys):
            raise ValueError("Native summary IoU coverage differs")
    counts = Counter(a["category_id"] for a in dataset["annotations"])
    meta = {
        category["id"]: {
            **category, "category_id": category["id"], "class_index": index,
            "gt_annotations": counts[category["id"]],
        }
        for index, category in enumerate(sorted(dataset["categories"], key=lambda c: c["id"]))
    }
    return selected, thresholds, max_dets, meta


def validate_manifest(manifest, source, annotations_sha256, image_ids, thresholds, max_dets):
    inputs = manifest["inputs"]
    fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    if manifest.get("fingerprint") != fingerprint or source.get("fingerprint") != fingerprint:
        raise ValueError("Pairing report/manifest fingerprint mismatch")
    if inputs["annotations_sha256"] != annotations_sha256:
        raise ValueError("Annotations SHA256 differs from the saved pairing manifest")
    if (set(inputs["image_ids"]) != set(image_ids)
            or sorted(inputs["iou_thresholds"]) != thresholds or inputs["max_dets"] != max_dets):
        raise ValueError("Manifest panel/matching protocol differs from report")


def validate_replay(actual, expected, variant):
    """Recomputed matching must reproduce the saved native panel totals/classes."""
    for key, row in expected["summary_by_iou"].items():
        checked = actual["summary_by_iou"][key]
        for metric in ("num_gt", "true_positives", "false_positives", "macro_recall_valid_categories"):
            if checked[metric] != row[metric]:
                raise ValueError(f"{variant} IoU={key} {metric}: replay {checked[metric]} != source {row[metric]}")
        target, observed = row["macro_recall"], checked["macro_recall"]
        if ((target is None) != (observed is None)
                or (target is not None and not math.isclose(target, observed, rel_tol=0, abs_tol=1e-10))):
            raise ValueError(f"{variant} IoU={key} macro_recall differs from saved report")
        by_class = {}
        for gt in actual["gt_by_iou"][key]:
            if not gt["gt_ignore"]:
                count = by_class.setdefault(gt["category_id"], [0, 0])
                count[0] += 1
                count[1] += int(gt["matched"])
        for category in expected["per_class"]:
            curve = category["iou_curves"][key]
            got = by_class.pop(category["category_id"], [0, 0])
            if got != [curve["num_gt"], curve["true_positives"]]:
                raise ValueError(f"{variant} IoU={key} category={category['category_id']} GT/TP replay differs")
        if by_class:
            raise ValueError(f"{variant} contains categories absent from saved PR")
    actual_classes = {row["category_id"]: row for row in actual["per_class"]}
    if set(actual_classes) != {row["category_id"] for row in expected["per_class"]}:
        raise ValueError(f"{variant} per-class PR category coverage differs")
    for category in expected["per_class"]:
        reproduced = actual_classes[category["category_id"]]
        for key, curve in category["iou_curves"].items():
            other = reproduced["iou_curves"][key]
            for metric in ("num_gt", "true_positives", "false_positives", "ignored_detections"):
                if other[metric] != curve[metric]:
                    raise ValueError(f"{variant} category={category['category_id']} {key} {metric} differs")
            if other["curve"] != curve["curve"]:
                raise ValueError(f"{variant} category={category['category_id']} {key} ranked score/TP/FP curve differs")


def validate_prediction_scope(predictions, selected, category_ids, max_dets):
    if not isinstance(predictions, list):
        raise ValueError("Predictions must be an LVIS detection list")
    image_set = set(selected)
    counts = Counter()
    for prediction in predictions:
        if prediction["image_id"] not in image_set or prediction["category_id"] not in category_ids:
            raise ValueError("Prediction references image/category outside the saved panel")
        counts[prediction["image_id"]] += 1
    if any(count > max_dets for count in counts.values()):
        raise ValueError("Saved native predictions exceed per-image top-k; wrong prediction JSON?")


def show_report(report):
    def number(value, width=9, signed=False):
        if value is None:
            return f"{'n/a':>{width}}"
        return f"{value:+{width}.4f}" if signed else f"{value:{width}.4f}"

    def packed_reasons(counts):
        primary = ("no_eligible_query", "correct_pair_below_topk", "matching_competition")
        values = [counts[name] for name in primary]
        values.append(sum(value for name, value in counts.items() if name not in primary))
        return "/".join(map(str, values))

    ap = report.get("ap_comparison")
    if ap is not None:
        print("\nFull-validation AP reports (separate from panel recall): "
              f"old APr={ap['official_apr_old']:.4f} new APr={ap['official_apr_new']:.4f} "
              f"delta={ap['delta_apr']:+.4f}", flush=True)
    for key, summary in report["summary_by_iou"].items():
        print(f"\n=== Actual rare GT transitions @IoU={key} ===", flush=True)
        print(f"GT={summary['num_gt']} valid_classes={summary['num_valid_classes']} "
              f"ignored_GT={summary['num_ignored_gt']}", flush=True)
        print("transitions:", summary["transitions"], flush=True)
        old, new = summary["old_summary"], summary["new_summary"]
        print(f"old TP/FP={old['true_positives']}/{old['false_positives']} "
              f"new TP/FP={new['true_positives']}/{new['false_positives']} "
              f"netTP={summary['net_tp_change']:+d} netFP={summary['net_fp_change']:+d} "
              f"delta_macro_recall_pp={number(summary['macro_recall_change_points'], signed=True).strip()}", flush=True)
        print(f"{'reason':38} {'lost':>6} {'gained':>6} {'loss MR-pp':>11} {'gain MR-pp':>11} {'net MR-pp':>11}", flush=True)
        for reason, count in summary["loss_reason_counts"].items():
            loss = summary["loss_reason_macro_recall_points"][reason]
            gain = summary["gain_reason_macro_recall_points"][reason]
            print(f"{reason:38} {count:6d} {summary['gain_reason_counts'][reason]:6d} "
                  f"{loss:+11.4f} {gain:+11.4f} {loss + gain:+11.4f}", flush=True)
        print("closure:", summary["invariants"], flush=True)
        print("\nPer-class transitions + independent full-validation AP deltas:", flush=True)
        print("L/G reasons = no-box / below-topk / matching / unresolved; MR = macro recall.", flush=True)
        print(f"{'class':27} {'GT':>4} {'TP old/new':>10} {'L':>4} {'G':>4} "
              f"{'L reasons':>13} {'G reasons':>13} {'MR contrib':>10} {'delta AP':>9} {'APr contrib':>11}", flush=True)
        # Also show AP changes without TP transitions (precision/ranking/other IoUs).
        classes = sorted(report["per_class_by_iou"][key], key=lambda row: (
            row["delta_AP"] if row["delta_AP"] is not None else row["macro_recall_contribution_points"],
            row["category_id"],
        ))
        for row in classes:
            if not (row["losses"] or row["gains"] or row["delta_AP"] not in (None, 0)):
                continue
            tp = f"{row['old_tp']}/{row['new_tp']}"
            print(f"{row['name']:27} {row['num_gt']:4d} {tp:>10} {row['losses']:4d} {row['gains']:4d} "
                  f"{packed_reasons(row['loss_reason_counts']):>13} {packed_reasons(row['gain_reason_counts']):>13} "
                  f"{number(row['macro_recall_contribution_points'], 10, True)} "
                  f"{number(row['delta_AP'], 9, True)} {number(row['apr_contribution_points'], 11, True)}", flush=True)
    print("\nScope: selected-panel official one-to-one TP/FN; sampled negatives for FP. "
          "Macro-recall contributions close exactly, but are NOT APr contributions. "
          "Full-validation per-class AP deltas are independent joined measurements. "
          "No training-cause or expected AP gain is established.", flush=True)


def run(args):
    from tools.lvis_gt_matching import match_panel_gt
    from tools.rare_transition_ops import build_transition_report

    source_path = Path(args.source_json)
    paths = {
        "pairing_report": source_path,
        "annotations": Path(args.annotations),
        "old_predictions": Path(args.old_predictions) if args.old_predictions else source_path.parent / "old_old_predictions.json",
        "new_predictions": Path(args.new_predictions) if args.new_predictions else source_path.parent / "new_new_predictions.json",
    }
    if bool(args.old_ap_report) != bool(args.new_ap_report):
        raise ValueError("Supply both --old-ap-report and --new-ap-report, or neither")
    if args.old_ap_report:
        paths.update(old_ap_report=Path(args.old_ap_report), new_ap_report=Path(args.new_ap_report))
    manifest_path = Path(args.manifest) if args.manifest else source_path.parent / "pairing_cache" / "manifest.json"
    if args.manifest or manifest_path.exists():
        paths["manifest"] = manifest_path
    output = Path(args.output)
    if any(output.resolve() == path.resolve() for path in paths.values()):
        raise ValueError("Output must not overwrite any input artifact")
    print("[load] saved pairing report + annotations; CPU only, no checkpoints/features", flush=True)
    source, dataset = load_json(source_path), load_json(paths["annotations"])
    selected, thresholds, max_dets, meta = validate_source(source, dataset)
    provenance = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    if "manifest" in paths:
        validate_manifest(load_json(paths["manifest"]), source, provenance["annotations"]["sha256"],
                          selected, thresholds, max_dets)
    else:
        print("[warning] No original manifest: annotation hash provenance cannot be verified; "
              "panel IDs and official per-class match counts will still be checked.", flush=True)

    matches = {}
    for label, variant in (("old", "old_old"), ("new", "new_new")):
        print(f"[match {label}] reading {paths[label + '_predictions']}", flush=True)
        predictions = load_json(paths[label + "_predictions"])
        validate_prediction_scope(predictions, selected, set(meta), max_dets)
        matches[label] = match_panel_gt(dataset, selected, predictions,
                                       iou_thresholds=thresholds, max_dets=max_dets)
        validate_replay(matches[label], source["panel_pr"][variant], variant)
        print(f"[match {label}] native per-class GT/TP, FP and macro recall replay: PASS", flush=True)
        del predictions

    ap_reports = None
    if args.old_ap_report:
        ap_reports = {label: load_json(paths[label + "_ap_report"]) for label in ("old", "new")}
        for report in ap_reports.values():
            if report["max_dets"] != max_dets:
                raise ValueError("Full AP report max_dets differs from pairing protocol")
    report = build_transition_report(matches["old"], matches["new"], source["rows"], meta,
                                     ap_reports=ap_reports)
    report.update(
        provenance=provenance, pairing_fingerprint=source.get("fingerprint"),
        annotations_manifest_verified="manifest" in paths,
        native_matching_replay_verified=True, complete=True,
        protocol={"image_ids": selected, "max_dets": max_dets, "iou_thresholds": thresholds},
    )
    show_report(report)
    write_json(output, report)
    print(f"[save] {output}", flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", required=True, help="Completed detector_tpa_pairing/report.json")
    parser.add_argument("--annotations", default="dataset/lvis/lvis_v1_val.json")
    parser.add_argument("--old-predictions", help="Default: old_old_predictions.json beside source")
    parser.add_argument("--new-predictions", help="Default: new_new_predictions.json beside source")
    parser.add_argument("--manifest", help="Default: sibling pairing_cache/manifest.json when present")
    parser.add_argument("--old-ap-report", help="Old CURRENT-PROTOCOL full-validation per-class AP report")
    parser.add_argument("--new-ap-report", help="New native full-validation per-class AP report")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
