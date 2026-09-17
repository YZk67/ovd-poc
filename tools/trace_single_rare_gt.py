#!/usr/bin/env python3
"""Trace one rare GT across saved stage predictions; CPU only, no raw-query claims.

Uses the completed full-IoU stage comparison's input hashes and official LVIS
matching on the GT image. No model, training, image decoding or GPU inference.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.compare_rare_pr_reports import file_identity, load_json, save_json

IOUS = tuple(f"{v / 100:.2f}" for v in range(50, 100, 5))


def scan_image(path, image_id, chunk_size=1024*1024):
    """Stream an ordinary JSON array; retain only this image, original order.

    Do not load a ~1GB prediction list into Python objects simultaneously. Both
    compact and pretty JSON are supported, and truncated/trailing data fails.
    """
    decoder = json.JSONDecoder()
    selected, total = [], 0
    with open(path, encoding="utf-8") as handle:
        buffer, pos, eof = "", 0, False

        def refill():
            nonlocal buffer, pos, eof
            more = handle.read(chunk_size)
            buffer, pos = buffer[pos:] + more, 0
            eof = not more

        def peek():
            nonlocal pos
            while True:
                while pos < len(buffer) and buffer[pos] in " \r\n\t":
                    pos += 1
                if pos < len(buffer):
                    return buffer[pos]
                if eof:
                    return ""
                refill()

        if peek() != "[":
            raise ValueError("Predictions must be a JSON array")
        pos += 1
        if peek() != "]":
            while True:
                if peek() != "{":
                    raise ValueError("Expected a prediction object; truncated/invalid JSON")
                while True:
                    try:
                        value, pos = decoder.raw_decode(buffer, pos)
                        break
                    except json.JSONDecodeError as error:
                        if eof or len(buffer)-pos > 16*1024*1024:
                            raise ValueError("Invalid/truncated or oversized prediction object") from error
                        refill()
                total += 1
                if value.get("image_id") == image_id:
                    selected.append(value)
                if total % 1000000 == 0:
                    print(f"[scan] {Path(path).name}: {total} records, retained={len(selected)}", flush=True)
                separator = peek()
                if separator == "]":
                    break
                if separator != ",":
                    raise ValueError("Missing prediction separator or closing array")
                pos += 1
        pos += 1  # closing ]
        if peek():
            raise ValueError("Trailing data after prediction array")
    return selected, total


def bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter = max(0., min(ax+aw, bx+bw)-max(ax, bx)) * max(0., min(ay+ah, by+bh)-max(ay, by))
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.


def select_target(dataset, report, name):
    categories = [c for c in dataset["categories"] if c["name"] == name]
    rows = [r for r in report["per_class"] if r["name"] == name]
    if len(categories) != 1 or categories[0].get("frequency") != "r" or len(rows) != 1:
        raise ValueError("Need one matching rare class in annotations and stage report")
    category, expected = categories[0], rows[0]
    if expected["category_id"] != category["id"] or expected["gt_annotations"] != 1:
        raise ValueError("This bounded trace requires exactly ONE validation GT")
    annotations = [a for a in dataset["annotations"] if a["category_id"] == category["id"]]
    if len(annotations) != 1:
        raise ValueError("Annotation GT count differs from the single-GT stage report")
    gt = annotations[0]
    images = [i for i in dataset["images"] if i["id"] == gt["image_id"]]
    if len(images) != 1 or set(expected["iou"]) != set(IOUS):
        raise ValueError("Missing unique GT image or ten-IoU stage evidence")
    return gt, images[0], expected


def validate_predictions(predictions, image_id, categories):
    for p in predictions:
        bbox = p.get("bbox", [])
        if (p.get("image_id") != image_id or p.get("category_id") not in categories
                or len(bbox) != 4 or any(not math.isfinite(v) for v in bbox)
                or min(bbox[2:]) < 0 or not math.isfinite(p.get("score", math.nan))):
            raise ValueError("Invalid selected-image prediction")


def describe_endpoint(gt, categories, raw_predictions, panel, expected, arm):
    """Evidence labels refer ONLY to stored detections, never all model queries."""
    selected = panel["selected_predictions"]
    scores = [p["score"] for p in selected]
    records = []
    for p in selected:
        records.append({**p, "category_name": categories[p["category_id"]]["name"],
                        "iou_to_gt": bbox_iou(p["bbox"], gt["bbox"]),
                        "score_rank_interval": [1+sum(s > p["score"] for s in scores),
                                                sum(s >= p["score"] for s in scores)]})
    records.sort(key=lambda p: (-p["iou_to_gt"], -p["score"], p["detection_id"]))
    same = [p for p in records if p["category_id"] == gt["category_id"]]
    by_iou = {}
    for key in IOUS:
        matches = [r for r in panel["gt_by_iou"][key] if r["gt_id"] == gt["id"]]
        if len(matches) != 1 or matches[0]["gt_ignore"]:
            raise ValueError("Unique target GT missing or officially ignored")
        official = matches[0]
        saved = expected["iou"][key][arm]
        # Whole-validation TP equals this image's TP because there is only ONE
        # GT globally. Whole-validation FP/ranks do NOT equal this image's FP/ranks.
        if saved["num_gt"] != 1 or int(official["matched"]) != saved["tp"]:
            raise ValueError(f"{arm} IoU={key} target TP differs from completed stage report")
        threshold = float(key)
        eligible = [p for p in records if p["iou_to_gt"] >= threshold]
        correct = [p for p in eligible if p["category_id"] == gt["category_id"]]
        if {p["detection_id"] for p in correct} != {p["detection_id"] for p in official["selected_candidates"]}:
            raise ValueError("Geometry/official eligible candidate IDs disagree")
        in_file = [p for p in raw_predictions if p["category_id"] == gt["category_id"]
                   and bbox_iou(p["bbox"], gt["bbox"]) >= threshold]
        if official["matched"]:
            reason = "official_true_positive"
        elif correct:
            reason = ("eligible_correct_detection_ignored" if all(p["ignored"] for p in official["selected_candidates"])
                      else "eligible_correct_detection_unmatched_check_assignments")
        elif in_file:
            reason = "eligible_correct_pair_in_file_but_removed_by_lvis_cap"
        elif eligible:
            reason = "eligible_geometry_survives_under_other_categories"
        else:
            reason = "no_eligible_geometry_in_saved_top300"
        by_iou[key] = {"reason": reason, "official": official,
                       "selected_any_class_eligible": len(eligible),
                       "selected_true_class_eligible": len(correct),
                       "file_true_class_eligible": len(in_file),
                       "eligible_other_categories": sorted({p["category_name"] for p in eligible if p not in correct}),
                       "raw_query_coverage": None, "true_class_rank_before_top300": None}
    return {"selected_predictions_count": len(selected), "image_predictions_in_file": len(raw_predictions),
            "all_class_cutoff_score": min(scores) if len(scores) == panel["max_dets"] else None,
            "best_any_class_iou": records[0]["iou_to_gt"] if records else None,
            "best_true_class_iou": same[0]["iou_to_gt"] if same else None,
            "true_class_detections": same, "selected_detections_by_gt_iou": records,
            "by_iou": by_iou}


def verify_file(identity, label):
    actual = file_identity(identity["path"])
    if actual != identity:
        raise ValueError(f"{label} changed since stage comparison; refusing mixed inputs")
    return actual


def run(args):
    stage = Path(args.stage_dir).resolve()
    paths = {name: stage/(name+".json") for name in ("report", "inputs", "COMPLETE")}
    receipts = {name: file_identity(p) for name, p in paths.items()}
    report, manifest, complete = (load_json(paths[k]) for k in ("report", "inputs", "COMPLETE"))
    if (not report.get("complete") or not complete.get("source_files_unchanged")
            or not manifest.get("full_iou") or not complete.get("full_iou")
            or report.get("gpu_inference") is not False or report.get("training_updates") != 0):
        raise ValueError("Need completed full-IoU CPU stage comparison")
    for side, arm in (("old", "A"), ("new", "B")):
        if (report["endpoint_labels"][arm] != manifest["stage_labels"][side]
                or abs(report[arm+"_apr"] - manifest["expected_apr"][side]) > .0002):
            raise ValueError("Stage labels/APr disagree with input manifest")
    sources = {k: manifest["inputs"][k] for k in ("old_predictions", "new_predictions", "annotations")}
    protected = [*paths.values(), *[Path(v["path"]).resolve() for v in sources.values()],
                 *[Path(v["path"]).resolve() for v in report["source_reports"].values()]]
    output = Path(args.output).resolve()
    if output.exists() or output in protected or stage == output.parent or stage in output.parents:
        raise ValueError("Use a new output outside the source stage directory; do not overwrite inputs")
    print("[verify] stage reports and prediction/annotation hashes", flush=True)
    for k, identity in {**sources, **report["source_reports"]}.items():
        verify_file(identity, k)
    dataset = load_json(sources["annotations"]["path"])
    gt, image, expected = select_target(dataset, report, args.category)
    categories = {c["id"]: c for c in dataset["categories"]}
    print(f"[target] {args.category} GT={gt['id']} image={gt['image_id']} bbox={gt['bbox']}", flush=True)
    # Keep every annotation and all category metadata for the one selected image.
    dataset = {**dataset, "images": [image],
               "annotations": [a for a in dataset["annotations"] if a["image_id"] == image["id"]]}
    from tools.lvis_gt_matching import match_panel_gt

    result = {"complete": False, "category": args.category, "gt": gt, "image": image,
              "endpoint_labels": report["endpoint_labels"], "stage_APr_delta": report["delta_apr"],
              "class_AP": {"old": expected["old_AP"], "new": expected["new_AP"],
                           "apr_contribution": expected["apr_contribution"]},
              "sources": {**receipts, **sources, "pr_reports": report["source_reports"]},
              "endpoints": {}, "gpu_inference": False, "training_updates": 0,
              "scope": ["Post-hoc selected single GT, not a new AP estimate or general training mechanism.",
                        "Official LVIS matching; all-category image cap before rare filtering.",
                        "No eligible saved box does NOT prove no eligible raw query.",
                        "Another selected category at a good box does NOT reveal the true-class pre-top300 rank.",
                        "Detection IDs are endpoint-local, not cross-model query identities.",
                        "Input hashes inherit stage provenance; no independent checkpoint/config authentication."]}
    for side, arm in (("old", "A"), ("new", "B")):
        print(f"[scan {side}] retain only image={image['id']}; no full prediction-list allocation", flush=True)
        predictions, count = scan_image(sources[side+"_predictions"]["path"], image["id"])
        validate_predictions(predictions, image["id"], categories)
        panel = match_panel_gt(dataset, [image["id"]], predictions, iou_thresholds=[float(x) for x in IOUS],
                               max_dets=300, include_selected_predictions=True)
        result["endpoints"][side] = describe_endpoint(gt, categories, predictions, panel, expected, arm)
        result["endpoints"][side]["total_file_records_scanned"] = count
    old = result["endpoints"]["old"]
    old_tp_id = old["by_iou"]["0.50"]["official"]["matched_detection_id"]
    old_tp = next((p for p in old["selected_detections_by_gt_iou"] if p["detection_id"] == old_tp_id), None)
    result["new_boxes_nearest_old_tp"] = (sorted(
        [{**p, "iou_to_old_tp": bbox_iou(p["bbox"], old_tp["bbox"])}
         for p in result["endpoints"]["new"]["selected_detections_by_gt_iou"]],
        key=lambda p: (-p["iou_to_old_tp"], -p["score"], p["detection_id"]))[:10] if old_tp else [])
    for name, identity in {**receipts, **sources, **report["source_reports"]}.items():
        verify_file(identity, name)
    result["complete"] = True
    save_json(output, result)
    print("\n=== Unique rare GT trace: official matching, saved candidates only ===")
    for side, data in result["endpoints"].items():
        print(side, "best selected IoU any/true class:", data["best_any_class_iou"], data["best_true_class_iou"])
        for key, row in data["by_iou"].items():
            print(f"  IoU={key} {row['reason']} any/correct={row['selected_any_class_eligible']}/{row['selected_true_class_eligible']}")
        for p in data["selected_detections_by_gt_iou"][:10]:
            print(f"  {p['category_name']} IoU={p['iou_to_gt']:.4f} score={p['score']:.6f} bbox={p['bbox']}")
    print("Raw-query coverage and true-class rank before top-300 remain unknown. No training cause inferred.")
    print(f"[save] {output}", flush=True)
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage-dir", required=True, help="Completed run_rare_stage_comparison.py --full-iou directory")
    p.add_argument("--category", default="scarecrow", help="Must have exactly one validation GT")
    p.add_argument("--output", required=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
