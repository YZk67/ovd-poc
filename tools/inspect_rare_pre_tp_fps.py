#!/usr/bin/env python3
"""Identify saved detections that precede rare-class true positives in LVIS PR.

This replays official bbox matching on the full validation set, including the
all-category per-image top-300 cap and LVIS federated ignore rules. Geometric
overlap labels are diagnostic evidence, not a claim of visual object identity.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Detectron2 also provides a top-level ``tools`` package.
sys.path.insert(0, str(ROOT))

from tools.compare_rare_pr_reports import file_identity, save_json
from tools.pairing_lvis_support import _lvis_from_dataset


def xywh_iou(left, right):
    ax, ay, aw, ah = map(float, left)
    bx, by, bw, bh = map(float, right)
    if min(aw, ah, bw, bh) <= 0:
        return 0.0
    width = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    height = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    overlap = width * height
    union = aw * ah + bw * bh - overlap
    return overlap / union if union > 0 else 0.0


def _best_overlap(detection, annotations, categories):
    if not annotations:
        return None
    annotation = max(annotations, key=lambda row: xywh_iou(detection["bbox"], row["bbox"]))
    return {
        "annotation_id": annotation["id"],
        "category_id": annotation["category_id"],
        "category_name": categories[annotation["category_id"]]["name"],
        "bbox": annotation["bbox"],
        "iou": xywh_iou(detection["bbox"], annotation["bbox"]),
        "ignored_gt": bool(annotation.get("ignore", 0)),
    }


def explain_false_positive(
    detection, *, image, annotations, categories, threshold, assigned_gt_ids
):
    """Describe annotated overlaps without asserting that an unlabeled region is background."""
    category_id = detection["category_id"]
    same = [row for row in annotations if row["category_id"] == category_id]
    other = [row for row in annotations if row["category_id"] != category_id]
    same_best = _best_overlap(detection, same, categories)
    other_best = _best_overlap(detection, other, categories)
    overlapping_assigned = [
        row["id"] for row in same
        if row["id"] in assigned_gt_ids
        and xywh_iou(detection["bbox"], row["bbox"]) >= threshold
    ]
    if overlapping_assigned:
        label = "overlaps_already_matched_same_class_gt"
    elif same_best is not None and same_best["iou"] >= threshold:
        label = "overlaps_unassigned_same_class_gt"
    elif other_best is not None and other_best["iou"] >= threshold:
        label = "overlaps_other_labeled_category"
    elif same_best is not None:
        label = "below_iou_for_same_class_gt"
    elif category_id in image.get("neg_category_ids", []):
        label = "no_same_class_gt_verified_negative"
    else:
        label = "no_same_class_gt_annotation"
    return {
        "overlap_label": label,
        "nearest_same_class_gt": same_best,
        "nearest_other_labeled_gt": other_best,
        "overlapping_already_matched_gt_ids": overlapping_assigned,
        "category_verified_negative": category_id in image.get("neg_category_ids", []),
        "category_not_exhaustive": category_id in image.get("not_exhaustive_category_ids", []),
    }


def ranked_detections(evaluator, category_index, iou_index):
    """Reconstruct the same stable, nonignored class-global stream as PR replay."""
    params = evaluator.params
    images = len(params.img_ids)
    area = params.area_rng_lbl.index("all")
    start = (category_index * len(params.area_rng) + area) * images
    items = [row for row in evaluator.eval_imgs[start:start + images] if row is not None]
    stream = []
    assigned = defaultdict(set)
    for row in items:
        matches = np.asarray(row["dt_matches"])[iou_index]
        ignored = np.asarray(row["dt_ignore"], dtype=bool)[iou_index]
        for detection_id, score, gt_id, skip in zip(
            row["dt_ids"], row["dt_scores"], matches, ignored
        ):
            if gt_id and not skip:
                assigned[row["image_id"]].add(int(gt_id))
            stream.append({
                "detection_id": int(detection_id),
                "score": float(score),
                "matched_gt_id": int(gt_id) if gt_id else None,
                "image_id": int(row["image_id"]),
                "ignored": bool(skip),
            })
    order = np.argsort([-row["score"] for row in stream], kind="mergesort")
    ordered = [stream[index] for index in order if not stream[index]["ignored"]]
    for rank, row in enumerate(ordered, 1):
        row["rank"] = rank
        del row["ignored"]
    return ordered, assigned


def verify_ranked_stream(ordered, expected, *, category, side, iou):
    if expected is None:
        raise ValueError(f"{side} {category} IoU={iou} has no saved comparison curve")
    true_positives = [row for row in ordered if row["matched_gt_id"] is not None]
    false_positives = len(ordered) - len(true_positives)
    if (len(ordered), len(true_positives), false_positives) != (
        expected["valid_detections"], expected["true_positives"], expected["false_positives"]
    ):
        raise ValueError(f"{side} {category} IoU={iou} TP/FP totals differ from comparison")
    events = expected["every_tp"]
    if len(events) != len(true_positives):
        raise ValueError(f"{side} {category} IoU={iou} TP event count differs")
    for index, (actual, event) in enumerate(zip(true_positives, events), 1):
        if (actual["rank"] != event["rank"] or
                actual["rank"] - index != event["fp_before"] or
                not math.isclose(actual["score"], event["score"], abs_tol=1e-6, rel_tol=0)):
            raise ValueError(f"{side} {category} IoU={iou} TP rank/score differs at #{index}")


def describe_stream(ordered, *, detections, images, annotations_by_image,
                    categories, threshold, assigned_gt_ids):
    tp_rows = [row for row in ordered if row["matched_gt_id"] is not None]
    first_rank = tp_rows[0]["rank"] if tp_rows else None
    last_rank = tp_rows[-1]["rank"] if tp_rows else None
    preceding = []
    for row in ordered[:last_rank] if last_rank is not None else []:
        if row["matched_gt_id"] is not None:
            continue
        detection = detections[row["detection_id"]]
        image = images[row["image_id"]]
        preceding.append({
            "rank": row["rank"],
            "detection_id": row["detection_id"],
            "image_id": row["image_id"],
            "image_file_name": image.get("file_name"),
            "image_coco_url": image.get("coco_url"),
            "score": row["score"],
            "bbox": detection["bbox"],
            **explain_false_positive(
                detection, image=image,
                annotations=annotations_by_image[row["image_id"]],
                categories=categories, threshold=threshold,
                assigned_gt_ids=assigned_gt_ids[row["image_id"]],
            ),
        })
    return {
        "true_positives": [
            {"rank": row["rank"], "detection_id": row["detection_id"],
             "image_id": row["image_id"], "matched_gt_id": row["matched_gt_id"],
             "score": row["score"], "bbox": detections[row["detection_id"]]["bbox"]}
            for row in tp_rows
        ],
        "first_tp_rank": first_rank,
        "last_tp_rank": last_rank,
        "false_positives_before_first_tp": (
            sum(row["rank"] < first_rank for row in preceding) if first_rank is not None else None
        ),
        "false_positives_before_last_tp": len(preceding) if last_rank is not None else None,
        "leading_false_positives": preceding,
        "overlap_labels": dict(Counter(row["overlap_label"] for row in preceding)),
    }


def _input_path(override, comparison, side, kind):
    if override:
        return Path(override)
    metadata = comparison.get("curve_replay", {}).get(side, {})
    source = metadata.get("sources", {}).get(kind, {})
    if not source.get("path"):
        raise ValueError(f"Missing {side} {kind} path; pass an explicit override")
    return Path(source["path"])


def run(args):
    comparison_path = Path(args.comparison)
    with comparison_path.open(encoding="utf-8") as stream:
        comparison = json.load(stream)
    if not comparison.get("complete"):
        raise ValueError("Comparison is incomplete; first fill missing full-validation curves")
    names = args.categories or [row["name"] for row in comparison["per_class"]]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate focus categories")
    expected = {row["name"]: row for row in comparison["per_class"]}
    if not set(names).issubset(expected):
        raise ValueError(f"Categories absent from comparison: {sorted(set(names) - set(expected))}")
    paths = {
        side: _input_path(getattr(args, side + "_predictions"), comparison, side, "predictions")
        for side in ("old", "new")
    }
    annotation_path = _input_path(args.annotations, comparison, "old", "annotations")
    if args.max_dets != comparison.get("scope", {}).get("max_dets"):
        raise ValueError("--max-dets disagrees with the completed ranking comparison")
    output = Path(args.output)
    if output.resolve() in {path.resolve() for path in
                            [comparison_path, annotation_path, *paths.values()]}:
        raise ValueError("Output must not overwrite comparison, predictions or annotations")
    thresholds = (0.5, 0.75)
    print(f"[hash] comparison and full-validation sources", flush=True)
    source_identities = {
        "comparison": file_identity(comparison_path),
        "annotations": file_identity(annotation_path),
    }
    for side, path in paths.items():
        source_identities[side + "_predictions"] = file_identity(path)
        metadata = comparison.get("curve_replay", {}).get(side, {}).get("sources", {})
        for kind, actual in (("predictions", source_identities[side + "_predictions"]),
                             ("annotations", source_identities["annotations"])):
            prior = metadata.get(kind, {}).get("sha256")
            if prior and actual["sha256"] != prior:
                raise ValueError(f"{side} {kind} changed since ranking comparison")
    print(f"[load] annotations={annotation_path} classes={names}", flush=True)
    from lvis import LVIS, LVISEval, LVISResults

    ground_truth = LVIS(str(annotation_path))
    images = ground_truth.imgs
    categories = ground_truth.cats
    annotations_by_image = ground_truth.img_ann_map
    focus_ids = []
    for name in names:
        row = expected[name]
        category_id = row["category_id"]
        if category_id not in categories or categories[category_id]["name"] != name:
            raise ValueError(f"Comparison category ID/name differs from annotations: {name}")
        focus_ids.append(category_id)
    report = {
        "scope": "official_LVIS_full_validation_ranked_FP_identity_diagnostic",
        "max_dets": args.max_dets,
        "iou_thresholds": list(thresholds),
        "sources": source_identities,
        "caveat": (
            "Overlap with another annotated category is not proof of visual "
            "misclassification. Unannotated objects may exist; TP ordinal is not "
            "a cross-checkpoint GT identity. Results include FPs preceding the last TP."
        ),
        "classes": {},
    }
    for side in ("old", "new"):
        print(f"[match {side}] loading all-class predictions={paths[side]}", flush=True)
        with paths[side].open(encoding="utf-8") as stream:
            predictions = json.load(stream)
        if not isinstance(predictions, list):
            raise ValueError(f"{side} predictions must be a JSON list")
        if predictions:
            results = LVISResults(ground_truth, predictions, max_dets=args.max_dets)
        else:
            results = _lvis_from_dataset(LVISResults, {
                **ground_truth.dataset, "annotations": [],
            })
        evaluator = LVISEval(ground_truth, results, "bbox")
        evaluator.params.cat_ids = sorted(focus_ids)
        evaluator.params.img_ids = sorted(images)
        evaluator.params.iou_thrs = np.asarray(thresholds)
        all_area = evaluator.params.area_rng[evaluator.params.area_rng_lbl.index("all")]
        evaluator.params.area_rng = [all_area]
        evaluator.params.area_rng_lbl = ["all"]
        evaluator.params.max_dets = args.max_dets
        print(f"[match {side}] images={len(images)} categories={len(focus_ids)}", flush=True)
        evaluator.evaluate()
        for cat_index, category_id in enumerate(evaluator.params.cat_ids):
            name = categories[category_id]["name"]
            for iou_index, threshold in enumerate(thresholds):
                key = f"{threshold:.2f}"
                ordered, assigned = ranked_detections(evaluator, cat_index, iou_index)
                target = expected[name]["curves"][key][side]
                verify_ranked_stream(ordered, target, category=name, side=side, iou=key)
                described = describe_stream(
                    ordered, detections=results.anns, images=images,
                    annotations_by_image=annotations_by_image, categories=categories,
                    threshold=threshold, assigned_gt_ids=assigned,
                )
                described["total_false_positives"] = target["false_positives"]
                report["classes"].setdefault(name, {}).setdefault(side, {})[key] = described
        print(f"[match {side}] comparison rank/score checks: PASS", flush=True)
        del predictions, results, evaluator
    save_json(output, report)
    print("\n=== FP before first/last TP (old -> new) ===", flush=True)
    for name in names:
        for key in ("0.50", "0.75"):
            old = report["classes"][name]["old"][key]
            new = report["classes"][name]["new"][key]
            print(
                f"{name:25} IoU={key} "
                f"first={old['false_positives_before_first_tp']}->{new['false_positives_before_first_tp']} "
                f"last={old['false_positives_before_last_tp']}->{new['false_positives_before_last_tp']} "
                f"new_labels={new['overlap_labels']}", flush=True,
            )
    for name in args.print_fp_classes:
        if name not in report["classes"]:
            continue
        rows = report["classes"][name]["new"]["0.50"]["leading_false_positives"]
        print(f"\n=== New {name}: FP preceding TP, IoU=0.50 ===", flush=True)
        for row in rows[:args.print_fp_limit]:
            other = row["nearest_other_labeled_gt"]
            other_name = other["category_name"] if other and other["iou"] >= .5 else "-"
            print(
                f"rank={row['rank']} image={row['image_id']} det_id={row['detection_id']} "
                f"score={row['score']:.6g} label={row['overlap_label']} "
                f"same_iou={row['nearest_same_class_gt']['iou'] if row['nearest_same_class_gt'] else None} "
                f"other={other_name} "
                f"other_iou={other['iou'] if other else None}", flush=True,
            )
        if len(rows) > args.print_fp_limit:
            print(f"... {len(rows) - args.print_fp_limit} further FPs in JSON", flush=True)
    print(f"[save] {output}", flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True,
                        help="Complete ranking_comparison.json from compare_rare_pr_reports.py")
    parser.add_argument("--old-predictions", help="Override source path recorded in comparison")
    parser.add_argument("--new-predictions", help="Override source path recorded in comparison")
    parser.add_argument("--annotations", help="Override annotation path recorded in comparison")
    parser.add_argument("--categories", nargs="+", help="Default: all comparison categories")
    parser.add_argument("--max-dets", type=int, default=300)
    parser.add_argument("--print-fp-classes", nargs="+", default=["koala", "roller_skate", "joystick"])
    parser.add_argument("--print-fp-limit", type=int, default=12)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.max_dets <= 0 or args.print_fp_limit < 0:
        parser.error("--max-dets must be positive and --print-fp-limit nonnegative")
    return args


if __name__ == "__main__":
    run(parse_args())
