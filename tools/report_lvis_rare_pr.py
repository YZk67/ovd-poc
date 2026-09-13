#!/usr/bin/env python
"""Report official LVIS rare per-class AP and selected classes' TP/FP PR curves.

Runs the LVIS bbox evaluator on *rare categories only* using an existing
prediction JSON. LVISResults still applies the official per-image top-300
limit before category filtering. This script never runs a model or GPU.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def valid_mean(values):
    values = np.asarray(values, dtype=np.float64)
    valid = values > -1
    return float(values[valid].mean()) if valid.any() else None


def per_class_ap(precision, recall, cat_index, params):
    """Read a category's official all-area precision tensor, in AP points."""
    area_index = params.area_rng_lbl.index("all")
    curves = precision[:, :, cat_index, area_index]
    iou50 = int(np.argmin(abs(params.iou_thrs - 0.50)))
    iou75 = int(np.argmin(abs(params.iou_thrs - 0.75)))
    if not np.isclose(params.iou_thrs[iou50], 0.50) or not np.isclose(
        params.iou_thrs[iou75], 0.75
    ):
        raise ValueError("LVIS evaluator lacks IoU 0.50 or 0.75")
    scale = lambda value: 100.0 * value if value is not None else None
    return {
        "AP": scale(valid_mean(curves)),
        "AP50": scale(valid_mean(curves[iou50])),
        "AP75": scale(valid_mean(curves[iou75])),
        "AR": scale(valid_mean(recall[:, cat_index, area_index])),
    }


def ranked_pr_from_eval_imgs(evaluator, cat_index, iou_index):
    """Reconstruct the official score-ordered TP/FP stream for one category.

    Uses eval_imgs' dt_matches and dt_ignore, so LVIS federated ignore rules
    are preserved. Ignored detections do not become false positives.
    """
    params = evaluator.params
    area_index = params.area_rng_lbl.index("all")
    num_images = len(params.img_ids)
    num_areas = len(params.area_rng)
    base = cat_index * num_areas * num_images + area_index * num_images
    image_evals = [
        result for result in evaluator.eval_imgs[base : base + num_images]
        if result is not None
    ]
    if not image_evals:
        return {
            "num_gt": 0,
            "valid_detections": 0,
            "ignored_detections": 0,
            "true_positives": 0,
            "false_positives": 0,
            "curve": [],
        }
    num_gt = sum(
        int(np.count_nonzero(np.asarray(item["gt_ignore"]) == 0))
        for item in image_evals
    )
    scores = np.concatenate(
        [np.asarray(item["dt_scores"], dtype=np.float64) for item in image_evals]
    )
    matches = np.concatenate(
        [np.asarray(item["dt_matches"])[iou_index] for item in image_evals]
    )
    ignored = np.concatenate(
        [np.asarray(item["dt_ignore"], dtype=bool)[iou_index] for item in image_evals]
    )
    if not (len(scores) == len(matches) == len(ignored)):
        raise ValueError("LVIS score/match/ignore arrays have different lengths")
    order = np.argsort(-scores, kind="mergesort")
    scores = scores[order]
    matches = matches[order]
    ignored = ignored[order]
    ignored_count = int(ignored.sum())
    valid = ~ignored
    scores = scores[valid]
    matched = matches[valid] > 0
    cumulative_tp = np.cumsum(matched, dtype=np.int64)
    cumulative_fp = np.cumsum(~matched, dtype=np.int64)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
    recall = cumulative_tp / num_gt if num_gt else np.zeros(len(scores))
    curve = [
        {
            "score": float(score),
            "tp": int(tp),
            "fp": int(fp),
            "precision": float(pre),
            "recall": float(rec),
        }
        for score, tp, fp, pre, rec in zip(
            scores, cumulative_tp, cumulative_fp, precision, recall
        )
    ]
    return {
        "num_gt": num_gt,
        "valid_detections": len(scores),
        "ignored_detections": ignored_count,
        "true_positives": int(cumulative_tp[-1]) if len(scores) else 0,
        "false_positives": int(cumulative_fp[-1]) if len(scores) else 0,
        "curve": curve,
    }


def operating_points(curve, score_thresholds=(0.05, 0.10, 0.30)):
    """Summarize the raw PR stream at explicit score thresholds."""
    points = {}
    for threshold in score_thresholds:
        eligible = [point for point in curve if point["score"] >= threshold]
        if eligible:
            last = eligible[-1]
            points[str(threshold)] = {
                key: last[key] for key in ("tp", "fp", "precision", "recall")
            }
        else:
            points[str(threshold)] = {
                "tp": 0, "fp": 0, "precision": None, "recall": 0.0
            }
    return points


def verify_apr(per_class_rows, official_apr, expected_apr, tolerance):
    valid = [row["AP"] for row in per_class_rows if row["AP"] is not None]
    if not valid:
        raise ValueError("no rare category has a valid AP")
    macro = float(np.mean(valid))
    if abs(macro - official_apr) > 1e-5:
        raise ValueError(
            f"per-class macro AP {macro:.4f} disagrees with LVIS APr "
            f"{official_apr:.4f}; evaluator indexing may be wrong"
        )
    if expected_apr is not None and abs(official_apr - expected_apr) > tolerance:
        raise ValueError(
            f"prediction JSON gives APr={official_apr:.4f}, not expected "
            f"{expected_apr:.4f}; check that it is from the intended checkpoint "
            "and inference protocol"
        )
    return macro


def focus_report(evaluator, category_id, cat_index, category_ap):
    params = evaluator.params
    area_index = params.area_rng_lbl.index("all")
    result = {"category_id": category_id, **category_ap, "iou_curves": {}}
    for iou in (0.50, 0.75):
        iou_index = int(np.argmin(abs(params.iou_thrs - iou)))
        if not np.isclose(params.iou_thrs[iou_index], iou):
            raise ValueError(f"LVIS evaluator lacks IoU {iou}")
        ranked = ranked_pr_from_eval_imgs(evaluator, cat_index, iou_index)
        ranked["score_operating_points"] = operating_points(ranked["curve"])
        official_curve = evaluator.eval["precision"][iou_index, :, cat_index, area_index]
        ranked["official_interpolated_pr"] = [
            {
                "recall": float(recall),
                "precision": float(value) if value >= 0 else None,
            }
            for recall, value in zip(params.rec_thrs, official_curve)
        ]
        result["iou_curves"][f"{iou:.2f}"] = ranked
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--annotations", default="dataset/lvis/lvis_v1_val.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-apr", type=float, required=True)
    parser.add_argument("--apr-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--focus", nargs="+", default=[
            "legume", "papaya", "crouton", "dragonfly", "garbage", "liquor",
            "cream_pitcher", "date_(fruit)", "egg_roll", "pin_(non_jewelry)",
        ]
    )
    args = parser.parse_args()
    prediction_path = Path(args.predictions)
    annotation_path = Path(args.annotations)
    if not prediction_path.is_file() or not annotation_path.is_file():
        raise FileNotFoundError("prediction JSON and LVIS annotation JSON must both exist")
    if args.apr_tolerance < 0:
        raise ValueError("--apr-tolerance must be nonnegative")

    from lvis import LVIS, LVISEval, LVISResults

    print(f"[load] annotations={annotation_path}", flush=True)
    lvis_gt = LVIS(str(annotation_path))
    categories = lvis_gt.load_cats(lvis_gt.get_cat_ids())
    rare = sorted(
        (category for category in categories if category["frequency"] == "r"),
        key=lambda category: category["id"],
    )
    if not rare:
        raise ValueError("LVIS annotations contain no rare categories")
    rare_ids = [category["id"] for category in rare]
    names = {category["name"]: category for category in rare}
    missing_focus = [name for name in args.focus if name not in names]
    if missing_focus:
        raise ValueError(f"focus classes are not rare LVIS categories: {missing_focus}")

    print(
        f"[load] predictions={prediction_path} bytes={prediction_path.stat().st_size} "
        f"rare_categories={len(rare_ids)} max_dets=300",
        flush=True,
    )
    lvis_dt = LVISResults(lvis_gt, str(prediction_path), max_dets=300)
    evaluator = LVISEval(lvis_gt, lvis_dt, "bbox")
    evaluator.params.cat_ids = rare_ids
    evaluator.run()
    official_apr = 100.0 * float(evaluator.get_results()["APr"])
    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    gt_counts = Counter(
        int(annotation["category_id"])
        for annotation in lvis_gt.dataset["annotations"]
    )
    per_class_rows = []
    for cat_index, category in enumerate(rare):
        per_class_rows.append(
            {
                "category_id": category["id"],
                "name": category["name"],
                "gt_annotations": gt_counts[category["id"]],
                **per_class_ap(precision, recall, cat_index, evaluator.params),
            }
        )
    macro_apr = verify_apr(
        per_class_rows, official_apr, args.expected_apr, args.apr_tolerance
    )
    valid_category_count = sum(row["AP"] is not None for row in per_class_rows)
    for row in per_class_rows:
        row["apr_deficit_vs_macro"] = (
            (macro_apr - row["AP"]) / valid_category_count
            if row["AP"] is not None else None
        )
    by_name = {row["name"]: (index, row) for index, row in enumerate(per_class_rows)}
    focus = {
        name: focus_report(evaluator, by_name[name][1]["category_id"], *by_name[name])
        for name in args.focus
    }

    print("\n=== Official LVIS rare per-class AP, same prediction JSON ===")
    print(f"APr={macro_apr:.4f} (expected {args.expected_apr:.4f}, tolerance {args.apr_tolerance})")
    print("class                 GT       AP     AP50     AP75   deficit   TP/FP@.1   P@.1%   TP/FP@.3   P@.3%")
    for name in args.focus:
        row = by_name[name][1]
        curve50 = focus[name]["iou_curves"]["0.50"]
        point10 = curve50["score_operating_points"]["0.1"]
        point30 = curve50["score_operating_points"]["0.3"]
        print(
            f"{name:20} {row['gt_annotations']:4d} "
            f"{row['AP'] if row['AP'] is not None else float('nan'):8.3f} "
            f"{row['AP50'] if row['AP50'] is not None else float('nan'):8.3f} "
            f"{row['AP75'] if row['AP75'] is not None else float('nan'):8.3f} "
            f"{row['apr_deficit_vs_macro']:+8.4f} "
            f"{point10['tp']:4d}/{point10['fp']:<5d} "
            f"{100 * point10['precision'] if point10['precision'] is not None else float('nan'):7.2f} "
            f"{point30['tp']:4d}/{point30['fp']:<5d} "
            f"{100 * point30['precision'] if point30['precision'] is not None else float('nan'):7.2f}"
        )
    lowest = sorted(
        (row for row in per_class_rows if row["AP"] is not None),
        key=lambda row: row["AP"],
    )[:15]
    print("\nLowest 15 rare-class APs (each class has equal macro weight):")
    for row in lowest:
        print(
            f"  {row['name']}: AP={row['AP']:.3f}, GT={row['gt_annotations']}, "
            f"APr deficit={row['apr_deficit_vs_macro']:+.4f}"
        )
    print(
        "TP/FP follow official LVIS ignore rules; raw ranked and 101-point "
        "interpolated PR curves are in the JSON. Counts alone are not AP.",
        flush=True,
    )
    report = {
        "predictions": str(prediction_path),
        "prediction_bytes": prediction_path.stat().st_size,
        "annotations": str(annotation_path),
        "max_dets": 300,
        "official_apr": official_apr,
        "expected_apr": args.expected_apr,
        "rare_category_count": len(rare),
        "per_class": per_class_rows,
        "focus": focus,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[save] {output}", flush=True)


if __name__ == "__main__":
    main()
