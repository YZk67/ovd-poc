"""Attach official LVIS matches and selected geometric candidates to rare GT.

This CPU-only helper reads saved all-class predictions. It does not rerun a
model or modify the existing pairing diagnostic and its cache fingerprints.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

import numpy as np

from tools.pairing_lvis_support import (
    _lvis_from_dataset,
    _ranked_pr_from_eval_imgs,
    _summary,
)


def _gt_rows_from_evaluator(evaluator, iou_index):
    """Extract one IoU's rows, mapping cached IoU columns by original GT ID."""
    threshold = min(float(evaluator.params.iou_thrs[iou_index]), 1 - 1e-10)
    rows = []
    for item in evaluator.eval_imgs:
        if item is None:
            continue
        image_id, category_id = item["image_id"], item["category_id"]
        gt_ids, dt_ids = list(item["gt_ids"]), list(item["dt_ids"])
        context = f"image={image_id} category={category_id} IoU={threshold:g}"
        gt_positions = {int(gt_id): index for index, gt_id in enumerate(gt_ids)}
        dt_positions = {int(dt_id): index for index, dt_id in enumerate(dt_ids)}
        if len(gt_positions) != len(gt_ids) or len(dt_positions) != len(dt_ids):
            raise ValueError(f"Duplicate official matching IDs: {context}")
        if any(identifier <= 0 for identifier in (*gt_positions, *dt_positions)):
            raise ValueError(f"Official matching IDs must be positive: {context}")
        gt_matches = np.asarray(item["gt_matches"])[iou_index]
        dt_matches = np.asarray(item["dt_matches"])[iou_index]
        gt_ignore = np.asarray(item["gt_ignore"], dtype=bool)
        dt_ignore = np.asarray(item["dt_ignore"], dtype=bool)[iou_index]
        scores = np.asarray(item["dt_scores"], dtype=np.float64)
        if not (
            gt_matches.shape == gt_ignore.shape == (len(gt_ids),)
            and dt_matches.shape == dt_ignore.shape == scores.shape == (len(dt_ids),)
        ):
            raise ValueError(f"Official matching array shapes disagree: {context}")

        # compute_iou sorts DT by score but retains the original GT order.
        # evaluate_img subsequently sorts GT ignore-last. Never interpret the
        # cached matrix's columns as item['gt_ids'] without this ID remapping.
        raw_gt, raw_dt = evaluator._get_gt_dt(image_id, category_id)
        raw_gt_positions = {int(ann["id"]): index for index, ann in enumerate(raw_gt)}
        raw_dt_order = np.argsort([-ann["score"] for ann in raw_dt], kind="mergesort")
        raw_dt_positions = {
            int(raw_dt[index]["id"]): position
            for position, index in enumerate(raw_dt_order)
        }
        if set(raw_gt_positions) != set(gt_positions) or set(raw_dt_positions) != set(dt_positions):
            raise ValueError(f"Cached IoU IDs disagree with matching IDs: {context}")
        if gt_ids and dt_ids:
            raw_ious = np.asarray(evaluator.ious[image_id, category_id], dtype=np.float64)
            if raw_ious.shape != (len(raw_dt), len(raw_gt)):
                raise ValueError(f"Cached IoU matrix has unexpected shape: {context}")
            ious = raw_ious[np.ix_(
                [raw_dt_positions[int(dt_id)] for dt_id in dt_ids],
                [raw_gt_positions[int(gt_id)] for gt_id in gt_ids],
            )]
        else:
            # Official mask IoU may return [] for either empty dimension.
            ious = np.empty((len(dt_ids), len(gt_ids)), dtype=np.float64)

        for dt_index, matched_gt in enumerate(dt_matches):
            if not matched_gt:
                continue
            gt_index = gt_positions.get(int(matched_gt))
            if gt_index is None or int(gt_matches[gt_index]) != int(dt_ids[dt_index]):
                raise ValueError(f"Nonreciprocal detection-to-GT match: {context}")
            if dt_ignore[dt_index] != gt_ignore[gt_index]:
                raise ValueError(f"Matched GT/detection ignore flags disagree: {context}")

        for gt_index, gt_id in enumerate(gt_ids):
            eligible_indices = np.flatnonzero(ious[:, gt_index] >= threshold)
            candidates = [
                {
                    "detection_id": int(dt_ids[index]),
                    "score": float(scores[index]),
                    "iou": float(ious[index, gt_index]),
                    "matched_gt_id": int(dt_matches[index]) if dt_matches[index] else None,
                    "ignored": bool(dt_ignore[index]),
                }
                for index in eligible_indices
            ]
            matched_id = int(gt_matches[gt_index]) if gt_matches[gt_index] else None
            matched_score = None
            if matched_id is not None:
                dt_index = dt_positions.get(matched_id)
                if dt_index is None or int(dt_matches[dt_index]) != int(gt_id):
                    raise ValueError(f"Nonreciprocal GT-to-detection match: {context} GT={gt_id}")
                if dt_index not in eligible_indices:
                    raise ValueError(f"Official match is not IoU eligible: {context} GT={gt_id}")
                matched_score = float(scores[dt_index])
            rows.append({
                "gt_id": int(gt_id),
                "image_id": int(image_id),
                "category_id": int(category_id),
                "gt_ignore": bool(gt_ignore[gt_index]),
                "matched": matched_id is not None,
                "matched_detection_id": matched_id,
                "matched_detection_score": matched_score,
                "selected_iou_eligible_count": len(candidates),
                "selected_candidates": candidates,
            })
    rows.sort(key=lambda row: (row["image_id"], row["category_id"], row["gt_id"]))
    if len({row["gt_id"] for row in rows}) != len(rows):
        raise ValueError("Global GT IDs are repeated in official evaluation rows")
    return rows


def match_panel_gt(
    dataset, image_ids, predictions, *, iou_thresholds=(0.5, 0.75), max_dets=300,
    include_selected_predictions=False,
):
    """Return official rare-GT matches and diagnostic panel TP/FP summaries.

    Input predictions are ordinary LVIS bbox results for ALL categories. The
    per-image max_dets cap precedes rare filtering, just as in LVISResults.
    Unverified categories and ignored GT retain official LVIS behavior.

    ``matched`` records the GT's official assignment even when gt_ignore=True;
    consumers must exclude ignored GT from recall and transition denominators.
    Candidate detections have the correct image/category and meet LVIS's IoU
    threshold, but can be assigned to another GT or be ignored. Detection IDs
    are local to this evaluation; GT IDs remain the original global IDs.

    ``include_selected_predictions`` exposes the actual all-category detections
    AFTER LVISResults' cap, with matching-local IDs. It does not restore any
    candidates missing from the supplied prediction JSON.
    """
    if not isinstance(max_dets, int) or isinstance(max_dets, bool) or max_dets <= 0:
        raise ValueError("max_dets must be a positive integer")
    thresholds = sorted({float(value) for value in iou_thresholds})
    if not thresholds or any(not np.isfinite(value) or not 0 < value <= 1 for value in thresholds):
        raise ValueError("iou_thresholds must contain finite values in (0, 1]")
    selected_ids = sorted(set(image_ids))
    selected_set = set(selected_ids)
    if not selected_set.issubset({image["id"] for image in dataset["images"]}):
        raise ValueError("image_ids include IDs absent from the LVIS dataset")
    rare_ids = sorted(
        category["id"] for category in dataset["categories"]
        if category.get("frequency") == "r"
    )
    subset = deepcopy({
        **dataset,
        "images": [image for image in dataset["images"] if image["id"] in selected_set],
        "annotations": [
            annotation for annotation in dataset["annotations"]
            if annotation["image_id"] in selected_set
        ],
    })
    selected_predictions = deepcopy([
        prediction for prediction in predictions if prediction["image_id"] in selected_set
    ])

    from lvis import LVIS, LVISEval, LVISResults

    ground_truth = _lvis_from_dataset(LVIS, subset)
    detections = (
        LVISResults(ground_truth, selected_predictions, max_dets=max_dets)
        if selected_predictions else
        _lvis_from_dataset(LVISResults, {**deepcopy(subset), "annotations": []})
    )
    evaluator = LVISEval(ground_truth, detections, "bbox")
    evaluator.params.img_ids = selected_ids
    evaluator.params.cat_ids = rare_ids
    evaluator.params.iou_thrs = np.asarray(thresholds, dtype=np.float64)
    all_area = evaluator.params.area_rng[evaluator.params.area_rng_lbl.index("all")]
    evaluator.params.area_rng = [all_area]
    evaluator.params.area_rng_lbl = ["all"]
    evaluator.params.max_dets = max_dets
    evaluator.evaluate()

    capped_counts = Counter(ann["category_id"] for ann in detections.dataset["annotations"])
    evaluated_counts = Counter()
    for item in evaluator.eval_imgs:
        if item is not None:
            evaluated_counts[item["category_id"]] += len(item["dt_ids"])
    category_names = {
        category["id"]: category.get("name", str(category["id"]))
        for category in dataset["categories"]
    }
    per_class = [
        {"category_id": category_id, "name": category_names[category_id], "iou_curves": {}}
        for category_id in rare_ids
    ]
    gt_by_iou, summary_by_iou = {}, {}
    for iou_index, threshold in enumerate(thresholds):
        key = f"{threshold:.2f}" if threshold == round(threshold, 2) else f"{threshold:.12g}"
        rows = _gt_rows_from_evaluator(evaluator, iou_index)
        curves = [
            _ranked_pr_from_eval_imgs(
                evaluator, cat_index, iou_index,
                capped_counts[category_id] - evaluated_counts[category_id],
            )
            for cat_index, category_id in enumerate(rare_ids)
        ]
        for category, curve in zip(per_class, curves):
            category["num_gt"] = curve["num_gt"]
            category["iou_curves"][key] = curve
        summary = _summary(curves)
        valid_rows = [row for row in rows if not row["gt_ignore"]]
        true_positives = sum(row["matched"] for row in valid_rows)
        if summary["num_gt"] != len(valid_rows) or summary["true_positives"] != true_positives:
            raise ValueError(f"Official GT/detection TP counts disagree at IoU={key}")
        if summary["valid_detections"] != summary["true_positives"] + summary["false_positives"]:
            raise ValueError(f"Official valid-detection counts disagree at IoU={key}")
        summary.update({
            "num_gt_including_ignored": len(rows),
            "ignored_gt": len(rows) - len(valid_rows),
            "false_negatives": len(valid_rows) - true_positives,
            "unmatched_gt_with_selected_iou_eligible_detection": sum(
                not row["matched"] and row["selected_iou_eligible_count"] > 0
                for row in valid_rows
            ),
            "unmatched_gt_without_selected_iou_eligible_detection": sum(
                not row["matched"] and row["selected_iou_eligible_count"] == 0
                for row in valid_rows
            ),
        })
        gt_by_iou[key] = rows
        summary_by_iou[key] = summary
    result = {
        "scope": "official_LVIS_matching_on_selected_panel_not_full_validation_APr",
        "image_ids": selected_ids,
        "rare_category_ids": rare_ids,
        "iou_thresholds": thresholds,
        "max_dets": max_dets,
        "detection_id_scope": "LVISResults IDs local to this evaluation, not cross-checkpoint query IDs",
        "summary_by_iou": summary_by_iou,
        "gt_by_iou": gt_by_iou,
        "per_class": per_class,
    }
    if include_selected_predictions:
        result["selected_predictions"] = [
            {"detection_id": int(ann["id"]), "image_id": int(ann["image_id"]),
             "category_id": int(ann["category_id"]), "score": float(ann["score"]),
             "bbox": [float(v) for v in ann["bbox"]]}
            for ann in detections.dataset["annotations"]
        ]
    return result
