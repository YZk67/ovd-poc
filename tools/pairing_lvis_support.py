"""Deterministic rare-image panels and diagnostic PR using official LVIS matches.

This module does not import model code. LVIS is loaded only for evaluation.
Panel precision/recall are diagnostics, not official full-validation APr.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import logging
import random

import numpy as np


def select_panel(dataset, negative_images=128, seed=42):
    """Keep every rare-GT image and sample verified-negative control images.

    Sampling depends only on annotations and the seed, never predictions. A
    control has no rare annotation and at least one verified rare-class absence.
    The latter does not imply that *all* rare classes are verified absent.
    """
    if not isinstance(negative_images, int) or isinstance(negative_images, bool):
        raise ValueError("negative_images must be a nonnegative integer")
    if negative_images < 0:
        raise ValueError("negative_images must be a nonnegative integer")
    rare_ids = {
        category["id"] for category in dataset["categories"]
        if category.get("frequency") == "r"
    }
    images = {image["id"]: image for image in dataset["images"]}
    rare_annotations = [
        annotation for annotation in dataset["annotations"]
        if annotation["category_id"] in rare_ids
    ]
    rare_images = {annotation["image_id"] for annotation in rare_annotations}
    if not rare_images.issubset(images):
        raise ValueError("rare annotations reference image IDs absent from images")
    candidates = sorted(
        image_id for image_id, image in images.items()
        if image_id not in rare_images
        and rare_ids.intersection(image.get("neg_category_ids", []))
    )
    negatives = sorted(random.Random(seed).sample(
        candidates, min(negative_images, len(candidates))
    ))
    covered_negatives = set()
    for image_id in negatives:
        covered_negatives.update(
            rare_ids.intersection(images[image_id].get("neg_category_ids", []))
        )
    return {
        "image_ids": sorted(rare_images.union(negatives)),
        "rare_image_ids": sorted(rare_images),
        "negative_image_ids": negatives,
        "rare_category_ids": sorted(rare_ids),
        "seed": seed,
        "requested_negative_images": negative_images,
        "control_policy": (
            "seeded_uniform_without_replacement_from_sorted_image_ids; "
            "no_rare_GT_and_at_least_one_verified_rare_negative; "
            "selection_independent_of_predictions"
        ),
        "counts": {
            "images": len(rare_images) + len(negatives),
            "rare_gt_images": len(rare_images),
            "negative_control_images": len(negatives),
            "available_negative_control_images": len(candidates),
            "rare_categories": len(rare_ids),
            "rare_categories_with_gt": len({
                annotation["category_id"] for annotation in rare_annotations
            }),
            "rare_annotations": len(rare_annotations),
            "verified_negative_rare_categories_in_controls": len(covered_negatives),
        },
    }


def _ranked_pr_from_eval_imgs(evaluator, cat_index, iou_index, ignored_before_matching=0):
    """Read official matches, including categories with negatives but no GT."""
    params = evaluator.params
    num_images = len(params.img_ids)
    area_index = params.area_rng_lbl.index("all")
    offset = (cat_index * len(params.area_rng) + area_index) * num_images
    image_evals = [
        item for item in evaluator.eval_imgs[offset:offset + num_images]
        if item is not None
    ]
    num_gt = sum(
        int(np.count_nonzero(np.asarray(item["gt_ignore"]) == 0))
        for item in image_evals
    )
    if image_evals:
        scores = np.concatenate([
            np.asarray(item["dt_scores"], dtype=np.float64)
            for item in image_evals
        ])
        matches = np.concatenate([
            np.asarray(item["dt_matches"])[iou_index] for item in image_evals
        ])
        ignored = np.concatenate([
            np.asarray(item["dt_ignore"], dtype=bool)[iou_index]
            for item in image_evals
        ])
    else:
        scores = np.empty(0, dtype=np.float64)
        matches = np.empty(0, dtype=np.float64)
        ignored = np.empty(0, dtype=bool)
    if not (len(scores) == len(matches) == len(ignored)):
        raise ValueError("LVIS score/match/ignore arrays have different lengths")
    order = np.argsort(-scores, kind="mergesort")
    order = order[~ignored[order]]
    cumulative_tp = np.cumsum(matches[order] > 0, dtype=np.int64)
    cumulative_fp = np.cumsum(matches[order] == 0, dtype=np.int64)
    curve = [
        {
            "score": float(scores[index]),
            "tp": int(tp),
            "fp": int(fp),
            "precision": float(tp / (tp + fp)),
            "recall": float(tp / num_gt) if num_gt else None,
        }
        for index, tp, fp in zip(order, cumulative_tp, cumulative_fp)
    ]
    true_positives = int(cumulative_tp[-1]) if len(order) else 0
    false_positives = int(cumulative_fp[-1]) if len(order) else 0
    ignored_matching = int(ignored.sum())
    return {
        "num_gt": num_gt,
        "valid_detections": len(order),
        "ignored_detections": ignored_matching + int(ignored_before_matching),
        "ignored_matching_detections": ignored_matching,
        "ignored_before_matching_detections": int(ignored_before_matching),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "precision": curve[-1]["precision"] if curve else None,
        "recall": true_positives / num_gt if num_gt else None,
        "curve": curve,
    }


def _summary(curves):
    precisions = [row["precision"] for row in curves if row["precision"] is not None]
    recalls = [row["recall"] for row in curves if row["recall"] is not None]
    totals = {
        key: sum(row[key] for row in curves)
        for key in (
            "num_gt", "valid_detections", "ignored_detections",
            "ignored_matching_detections", "ignored_before_matching_detections",
            "true_positives", "false_positives",
        )
    }
    return {
        **totals,
        "micro_precision": (
            totals["true_positives"] / totals["valid_detections"]
            if totals["valid_detections"] else None
        ),
        "micro_recall": (
            totals["true_positives"] / totals["num_gt"] if totals["num_gt"] else None
        ),
        "macro_precision": float(np.mean(precisions)) if precisions else None,
        "macro_recall": float(np.mean(recalls)) if recalls else None,
        "macro_precision_valid_categories": len(precisions),
        "macro_recall_valid_categories": len(recalls),
    }


def _lvis_from_dataset(cls, dataset):
    """Build the same LVIS index as its file constructor without temporary I/O."""
    result = cls.__new__(cls)
    result.logger = logging.getLogger("lvis")
    result.dataset = dataset
    result._create_index()
    return result


def evaluate_panel_predictions(
    dataset, image_ids, predictions, *, iou_thresholds=(0.5, 0.75), max_dets=300
):
    """Evaluate all-class LVIS bbox results on a selected image subset.

    LVISResults enforces the per-image cap *before* rare-category filtering.
    Only official evaluate() matching runs: no interpolated AP or APr is emitted.
    All GT fields are retained; ignore/crowd behavior is the installed LVIS API's
    behavior (upstream LVIS does not implement COCO's special crowd matching).

    Curves contain nonignored detections with cumulative TP/FP and raw precision
    and recall. Detections dropped before matching (normally unverified classes,
    also zero-area results) are counted separately and included in
    ignored_detections. Undefined precision
    (no evaluated predictions) or recall (no positive GT) is None; macro means
    include only categories with a defined respective denominator.
    """
    if not isinstance(max_dets, int) or isinstance(max_dets, bool) or max_dets <= 0:
        raise ValueError("max_dets must be a positive integer")
    thresholds = sorted({float(value) for value in iou_thresholds})
    if not thresholds or any(not np.isfinite(value) or not 0 < value <= 1 for value in thresholds):
        raise ValueError("iou_thresholds must contain finite values in (0, 1]")
    selected_ids = sorted(set(image_ids))
    selected_set = set(selected_ids)
    known_ids = {image["id"] for image in dataset["images"]}
    if not selected_set.issubset(known_ids):
        raise ValueError("image_ids include IDs absent from the LVIS dataset")
    rare_categories = sorted(
        (category for category in dataset["categories"] if category.get("frequency") == "r"),
        key=lambda category: category["id"],
    )
    rare_ids = [category["id"] for category in rare_categories]
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
    if selected_predictions:
        detections = LVISResults(ground_truth, selected_predictions, max_dets=max_dets)
    else:
        # LVISResults.__init__ indexes results[0], so initialize its empty index
        # directly. This preserves zero detections without introducing a dummy FP.
        detections = _lvis_from_dataset(LVISResults, {
            **deepcopy(subset), "annotations": []
        })
    evaluator = LVISEval(ground_truth, detections, "bbox")
    evaluator.params.img_ids = selected_ids
    evaluator.params.cat_ids = rare_ids
    evaluator.params.iou_thrs = np.asarray(thresholds, dtype=np.float64)
    all_area = evaluator.params.area_rng[evaluator.params.area_rng_lbl.index("all")]
    evaluator.params.area_rng = [all_area]
    evaluator.params.area_rng_lbl = ["all"]
    evaluator.params.max_dets = max_dets
    evaluator.evaluate()

    capped_counts = Counter(
        annotation["category_id"] for annotation in detections.dataset["annotations"]
    )
    matched_counts = Counter()
    for item in evaluator.eval_imgs:
        if item is not None:
            matched_counts[item["category_id"]] += len(item["dt_scores"])
    iou_keys = [
        f"{value:.2f}" if value == round(value, 2) else f"{value:.12g}"
        for value in thresholds
    ]
    per_class = []
    for cat_index, category in enumerate(rare_categories):
        category_id = category["id"]
        ignored_before_matching = capped_counts[category_id] - matched_counts[category_id]
        curves = {
            key: _ranked_pr_from_eval_imgs(
                evaluator, cat_index, iou_index, ignored_before_matching
            )
            for iou_index, key in enumerate(iou_keys)
        }
        per_class.append({
            "category_id": category_id,
            "name": category.get("name", str(category_id)),
            "num_gt": curves[iou_keys[0]]["num_gt"],
            "iou_curves": curves,
        })
    return {
        "scope": "selected_panel_diagnostic_not_official_APr",
        "image_ids": selected_ids,
        "rare_category_ids": rare_ids,
        "iou_thresholds": thresholds,
        "max_dets": max_dets,
        "detection_limit_policy": "per_image_across_all_categories_before_rare_filtering",
        "macro_policy": (
            "precision: categories with nonignored detections (including no-GT classes); "
            "recall: categories with nonignored GT; endpoint uses every retained detection"
        ),
        "counts": {
            "images": len(selected_ids),
            "rare_categories": len(rare_ids),
            "input_predictions_in_panel": len(selected_predictions),
            "capped_predictions_all_categories": sum(capped_counts.values()),
            "capped_rare_predictions": sum(capped_counts[category_id] for category_id in rare_ids),
        },
        "summary_by_iou": {
            key: _summary([row["iou_curves"][key] for row in per_class])
            for key in iou_keys
        },
        "per_class": per_class,
    }
