from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools.pairing_lvis_support import (
    _ranked_pr_from_eval_imgs,
    _summary,
    evaluate_panel_predictions,
    select_panel,
)


def image(image_id, negatives=(), not_exhaustive=()):
    return {
        "id": image_id,
        "height": 100,
        "width": 100,
        "neg_category_ids": list(negatives),
        "not_exhaustive_category_ids": list(not_exhaustive),
    }


def annotation(ann_id, image_id, category_id=1, **extra):
    return {
        "id": ann_id, "image_id": image_id, "category_id": category_id,
        "bbox": [0, 0, 10, 10], "area": 100, **extra,
    }


def prediction(image_id, category_id=1, score=0.9, bbox=None):
    return {
        "image_id": image_id, "category_id": category_id, "score": score,
        "bbox": [0, 0, 10, 10] if bbox is None else bbox,
    }


def dataset(images=None, annotations=None):
    return {
        "images": images if images is not None else [image(1)],
        "annotations": annotations if annotations is not None else [annotation(1, 1)],
        "categories": [
            {"id": 1, "name": "rare_one", "frequency": "r"},
            {"id": 2, "name": "rare_two", "frequency": "r"},
            {"id": 3, "name": "frequent", "frequency": "f"},
        ],
    }


def test_panel_keeps_all_rare_images_and_only_verified_negative_controls():
    source = dataset(
        images=[
            image(1, negatives=[2]), image(2), image(3, negatives=[1]),
            image(4, negatives=[3]), image(5), image(6, negatives=[2]),
        ],
        annotations=[annotation(1, 1), annotation(2, 2, 2), annotation(3, 3, 3)],
    )
    original = deepcopy(source)
    panel = select_panel(source, negative_images=100, seed=13)
    assert panel["image_ids"] == [1, 2, 3, 6]
    assert panel["rare_image_ids"] == [1, 2]  # Both singleton rare categories survive.
    assert panel["negative_image_ids"] == [3, 6]
    assert panel["rare_category_ids"] == [1, 2]
    assert panel["counts"]["available_negative_control_images"] == 2
    assert panel["counts"]["verified_negative_rare_categories_in_controls"] == 2
    assert panel["counts"]["rare_categories_with_gt"] == 2
    assert source == original


def test_panel_sampling_is_seeded_and_independent_of_annotation_order():
    source = dataset(
        images=[image(1)] + [image(index, negatives=[1]) for index in range(2, 32)]
    )
    shuffled = deepcopy(source)
    shuffled["images"].reverse()
    shuffled["categories"].reverse()
    expected = select_panel(source, negative_images=5, seed=42)
    assert select_panel(shuffled, negative_images=5, seed=42) == expected
    assert len(expected["negative_image_ids"]) == 5
    assert select_panel(source, negative_images=5, seed=43)["negative_image_ids"] != expected["negative_image_ids"]


def test_panel_zero_negatives_and_no_eligible_controls():
    source = dataset(images=[image(1), image(2, negatives=[1])])
    assert select_panel(source, negative_images=0)["image_ids"] == [1]
    assert select_panel(dataset())["negative_image_ids"] == []


def test_panel_retains_images_with_ignored_or_crowd_rare_annotations():
    source = dataset(
        images=[image(1), image(2)],
        annotations=[annotation(1, 1, ignore=1), annotation(2, 2, 2, iscrowd=1)],
    )
    assert select_panel(source, negative_images=0)["rare_image_ids"] == [1, 2]


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_panel_rejects_invalid_negative_counts(value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        select_panel(dataset(), negative_images=value)


def test_raw_curves_respect_official_ignore_and_stable_global_score_order():
    evaluator = SimpleNamespace(
        params=SimpleNamespace(img_ids=[1, 2], area_rng=[[0, 1e10]], area_rng_lbl=["all"]),
        eval_imgs=[
            {
                "dt_scores": [0.9, 0.8, 0.3],
                "dt_matches": np.array([[1, 2, 0], [0, 2, 0]]),
                "dt_ignore": np.array([[False, True, True], [False, True, True]]),
                "gt_ignore": np.array([False, True]),
            },
            {
                "dt_scores": [0.9, 0.2],
                "dt_matches": np.array([[0, 3], [0, 3]]),
                "dt_ignore": np.zeros((2, 2), dtype=bool),
                "gt_ignore": np.array([False]),
            },
        ],
    )
    result = _ranked_pr_from_eval_imgs(evaluator, 0, 0, ignored_before_matching=4)
    assert result["num_gt"] == 2
    assert result["true_positives"] == 2
    assert result["false_positives"] == 1
    assert result["ignored_matching_detections"] == 2
    assert result["ignored_before_matching_detections"] == 4
    assert result["ignored_detections"] == 6
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == 1
    assert result["curve"][0] == {
        "score": 0.9, "tp": 1, "fp": 0, "precision": 1.0, "recall": 0.5,
    }
    assert result["curve"][1]["fp"] == 1
    assert _ranked_pr_from_eval_imgs(evaluator, 0, 1)["true_positives"] == 1


def test_no_positive_class_retains_fp_counts_and_has_undefined_recall():
    evaluator = SimpleNamespace(
        params=SimpleNamespace(img_ids=[1], area_rng=[[0, 1e10]], area_rng_lbl=["all"]),
        eval_imgs=[{
            "dt_scores": [0.8, 0.4], "dt_matches": np.array([[0, 0]]),
            "dt_ignore": np.array([[False, False]]), "gt_ignore": np.array([]),
        }],
    )
    row = _ranked_pr_from_eval_imgs(evaluator, 0, 0)
    assert row["false_positives"] == 2
    assert row["precision"] == 0
    assert row["recall"] is None
    assert all(point["recall"] is None for point in row["curve"])
    summary = _summary([row])
    assert summary["macro_precision"] == 0
    assert summary["macro_precision_valid_categories"] == 1
    assert summary["macro_recall"] is None
    assert summary["macro_recall_valid_categories"] == 0


def test_empty_class_and_gt_without_predictions_have_distinct_recall_validity():
    evaluator = SimpleNamespace(
        params=SimpleNamespace(img_ids=[1], area_rng=[[0, 1e10]], area_rng_lbl=["all"]),
        eval_imgs=[None, {
            "dt_scores": [], "dt_matches": np.empty((1, 0)),
            "dt_ignore": np.empty((1, 0)), "gt_ignore": np.array([0]),
        }],
    )
    empty = _ranked_pr_from_eval_imgs(evaluator, 0, 0)
    missed = _ranked_pr_from_eval_imgs(evaluator, 1, 0)
    assert empty["precision"] is None and empty["recall"] is None
    assert missed["precision"] is None and missed["recall"] == 0
    assert _summary([empty, missed])["macro_recall_valid_categories"] == 1
    json.dumps(_summary([empty, missed]), allow_nan=False)


@pytest.mark.parametrize("thresholds", [(), (0,), (1.1,), (float("nan"),)])
def test_invalid_iou_thresholds_fail_before_loading_lvis(thresholds):
    with pytest.raises(ValueError, match="iou_thresholds"):
        evaluate_panel_predictions(dataset(), [1], [], iou_thresholds=thresholds)


def test_unknown_panel_image_fails_before_loading_lvis():
    with pytest.raises(ValueError, match="absent"):
        evaluate_panel_predictions(dataset(), [100], [])


def test_official_lvis_retains_negative_only_fp_and_ignores_unverified_classes():
    pytest.importorskip("lvis")
    source = dataset(images=[image(1, negatives=[2]), image(2, negatives=[1])])
    predictions = [
        prediction(1, 1, 0.9),  # Positive category, TP.
        prediction(1, 2, 0.8),  # Verified absent class with no GT anywhere: FP.
        prediction(2, 1, 0.7),  # Negative control: FP.
        prediction(2, 2, 0.6),  # Unverified class: neither TP nor FP.
    ]
    original_source, original_predictions = deepcopy(source), deepcopy(predictions)
    result = evaluate_panel_predictions(source, [1, 2], predictions)
    first, second = [row["iou_curves"]["0.50"] for row in result["per_class"]]
    assert (first["true_positives"], first["false_positives"]) == (1, 1)
    assert second["num_gt"] == 0
    assert second["false_positives"] == 1
    assert second["ignored_before_matching_detections"] == 1
    assert second["recall"] is None
    summary = result["summary_by_iou"]["0.50"]
    assert summary["false_positives"] == 2
    assert summary["macro_precision"] == pytest.approx(0.25)
    assert summary["macro_recall"] == 1
    assert source == original_source and predictions == original_predictions
    assert "APr" not in result and "AP" not in result
    json.dumps(result, allow_nan=False)


def test_official_lvis_applies_all_class_cap_before_rare_filtering():
    pytest.importorskip("lvis")
    predictions = [prediction(1, 3, 0.99), prediction(1, 1, 0.9)]
    result = evaluate_panel_predictions(dataset(), [1], predictions, max_dets=1)
    assert result["counts"]["capped_predictions_all_categories"] == 1
    assert result["counts"]["capped_rare_predictions"] == 0
    rare = result["per_class"][0]["iou_curves"]["0.50"]
    assert rare["true_positives"] == 0
    assert rare["recall"] == 0


def test_official_lvis_preserves_gt_ignore_and_not_exhaustive_ignore():
    pytest.importorskip("lvis")
    source = dataset(
        images=[image(1, not_exhaustive=[1]), image(2)],
        annotations=[annotation(1, 1), annotation(2, 2, ignore=1, iscrowd=1)],
    )
    result = evaluate_panel_predictions(source, [1, 2], [
        prediction(1, 1, 0.9),
        prediction(1, 1, 0.8, bbox=[30, 30, 10, 10]),
        prediction(2, 1, 0.7),
    ])
    rare = result["per_class"][0]["iou_curves"]["0.50"]
    assert rare["num_gt"] == 1
    assert rare["true_positives"] == 1
    assert rare["false_positives"] == 0
    assert rare["ignored_matching_detections"] == 2


def test_official_lvis_handles_empty_predictions_without_dummy_detection():
    pytest.importorskip("lvis")
    result = evaluate_panel_predictions(dataset(), [1], [])
    summary = result["summary_by_iou"]["0.50"]
    assert summary["num_gt"] == 1
    assert summary["true_positives"] == summary["false_positives"] == 0
    assert summary["macro_recall"] == 0
    assert summary["macro_precision"] is None
    assert result["counts"]["capped_predictions_all_categories"] == 0


def test_official_lvis_filters_to_selected_images_and_uses_requested_ious():
    pytest.importorskip("lvis")
    source = dataset(
        images=[image(1), image(2)],
        annotations=[annotation(1, 1), annotation(2, 2)],
    )
    result = evaluate_panel_predictions(source, [1], [
        prediction(1, bbox=[2, 0, 10, 10]),  # IoU = 2/3.
        prediction(2),
    ])
    assert result["image_ids"] == [1]
    assert result["counts"]["input_predictions_in_panel"] == 1
    assert result["summary_by_iou"]["0.50"]["true_positives"] == 1
    assert result["summary_by_iou"]["0.75"]["true_positives"] == 0
    assert result["summary_by_iou"]["0.75"]["false_positives"] == 1
