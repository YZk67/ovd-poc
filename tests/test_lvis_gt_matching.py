from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools.lvis_gt_matching import _gt_rows_from_evaluator, match_panel_gt
from tools.pairing_lvis_support import evaluate_panel_predictions


def image(image_id, negatives=(), not_exhaustive=()):
    return {
        "id": image_id, "height": 100, "width": 100,
        "neg_category_ids": list(negatives),
        "not_exhaustive_category_ids": list(not_exhaustive),
    }


def annotation(gt_id, image_id=1, category_id=1, bbox=None, **extra):
    bbox = [0, 0, 10, 10] if bbox is None else bbox
    return {
        "id": gt_id, "image_id": image_id, "category_id": category_id,
        "bbox": bbox, "area": bbox[2] * bbox[3], **extra,
    }


def prediction(image_id=1, category_id=1, score=0.9, bbox=None, **extra):
    return {
        "image_id": image_id, "category_id": category_id, "score": score,
        "bbox": [0, 0, 10, 10] if bbox is None else bbox, **extra,
    }


def dataset(images=None, annotations=None):
    return {
        "images": [image(1)] if images is None else images,
        "annotations": [annotation(501)] if annotations is None else annotations,
        "categories": [
            {"id": 1, "name": "rare_one", "frequency": "r"},
            {"id": 2, "name": "rare_without_gt", "frequency": "r"},
            {"id": 3, "name": "frequent", "frequency": "f"},
        ],
    }


def test_one_selected_detection_for_two_gt_exposes_official_competition():
    pytest.importorskip("lvis")
    source = dataset(annotations=[annotation(501), annotation(902, bbox=[2, 0, 10, 10])])
    predictions = [prediction()]
    result = match_panel_gt(source, [1], predictions)
    rows = {row["gt_id"]: row for row in result["gt_by_iou"]["0.50"]}
    assert rows[501]["matched"]
    assert not rows[902]["matched"]
    assert rows[902]["matched_detection_id"] is None
    assert rows[902]["matched_detection_score"] is None
    assert rows[902]["selected_iou_eligible_count"] == 1
    assert rows[902]["selected_candidates"] == [{
        "detection_id": 1, "score": 0.9, "iou": pytest.approx(2 / 3),
        "matched_gt_id": 501, "ignored": False,
    }]
    summary = result["summary_by_iou"]["0.50"]
    assert (summary["num_gt"], summary["true_positives"], summary["false_positives"]) == (2, 1, 0)
    assert summary["macro_recall"] == 0.5
    assert summary["unmatched_gt_with_selected_iou_eligible_detection"] == 1
    assert summary["unmatched_gt_without_selected_iou_eligible_detection"] == 0
    assert result["gt_by_iou"]["0.75"][1]["selected_iou_eligible_count"] == 0
    reference = evaluate_panel_predictions(source, [1], predictions)
    assert result["per_class"] == reference["per_class"]
    for key, value in reference["summary_by_iou"]["0.50"].items():
        assert summary[key] == value


def test_ignored_gt_reordering_preserves_global_ids_scores_and_input_data():
    pytest.importorskip("lvis")
    source = dataset(annotations=[
        annotation(888, ignore=1, iscrowd=1),
        annotation(17, bbox=[30, 0, 10, 10]),
    ])
    predictions = [
        prediction(score=0.2, bbox=[30, 0, 10, 10], id=9761),
        prediction(score=0.9),
    ]
    original_source, original_predictions = deepcopy(source), deepcopy(predictions)
    result = match_panel_gt(source, [1], predictions)
    rows = {row["gt_id"]: row for row in result["gt_by_iou"]["0.50"]}
    regular, ignored = rows[17], rows[888]
    assert regular["matched"] and not regular["gt_ignore"]
    assert regular["matched_detection_id"] == 1
    assert regular["matched_detection_score"] == 0.2
    assert regular["selected_candidates"] == [{
        "detection_id": 1, "score": 0.2, "iou": 1.0, "matched_gt_id": 17, "ignored": False,
    }]
    assert ignored["matched"] and ignored["gt_ignore"]
    assert ignored["matched_detection_id"] == 2
    assert ignored["matched_detection_score"] == 0.9
    assert ignored["selected_candidates"][0]["ignored"]
    summary = result["summary_by_iou"]["0.50"]
    assert summary["num_gt"] == summary["true_positives"] == 1
    assert summary["ignored_gt"] == summary["ignored_matching_detections"] == 1
    assert source == original_source and predictions == original_predictions
    json.dumps(result, allow_nan=False)


def test_classes_without_positive_gt_still_contribute_verified_negative_fp():
    pytest.importorskip("lvis")
    source = dataset(images=[image(1, negatives=[2]), image(2, negatives=[1])])
    predictions = [
        prediction(1, 1, 0.9), prediction(1, 2, 0.8),
        prediction(2, 1, 0.7), prediction(2, 2, 0.6),
    ]
    result = match_panel_gt(source, [1, 2], predictions)
    assert result["per_class"] == evaluate_panel_predictions(source, [1, 2], predictions)["per_class"]
    assert len(result["gt_by_iou"]["0.50"]) == 1
    summary = result["summary_by_iou"]["0.50"]
    assert summary["true_positives"] == 1
    assert summary["false_positives"] == 2
    assert summary["ignored_before_matching_detections"] == 1
    assert summary["macro_recall"] == 1
    assert summary["macro_recall_valid_categories"] == 1
    assert summary["macro_precision"] == 0.25


def test_empty_predictions_retain_unmatched_gt_without_dummy_detection():
    pytest.importorskip("lvis")
    source = dataset()
    original = deepcopy(source)
    result = match_panel_gt(source, [1], [])
    row = result["gt_by_iou"]["0.50"][0]
    assert row["gt_id"] == 501
    assert not row["matched"]
    assert row["matched_detection_id"] is row["matched_detection_score"] is None
    assert row["selected_iou_eligible_count"] == 0
    assert row["selected_candidates"] == []
    assert result["summary_by_iou"]["0.50"]["macro_recall"] == 0
    assert result["summary_by_iou"]["0.50"]["false_positives"] == 0
    assert source == original


def test_zero_gt_and_empty_image_subset_are_supported():
    pytest.importorskip("lvis")
    source = dataset(images=[image(1, negatives=[1])], annotations=[])
    result = match_panel_gt(source, [1], [prediction()])
    assert result["gt_by_iou"]["0.50"] == []
    summary = result["summary_by_iou"]["0.50"]
    assert summary["num_gt"] == summary["true_positives"] == 0
    assert summary["false_positives"] == 1
    assert summary["macro_recall"] is None
    empty = match_panel_gt(dataset(), [], [prediction()])
    assert empty["gt_by_iou"]["0.50"] == []
    assert empty["summary_by_iou"]["0.50"]["false_positives"] == 0


def test_all_class_topk_and_selected_images_precede_rare_gt_matching():
    pytest.importorskip("lvis")
    source = dataset(
        images=[image(1), image(2)],
        annotations=[annotation(501), annotation(701, image_id=2)],
    )
    result = match_panel_gt(source, [1], [
        prediction(1, 3, 0.99), prediction(1, 1, 0.9), prediction(2, 1, 0.8),
    ], max_dets=1)
    rows = result["gt_by_iou"]["0.50"]
    assert len(rows) == 1 and rows[0]["gt_id"] == 501
    assert not rows[0]["matched"] and rows[0]["selected_iou_eligible_count"] == 0
    assert result["summary_by_iou"]["0.50"]["macro_recall"] == 0


def fake_evaluator():
    # Raw GT order has ignored GT first; official GT order moves it last.
    raw_gt = [{"id": 888}, {"id": 17}]
    raw_dt = [{"id": 22, "score": 0.2}, {"id": 21, "score": 0.9}]
    return SimpleNamespace(
        params=SimpleNamespace(iou_thrs=np.array([1.0])),
        _get_gt_dt=lambda image_id, category_id: (raw_gt, raw_dt),
        ious={(1, 1): np.array([[1.0, 0.0], [0.0, 1.0 - 5e-11]])},
        eval_imgs=[{
            "image_id": 1, "category_id": 1,
            "gt_ids": [17, 888], "dt_ids": [21, 22], "dt_scores": [0.9, 0.2],
            "gt_matches": np.array([[22, 21]]), "dt_matches": np.array([[888, 17]]),
            "gt_ignore": np.array([False, True]), "dt_ignore": np.array([[True, False]]),
        }],
    )


def test_cached_iou_mapping_uses_ids_and_exact_official_threshold_clamp():
    rows = {row["gt_id"]: row for row in _gt_rows_from_evaluator(fake_evaluator(), 0)}
    assert rows[17]["matched_detection_id"] == 22
    assert rows[17]["selected_candidates"][0]["iou"] == 1.0 - 5e-11
    assert rows[888]["matched_detection_id"] == 21
    assert rows[888]["selected_candidates"][0]["ignored"]


def test_nonreciprocal_official_arrays_fail_with_identity_context():
    evaluator = fake_evaluator()
    evaluator.eval_imgs[0]["gt_matches"][0, 0] = 21
    with pytest.raises(ValueError, match="Nonreciprocal.*image=1 category=1"):
        _gt_rows_from_evaluator(evaluator, 0)


@pytest.mark.parametrize("thresholds", [(), (0,), (1.1,), (float("nan"),)])
def test_invalid_iou_thresholds_fail_before_lvis_import(thresholds):
    with pytest.raises(ValueError, match="iou_thresholds"):
        match_panel_gt(dataset(), [1], [], iou_thresholds=thresholds)


def test_unknown_selected_image_fails_before_lvis_import():
    with pytest.raises(ValueError, match="absent"):
        match_panel_gt(dataset(), [9], [])
