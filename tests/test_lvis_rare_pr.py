from types import SimpleNamespace

import numpy as np
import pytest

from tools.report_lvis_rare_pr import (
    operating_points,
    per_class_ap,
    ranked_pr_from_eval_imgs,
    verify_apr,
)


def test_per_class_ap_uses_official_precision_axes_and_ignores_minus_one():
    params = SimpleNamespace(
        area_rng_lbl=["all"], iou_thrs=np.array([0.50, 0.75])
    )
    precision = np.array(
        [
            [[[1.0]], [[0.5]], [[0.0]]],
            [[[0.8]], [[0.4]], [[-1.0]]],
        ]
    )
    recall = np.array([[[0.5]], [[0.25]]])
    result = per_class_ap(precision, recall, 0, params)
    assert result["AP"] == pytest.approx(54.0)
    assert result["AP50"] == pytest.approx(50.0)
    assert result["AP75"] == pytest.approx(60.0)
    assert result["AR"] == pytest.approx(37.5)


def test_ranked_pr_uses_lvis_ignore_flags_for_federated_false_positives():
    params = SimpleNamespace(
        area_rng_lbl=["all"],
        area_rng=[[0, 1]],
        img_ids=[1, 2],
        iou_thrs=np.array([0.50, 0.75]),
    )
    evaluator = SimpleNamespace(
        params=params,
        eval_imgs=[
            {
                "dt_scores": [0.9, 0.2],
                "dt_matches": np.array([[1, 0], [0, 0]]),
                "dt_ignore": np.array([[False, True], [False, True]]),
                "gt_ignore": np.array([False]),
            },
            {
                "dt_scores": [0.8],
                "dt_matches": np.array([[0], [2]]),
                "dt_ignore": np.array([[False], [False]]),
                "gt_ignore": np.array([False]),
            },
        ],
    )
    report = ranked_pr_from_eval_imgs(evaluator, cat_index=0, iou_index=0)
    assert report["num_gt"] == 2
    assert report["valid_detections"] == 2
    assert report["ignored_detections"] == 1
    assert report["true_positives"] == 1
    assert report["false_positives"] == 1
    assert report["curve"][0] == {
        "score": 0.9, "tp": 1, "fp": 0, "precision": 1.0, "recall": 0.5
    }
    assert report["curve"][1]["precision"] == 0.5


def test_apr_checks_per_class_macro_and_expected_checkpoint_result():
    rows = [{"AP": 40.0}, {"AP": 60.0}, {"AP": None}]
    assert verify_apr(rows, 50.0, 50.0, 0.02) == 50.0
    with pytest.raises(ValueError, match="prediction JSON gives APr"):
        verify_apr(rows, 50.0, 41.59, 0.02)
    with pytest.raises(ValueError, match="disagrees with LVIS APr"):
        verify_apr(rows, 49.0, None, 0.02)


def test_operating_points_follow_score_ordered_tp_fp_curve():
    curve = [
        {"score": 0.9, "tp": 1, "fp": 0, "precision": 1.0, "recall": 0.5},
        {"score": 0.2, "tp": 1, "fp": 1, "precision": 0.5, "recall": 0.5},
    ]
    points = operating_points(curve, (0.1, 0.3, 0.95))
    assert points["0.1"] == {
        "tp": 1, "fp": 1, "precision": 0.5, "recall": 0.5
    }
    assert points["0.3"] == {
        "tp": 1, "fp": 0, "precision": 1.0, "recall": 0.5
    }
    assert points["0.95"]["precision"] is None
