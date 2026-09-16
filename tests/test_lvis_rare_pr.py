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


def test_all_curves_saved_in_single_evaluation_including_unobserved_classes(tmp_path, monkeypatch):
    """Synthetic evaluator seam; does not claim real LVIS integration coverage."""
    import json
    import sys
    from tools.report_lvis_rare_pr import main

    categories = [{"id": 1, "name": "koala", "frequency": "r"},
                  {"id": 2, "name": "unobserved", "frequency": "r"}]
    evaluations = []

    class GroundTruth:
        def __init__(self, path):
            self.dataset = {"annotations": [{"category_id": 1}]}

        def get_cat_ids(self):
            return [1, 2]

        def load_cats(self, ids):
            return categories

    class Evaluator:
        def __init__(self, *args):
            self.params = SimpleNamespace(
                area_rng_lbl=["all"], area_rng=[[0, 1e10]], img_ids=[1],
                iou_thrs=np.array([.5, .75]), rec_thrs=np.linspace(0, 1, 101),
            )
            precision = np.ones((2, 101, 2, 1))
            precision[:, :, 1, :] = -1
            self.eval = {"precision": precision, "recall": np.array([[[1.], [-1.]], [[1.], [-1.]]])}
            self.eval_imgs = [{"dt_scores": [.8], "dt_matches": np.array([[1], [1]]),
                               "dt_ignore": np.zeros((2, 1), dtype=bool), "gt_ignore": [False]}, None]

        def run(self):
            evaluations.append(1)

        def get_results(self):
            return {"APr": 1.}

    monkeypatch.setitem(sys.modules, "lvis", SimpleNamespace(
        LVIS=GroundTruth, LVISEval=Evaluator, LVISResults=lambda *a, **kw: None,
    ))
    for name in ("predictions", "annotations"):
        (tmp_path / (name + ".json")).write_text("[]")
    monkeypatch.setattr(sys, "argv", [
        "report_lvis_rare_pr.py", "--predictions", str(tmp_path / "predictions.json"),
        "--annotations", str(tmp_path / "annotations.json"),
        "--output", str(tmp_path / "report.json"), "--expected-apr", "100",
        "--focus", "koala", "--all-curves",
    ])
    main()
    saved = json.loads((tmp_path / "report.json").read_text())
    assert evaluations == [1]
    assert saved["curve_scope"] == "all_rare_categories"
    assert set(saved["focus"]) == {"koala", "unobserved"}
    assert saved["focus"]["unobserved"]["iou_curves"]["0.50"]["num_gt"] == 0
