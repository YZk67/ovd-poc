import pytest
import torch

from tools.analyze_ovd_error_decomposition import (
    classify_rare_detections,
    summarize_rare_false_positives,
)


def test_rare_detection_matching_partitions_false_positive_reasons():
    gt_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    gt_classes = torch.tensor([0])
    detection_boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],   # TP for class 0
            [0.0, 0.0, 10.0, 10.0],   # duplicate class 0
            [0.0, 0.0, 10.0, 10.0],   # wrong class on the GT
            [5.0, 0.0, 15.0, 10.0],   # IoU 1/3, localization
            [20.0, 20.0, 30.0, 30.0], # evaluated background
            [20.0, 20.0, 30.0, 30.0], # unknown category
            [20.0, 20.0, 30.0, 30.0], # not-exhaustive category
        ]
    )
    detection_scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    detection_classes = torch.tensor([0, 0, 1, 0, 1, 2, 3])

    records = classify_rare_detections(
        detection_boxes,
        detection_scores,
        detection_classes,
        gt_boxes,
        gt_classes,
        negative_classes={1},
        not_exhaustive_classes={3},
        iou_threshold=0.5,
        background_iou=0.1,
    )
    assert [record[2] for record in records] == [
        "tp",
        "duplicate",
        "classification",
        "localization",
        "background",
        "ignored_unknown",
        "ignored_not_exhaustive",
    ]


def test_false_positive_summary_keeps_per_class_prefix_through_last_tp():
    records = [
        (0, 0.9, "classification"),
        (0, 0.8, "tp"),
        (0, 0.1, "background"),
        (1, 0.7, "background"),
        (1, 0.6, "ignored_unknown"),
    ]
    report = summarize_rare_false_positives(records)

    assert report["categories_with_predictions"] == 2
    assert report["categories_without_true_positive"] == 1
    assert report["all_selected"]["evaluated"] == 4
    assert report["all_selected"]["false_positives"] == 3
    prefix = report["through_last_true_positive_per_category"]
    assert prefix["evaluated"] == 3
    assert prefix["true_positives"] == 1
    assert prefix["diagnostic_precision"] == pytest.approx(1.0 / 3.0)
