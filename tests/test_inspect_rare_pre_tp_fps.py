import argparse
import json

import pytest

from tools.compare_rare_pr_reports import file_identity
from tools.inspect_rare_pre_tp_fps import (
    explain_false_positive, run, verify_ranked_stream, xywh_iou,
)


def test_geometric_overlap_labels_are_evidence_not_visual_identity():
    categories = {11: {"name": "koala"}, 99: {"name": "other"}}
    image = {"neg_category_ids": [11], "not_exhaustive_category_ids": []}
    same = {"id": 101, "category_id": 11, "bbox": [0, 0, 10, 10]}
    other = {"id": 202, "category_id": 99, "bbox": [40, 0, 10, 10]}
    detection = {"category_id": 11, "bbox": [0, 0, 10, 10]}
    assert xywh_iou(detection["bbox"], [5, 0, 10, 10]) == pytest.approx(1 / 3)
    duplicate = explain_false_positive(
        detection, image=image, annotations=[same, other], categories=categories,
        threshold=.5, assigned_gt_ids={101},
    )
    assert duplicate["overlap_label"] == "overlaps_already_matched_same_class_gt"
    assert duplicate["overlapping_already_matched_gt_ids"] == [101]
    cross = explain_false_positive(
        {"category_id": 11, "bbox": other["bbox"]}, image=image,
        annotations=[same, other], categories=categories,
        threshold=.5, assigned_gt_ids={101},
    )
    assert cross["overlap_label"] == "overlaps_other_labeled_category"
    assert cross["nearest_other_labeled_gt"]["category_name"] == "other"
    assert cross["nearest_other_labeled_gt"]["iou"] == 1


def test_full_validation_official_matching_identifies_fps_before_tp(tmp_path):
    pytest.importorskip("lvis")
    source = {
        "images": [
            {"id": 1, "width": 100, "height": 100, "file_name": "one.jpg",
             "neg_category_ids": [], "not_exhaustive_category_ids": []},
            {"id": 2, "width": 100, "height": 100, "file_name": "two.jpg",
             "neg_category_ids": [11], "not_exhaustive_category_ids": []},
        ],
        "annotations": [
            {"id": 101, "image_id": 1, "category_id": 11,
             "bbox": [0, 0, 10, 10], "area": 100},
            {"id": 202, "image_id": 2, "category_id": 99,
             "bbox": [40, 0, 10, 10], "area": 100},
        ],
        "categories": [
            {"id": 11, "name": "koala", "frequency": "r"},
            {"id": 99, "name": "other", "frequency": "f"},
        ],
    }
    old = [
        {"image_id": 1, "category_id": 11, "score": .9, "bbox": [0, 0, 10, 10]},
        {"image_id": 2, "category_id": 11, "score": .5, "bbox": [40, 0, 10, 10]},
    ]
    new = [
        {"image_id": 2, "category_id": 11, "score": .99, "bbox": [40, 0, 10, 10]},
        {"image_id": 1, "category_id": 11, "score": .95, "bbox": [20, 0, 10, 10]},
        {"image_id": 1, "category_id": 11, "score": .8, "bbox": [0, 0, 10, 10]},
        {"image_id": 1, "category_id": 11, "score": .7, "bbox": [0, 0, 10, 10]},
    ]
    paths = {}
    for name, content in (("annotations", source), ("old", old), ("new", new)):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(content), encoding="utf-8")
        paths[name] = path
    def summary(tp_rank, tp_score, valid, fp):
        return {
            "valid_detections": valid, "true_positives": 1,
            "false_positives": fp,
            "every_tp": [{"rank": tp_rank, "fp_before": tp_rank - 1,
                          "score": tp_score}],
        }
    comparison = {
        "complete": True,
        "scope": {"max_dets": 300},
        "per_class": [{
            "category_id": 11, "name": "koala",
            "curves": {key: {
                "old": summary(1, .9, 2, 1),
                "new": summary(3, .8, 4, 3),
            } for key in ("0.50", "0.75")},
        }],
        "curve_replay": {
            side: {"sources": {
                "predictions": file_identity(paths[side]),
                "annotations": file_identity(paths["annotations"]),
            }} for side in ("old", "new")
        },
    }
    comparison_path = tmp_path / "ranking.json"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    output = tmp_path / "attribution.json"
    args = argparse.Namespace(
        comparison=str(comparison_path), old_predictions=None, new_predictions=None,
        annotations=None, categories=None, max_dets=300,
        print_fp_classes=["koala"], print_fp_limit=5, output=str(output),
    )
    result = run(args)
    assert json.loads(output.read_text()) == result
    old = result["classes"]["koala"]["old"]["0.50"]
    current = result["classes"]["koala"]["new"]["0.50"]
    assert (old["first_tp_rank"], current["first_tp_rank"]) == (1, 3)
    assert current["false_positives_before_first_tp"] == 2
    assert current["leading_false_positives"][0]["overlap_label"] == "overlaps_other_labeled_category"
    assert current["leading_false_positives"][0]["image_file_name"] == "two.jpg"
    assert current["leading_false_positives"][1]["overlap_label"] == "below_iou_for_same_class_gt"
    assert current["true_positives"][0]["matched_gt_id"] == 101
    assert current["total_false_positives"] == 3
    assert result["classes"]["koala"]["new"]["0.75"]["first_tp_rank"] == 3


def test_rejects_source_hash_or_rank_disagreement(tmp_path):
    # Source integrity is checked before importing LVIS or reading a large result.
    expected = tmp_path / "predictions.json"
    expected.write_text("[]", encoding="utf-8")
    annotations = tmp_path / "annotations.json"
    annotations.write_text("{}", encoding="utf-8")
    comparison = {
        "complete": True,
        "scope": {"max_dets": 300},
        "per_class": [{"category_id": 11, "name": "koala", "curves": {}}],
        "curve_replay": {
            side: {"sources": {
                "predictions": file_identity(expected),
                "annotations": file_identity(annotations),
            }} for side in ("old", "new")
        },
    }
    comparison_path = tmp_path / "ranking.json"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    expected.write_text("[{}]", encoding="utf-8")
    args = argparse.Namespace(
        comparison=str(comparison_path), old_predictions=None, new_predictions=None,
        annotations=None, categories=None, max_dets=300,
        print_fp_classes=[], print_fp_limit=0, output=str(tmp_path / "out.json"),
    )
    with pytest.raises(ValueError, match="old predictions changed"):
        run(args)


def test_rejects_a_tp_order_mismatch_even_when_totals_match():
    ordered = [
        {"rank": 1, "score": .9, "matched_gt_id": None},
        {"rank": 2, "score": .8, "matched_gt_id": 101},
    ]
    expected = {
        "valid_detections": 2, "true_positives": 1, "false_positives": 1,
        "every_tp": [{"rank": 1, "fp_before": 0, "score": .8}],
    }
    with pytest.raises(ValueError, match="TP rank/score differs"):
        verify_ranked_stream(ordered, expected, category="koala", side="new", iou="0.50")
