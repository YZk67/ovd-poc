from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.decoder_aux_outcome_ops import (
    box_iou_xywh, build_top300_transitions, distribution, summarize_gradient_logs, summarize_pr,
)
from tools import analyze_decoder_aux_outcome as cli


def curve_change(shared, lost=0, gained=0, fp=0):
    return {"delta_AP_points": shared+lost+gained,
            "delta_max_recall_points": gained+lost,
            "ap_partition_points": {"shared_recall_precision": shared,
                                     "lost_recall_support": lost,
                                     "gained_recall_support": gained},
            "same_recall_mean_delta_fp_before": fp}


def comparison():
    rows = []
    for category_id, name, gt, delta, fp in ((1, "one", 1, -20., 2.), (2, "two", 4, 10., -1.)):
        old_curve = {"true_positives": gt, "false_positives": 2}
        new_curve = {"true_positives": gt, "false_positives": 2+int(fp)}
        rows.append({"category_id": category_id, "name": name, "gt_annotations": gt,
                     "old_AP": 50., "new_AP": 50+delta, "delta_AP": delta,
                     "old_AP50": 50., "new_AP50": 50+delta, "delta_AP50": delta,
                     "old_AP75": 50., "new_AP75": 50+delta, "delta_AP75": delta,
                     "curves": {key: {"old": old_curve, "new": new_curve}
                                for key in ("0.50", "0.75")},
                     "recall_ranking": {"0.50": curve_change(delta, fp=fp),
                                        "0.75": curve_change(delta, fp=fp)}})
    return {"per_class": rows, "macro_attribution": {
        "valid_category_count": 2,
        "per_class": [{"category_id": r["category_id"], "name": r["name"],
                       "gt_annotations": r["gt_annotations"], "old_AP": r["old_AP"],
                       "new_AP": r["new_AP"], "delta_AP": r["delta_AP"],
                       "old_AP50": r["old_AP50"], "new_AP50": r["new_AP50"],
                       "delta_AP50": r["delta_AP50"], "old_AP75": r["old_AP75"],
                       "new_AP75": r["new_AP75"], "delta_AP75": r["delta_AP75"],
                       "apr_contribution": r["delta_AP"]/2} for r in rows]}}


def gt(gt_id, category, matched, candidates=0):
    selected = ([{"detection_id": 9, "score": .5, "iou": .8,
                  "matched_gt_id": None, "ignored": False}] if candidates else [])
    return {"gt_id": gt_id, "image_id": 10, "category_id": category, "gt_ignore": False,
            "matched": matched, "matched_detection_id": 9 if matched else None,
            "matched_detection_score": .5 if matched else None,
            "selected_iou_eligible_count": len(selected), "selected_candidates": selected}


def matches(rows):
    return {"gt_by_iou": {key: deepcopy(rows) for key in ("0.50", "0.75")},
            "summary_by_iou": {key: {"true_positives": sum(r["matched"] and not r["gt_ignore"] for r in rows),
                                     "false_positives": 3} for key in ("0.50", "0.75")}}


def test_box_iou_and_final_top300_loss_reasons_close():
    assert box_iou_xywh([0, 0, 10, 10], [5, 0, 10, 10]) == pytest.approx(1/3)
    dataset = {"categories": [{"id": 1, "name": "one"}, {"id": 2, "name": "two"}],
               "annotations": [{"id": 1, "image_id": 10, "category_id": 1, "bbox": [0,0,10,10]},
                               {"id": 2, "image_id": 10, "category_id": 2, "bbox": [20,0,10,10]},
                               {"id": 3, "image_id": 10, "category_id": 2, "bbox": [40,0,10,10]}]}
    before = matches([gt(1,1,True), gt(2,2,True), gt(3,2,True)])
    after = matches([gt(1,1,False,1), gt(2,2,False), gt(3,2,False)])
    predictions = [{"image_id": 10, "category_id": 1, "bbox": [20,0,10,10], "score": .4}]
    report = build_top300_transitions(before, after, dataset, [], predictions, comparison())
    row = report["summary_by_iou"]["0.50"]
    assert row["delta_tp"] == -3
    assert row["loss_reasons"] == {
        "correct_class_iou_candidate_unmatched": 1,
        "geometry_present_true_class_absent_from_final_top300": 1,
        "no_iou_eligible_detection_in_final_top300": 1,
    }
    assert sum(row["transitions"].values()) == 3
    per_class = report["per_class_by_iou"]["0.50"]
    one = next(row for row in per_class if row["category_id"] == 1)
    assert one["full_validation_pr"]["delta_fp"] == 2
    decline = report["declining_class_summary_by_iou"]["0.50"]
    assert decline["classes"] == 1
    assert decline["A_hit_B_miss_gt"] == 1
    assert decline["classes_with_more_fp_before_same_recall_tp"] == 1


def test_gt_transition_gain_and_ignore_do_not_pollute_denominator():
    dataset = {"categories": [{"id": 1, "name": "one"}],
               "annotations": [{"id": 1, "image_id": 10, "category_id": 1, "bbox": [0,0,10,10]},
                               {"id": 2, "image_id": 10, "category_id": 1, "bbox": [20,0,10,10]}]}
    a, b = gt(1,1,False), gt(1,1,True)
    ignored_a, ignored_b = gt(2,1,True), gt(2,1,False)
    ignored_a["gt_ignore"] = ignored_b["gt_ignore"] = True
    result = build_top300_transitions(matches([a, ignored_a]), matches([b, ignored_b]), dataset,
                                      [], [], comparison())
    assert result["summary_by_iou"]["0.50"]["transitions"] == {"A_miss_B_hit": 1}
    assert result["gt_by_iou"]["0.50"][1]["transition"] == "ignored"


def test_pr_partition_uses_valid_macro_denominator():
    value = summarize_pr(comparison())["0.50"]
    assert value["macro_delta_partition_points"]["shared_recall_precision"] == -5
    assert value["macro_delta_AP_points"] == -5
    assert value["closure_error_points"] == 0
    assert value["classes_with_more_fps_before_same_recall_tp"] == 1
    assert value["classes_with_fewer_fps_before_same_recall_tp"] == 1


def test_distribution_interpolates_percentiles_and_rejects_nonfinite():
    value = distribution([1, 2, 3])
    assert value["median"] == value["mean"] == 2
    assert value["p05"] == pytest.approx(1.1)
    with pytest.raises(ValueError, match="finite"):
        distribution([1, float("nan")])


def test_gradient_summary_has_no_false_class_correlation():
    def row(i, offset):
        return {"iteration": i, "full_detector_norm": 10+offset,
                "clip_coefficient": .05, "aux_norm": 2+offset, "core_post_norm": .4}
    logs = {arm: [[row(i, rank) for i in range(2)] for rank in range(4)] for arm in ("A", "B")}
    summary = summarize_gradient_logs(logs)
    assert summary["by_arm"]["A"]["full_detector_norm"]["n"] == 2
    assert summary["by_arm"]["B"]["aux_over_full_norm"]["mean"] > 0
    assert "not identifiable" in summary["correlation_scope"]
    logs["A"][0][0]["clip_coefficient"] = 2
    with pytest.raises(ValueError, match="invalid norms"):
        summarize_gradient_logs(logs)


def test_selected_prediction_loader_validates_top300(tmp_path, monkeypatch):
    path = tmp_path/"predictions.json"
    rows = [{"image_id": 1, "category_id": 1, "bbox": [0,0,1,1], "score": .5}]
    path.write_text(json.dumps(rows))
    identity = cli.file_identity(path)
    selected = cli.read_selected_predictions(identity, {1}, {1}, {1})
    assert selected == rows
    bad = rows*301
    path.write_text(json.dumps(bad))
    identity = cli.file_identity(path)
    with pytest.raises(ValueError, match="top-300"):
        cli.read_selected_predictions(identity, {1}, {1}, {1})


def test_missing_or_non500_trial_rejected_before_cpu_evaluation(tmp_path, monkeypatch):
    (tmp_path/"manifest.json").write_text("{}")
    monkeypatch.setattr(cli, "read_manifest", lambda p: {"fingerprint": "x", "updates": 100})
    (tmp_path/"summary.json").write_text(json.dumps({"complete": True,
        "manifest_fingerprint": "x", "updates_per_arm": 100}))
    with pytest.raises(ValueError, match="500-update"):
        cli.validate_trial(tmp_path)
