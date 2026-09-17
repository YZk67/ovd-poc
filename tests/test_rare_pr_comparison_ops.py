from copy import deepcopy
import json

import pytest

from tools.rare_pr_comparison_ops import (
    compare_recall_and_ranking, compare_reports, summarize_ranked_curve,
)


def _curve(matches, *, num_gt=2, scores=None, ignored=0):
    scores = scores if scores is not None else [0.99 - index * 0.01 for index in range(len(matches))]
    points = []
    tp = fp = 0
    for matched, score in zip(matches, scores):
        tp += int(matched)
        fp += int(not matched)
        points.append({"score": score, "tp": tp, "fp": fp,
                       "precision": tp / (tp + fp), "recall": tp / num_gt if num_gt else 0.0})
    return {"num_gt": num_gt, "valid_detections": len(points), "ignored_detections": ignored,
            "true_positives": tp, "false_positives": fp, "curve": points}


def _report(*, is_new=False):
    # One FP preceding each TP gives AP50=AP75=50. Reversing that ranking
    # to put both TP first gives 100, despite identical terminal TP/FP counts.
    matches = [False, True, False, True] if is_new else [True, True, False, False]
    ap = 50.0 if is_new else 100.0
    row = {"category_id": 7, "name": "koala", "gt_annotations": 2,
           "AP": ap, "AP50": ap, "AP75": ap}
    empty_row = {"category_id": 12, "name": "absent", "gt_annotations": 0,
                 "AP": None, "AP50": None, "AP75": None}
    return {"max_dets": 300, "official_apr": ap, "rare_category_count": 2,
            "per_class": [row, empty_row],
            "focus": {"koala": {**row, "iou_curves": {key: _curve(matches, ignored=5)
                                                        for key in ("0.50", "0.75")}}}}


def test_each_tp_has_valid_rank_and_cumulative_fp_before_it():
    summary = summarize_ranked_curve(_curve([False, False, True, False, True], ignored=9))
    assert summary["true_positives"] == 2
    assert summary["false_positives"] == 3
    assert summary["first_tp_rank"] == 3
    assert summary["fp_before_first_tp"] == 2
    assert summary["first_tp_score"] == 0.97
    assert summary["every_tp"] == [
        {"tp_index": 1, "rank": 3, "score": 0.97, "fp_before": 2, "precision": 1 / 3, "recall": 0.5},
        {"tp_index": 2, "rank": 5, "score": 0.95, "fp_before": 3, "precision": 2 / 5, "recall": 1.0},
    ]
    assert summary["fp_before_tp_median"] == 2.5
    assert summary["max_recall"] == 1
    assert summary["ignored_detections"] == 9
    assert summary["interpolated_AP_points"] == pytest.approx(40)


@pytest.mark.parametrize("matches,num_gt", [([], 2), ([False, False], 2), ([], 0), ([False], 0)])
def test_no_tp_is_null_even_when_present_curve_is_all_fp_or_empty(matches, num_gt):
    summary = summarize_ranked_curve(_curve(matches, num_gt=num_gt))
    for key in ("first_tp_rank", "fp_before_first_tp", "first_tp_score", "fp_before_tp_median"):
        assert summary[key] is None
    assert summary["every_tp"] == []
    assert summary["max_recall"] == (0 if num_gt else None)
    assert summary["interpolated_AP_points"] == (0 if num_gt else None)
    json.dumps(summary, allow_nan=False)


def test_score_ties_preserve_source_order_instead_of_resorting_tp_first():
    source = _curve([False, True], num_gt=1, scores=[0.5, 0.5])
    original = deepcopy(source)
    summary = summarize_ranked_curve(source)
    assert summary["has_score_ties"] is True
    assert summary["first_tp_rank"] == 2
    assert summary["fp_before_first_tp"] == 1
    assert source == original


@pytest.mark.parametrize("mutation,error", [
    (lambda c: c["curve"][1].update(score=1.1), "descending"),
    (lambda c: c["curve"][1].update(tp=0, fp=0), "advance exactly one"),
    (lambda c: c["curve"][0].update(tp=1, fp=1), "advance exactly one"),
    (lambda c: c["curve"][0].update(precision=1.0), "precision denominator"),
    (lambda c: c["curve"][1].update(recall=1.0), "recall denominator"),
    (lambda c: c["curve"][0].update(ignored=True), "ignored detections"),
    (lambda c: c["curve"][0].update(score=float("nan")), "finite number"),
    (lambda c: c.update(valid_detections=3), "curve length"),
    (lambda c: c.update(false_positives=3), "reported totals"),
    (lambda c: c.update(ignored_detections=-1), "nonnegative integer"),
    (lambda c: c.update(num_gt=0), "TP exceeds"),
])
def test_invalid_rank_streams_are_rejected(mutation, error):
    source = _curve([False, True])
    mutation(source)
    with pytest.raises(ValueError, match=error):
        summarize_ranked_curve(source)


def test_101_point_interpolation_includes_endpoint_and_precision_envelope():
    # Recall reaches 0.5: thresholds 0.00 through 0.50 inclusive are valid.
    source = _curve([False, True, False], num_gt=2)
    source["official_interpolated_pr"] = [
        {"recall": index * 0.01, "precision": 0.5 if index <= 50 else 0.0}
        for index in range(101)
    ]
    assert summarize_ranked_curve(source)["interpolated_AP_points"] == pytest.approx(100 * 25.5 / 101)
    source["official_interpolated_pr"][0]["precision"] = 0
    with pytest.raises(ValueError, match="raw/official precision"):
        summarize_ranked_curve(source)


def test_zero_gt_official_curve_requires_null_precision():
    source = _curve([], num_gt=0)
    source["official_interpolated_pr"] = [
        {"recall": index * 0.01, "precision": None} for index in range(101)
    ]
    assert summarize_ranked_curve(source)["interpolated_AP_points"] is None
    source["official_interpolated_pr"][0]["precision"] = 0
    with pytest.raises(ValueError, match="raw/official precision"):
        summarize_ranked_curve(source)


def test_compare_ap_and_rank_regression_with_unchanged_final_tp_fp():
    old, new = _report(), _report(is_new=True)
    originals = deepcopy((old, new))
    result = compare_reports(old, new, ["koala"])
    assert result["complete"] is True
    assert result["missing_curves"] == []
    assert result["delta_apr"] == -50
    assert result["scope"]["valid_rare_category_count"] == 1
    row = result["per_class"][0]
    for metric in ("AP", "AP50", "AP75"):
        assert row[f"old_{metric}"] == 100
        assert row[f"new_{metric}"] == 50
        assert row[f"delta_{metric}"] == -50
    for curves in row["curves"].values():
        assert curves["old"]["true_positives"] == curves["new"]["true_positives"] == 2
        assert curves["old"]["false_positives"] == curves["new"]["false_positives"] == 2
        assert curves["old"]["fp_before_first_tp"] == 0
        assert curves["new"]["fp_before_first_tp"] == 1
        assert [point["fp_before"] for point in curves["new"]["every_tp"]] == [1, 2]
    assert (old, new) == originals
    json.dumps(result, allow_nan=False)


def test_missing_curve_is_null_and_distinct_from_zero_tp():
    old, new = _report(), _report(is_new=True)
    del new["focus"]["koala"]["iou_curves"]["0.75"]
    result = compare_reports(old, new, ["koala"])
    assert result["complete"] is False
    assert result["missing_curves"] == [{"side": "new", "name": "koala", "iou": "0.75"}]
    row = result["per_class"][0]
    assert row["curves"]["0.75"]["new"] is None
    assert row["new_AP75"] == 50
    assert row["curves"]["0.50"]["new"] is not None


def test_absent_focus_and_null_ap_stay_explicitly_missing():
    result = compare_reports(_report(), _report(is_new=True), ["absent", "koala"])
    assert [row["name"] for row in result["per_class"]] == ["absent", "koala"]
    assert result["per_class"][0]["delta_AP"] is None
    assert len(result["missing_curves"]) == 4
    assert result["complete"] is False


@pytest.mark.parametrize("mutation,error", [
    (lambda r: r.update(max_dets=100), "max_dets disagree"),
    (lambda r: r.update(rare_category_count=3), "rare_category_count"),
    (lambda r: r.update(official_apr=49), "macro APr"),
    (lambda r: r["per_class"][1].update(category_id=13), "category ID sets"),
    (lambda r: r["per_class"][1].update(name="renamed"), "name disagree"),
    (lambda r: r["per_class"][1].update(gt_annotations=1), "gt_annotations disagree"),
    (lambda r: r["per_class"][1].update(category_id=7), "duplicate category"),
    (lambda r: r["focus"]["koala"].update(AP50=51), "focus koala AP50"),
    (lambda r: r["focus"]["koala"].update(name="wrong"), "focus koala name"),
    (lambda r: r["focus"]["koala"].update(category_id=8), "focus koala category_id"),
])
def test_incompatible_or_inconsistent_reports_fail(mutation, error):
    new = _report(is_new=True)
    mutation(new)
    with pytest.raises(ValueError, match=error):
        compare_reports(_report(), new, ["koala"])


def test_raw_ap_closure_catches_stale_curves_even_when_focus_ap_metadata_agrees():
    new = _report(is_new=True)
    new["focus"]["koala"]["iou_curves"]["0.50"] = _curve([True, True, False, False])
    with pytest.raises(ValueError, match="raw 101-point AP50"):
        compare_reports(_report(), new, ["koala"])


def test_apr_uses_all_valid_classes_with_equal_weight_outside_requested_focus():
    old, new = _report(), _report(is_new=True)
    other = {"category_id": 99, "name": "many_gt", "gt_annotations": 1000,
             "AP": 20.0, "AP50": 30.0, "AP75": 10.0}
    for report in (old, new):
        report["per_class"].append(deepcopy(other))
        report["rare_category_count"] = 3
        report["official_apr"] = (report["official_apr"] + 20) / 2
    new["per_class"].reverse()
    result = compare_reports(old, new, ["koala"])
    assert result["old_apr"] == 60
    assert result["new_apr"] == 35
    assert result["delta_apr"] == -25
    assert result["scope"]["valid_rare_category_count"] == 2
    assert [row["name"] for row in result["per_class"]] == ["koala"]


def test_curve_gt_denominators_must_agree_between_sides_and_ious():
    old, new = _report(), _report(is_new=True)
    # Both classes retain AP=0, making denominator consistency an independent
    # check beyond AP closure; the saved GT annotation count is still two.
    for report in (old, new):
        report["official_apr"] = 0
        for entry in (report["per_class"][0], report["focus"]["koala"]):
            entry.update(AP=0, AP50=0, AP75=0)
        report["focus"]["koala"]["iou_curves"] = {
            key: _curve([False], num_gt=2) for key in ("0.50", "0.75")
        }
    new["focus"]["koala"]["iou_curves"]["0.75"]["num_gt"] = 1
    with pytest.raises(ValueError, match="num_gt disagree"):
        compare_reports(old, new, ["koala"])


@pytest.mark.parametrize("focus,error", [("koala", "sequence"), (["koala", "koala"], "duplicates"),
                                          (["nonexistent"], "not rare"), ([7], "strings")])
def test_invalid_focus_is_not_silently_dropped(focus, error):
    with pytest.raises(ValueError, match=error):
        compare_reports(_report(), _report(is_new=True), focus)


def test_macro_attribution_covers_nonfocus_classes_with_valid_class_denominator():
    old, new = _report(), _report(is_new=True)
    for report, ap in ((old, 20), (new, 40)):
        report["per_class"].append({
            "category_id": 99, "name": "many_gt", "gt_annotations": 1000,
            "AP": ap, "AP50": ap, "AP75": ap,
        })
        report["rare_category_count"] = 3
        report["official_apr"] = (report["official_apr"] + ap) / 2
    result = compare_reports(old, new, ["koala"])
    summary = result["macro_attribution"]
    assert summary["valid_category_count"] == 2  # Not taxonomy size=3 or GT weighted.
    assert summary["negative_contribution"] == -25
    assert summary["positive_contribution"] == 10
    assert summary["sum_contribution"] == result["delta_apr"] == -15
    assert [row["name"] for row in summary["per_class"]] == ["koala", "many_gt"]
    assert [row["gt_range"] for row in summary["gt_strata"]] == ["2-4", "20+"]
    assert sum(row["apr_contribution"] for row in summary["gt_strata"]) == -15


def partition(old_matches, new_matches, num_gt=2):
    return compare_recall_and_ranking(
        summarize_ranked_curve(_curve(old_matches, num_gt=num_gt)),
        summarize_ranked_curve(_curve(new_matches, num_gt=num_gt)),
    )


def test_same_terminal_counts_can_lose_ap_only_on_shared_recall():
    change = partition([True, True, False, False], [False, True, False, True])
    assert change["delta_max_recall_points"] == 0
    assert change["ap_partition_points"] == pytest.approx({
        "shared_recall_precision": -50, "lost_recall_support": 0, "gained_recall_support": 0,
    })
    assert change["same_recall_mean_delta_fp_before"] == 1.5
    assert change["same_recall_ordinals_with_more_fp_before"] == 2


def test_pure_recall_loss_includes_101_point_endpoint_correctly():
    change = partition([True, True], [True])
    assert change["delta_max_recall_points"] == -50
    assert change["ap_partition_points"] == pytest.approx({
        "shared_recall_precision": 0, "lost_recall_support": -5000 / 101,
        "gained_recall_support": 0,
    })
    assert change["same_recall_mean_delta_fp_before"] == 0


def test_recall_and_ranking_can_both_worsen_or_have_opposite_signs():
    change = partition([True, True], [False, True])
    parts = change["ap_partition_points"]
    assert parts["lost_recall_support"] < 0 and parts["shared_recall_precision"] < 0
    crossing = partition([True], [False, True, False, True])
    parts = crossing["ap_partition_points"]
    assert parts["gained_recall_support"] > 0 and parts["shared_recall_precision"] < 0
    assert sum(parts.values()) == pytest.approx(crossing["delta_AP_points"])


def test_extra_low_rank_fps_do_not_look_like_ranking_loss():
    change = partition([True], [True, False, False])
    assert change["delta_AP_points"] == 0
    assert change["same_recall_mean_delta_fp_before"] == 0
    assert all(value == 0 for value in change["ap_partition_points"].values())


def test_empty_or_missing_curves_are_not_mislabelled_as_zero_fp_evidence():
    change = partition([True, True], [])
    assert sum(change["ap_partition_points"].values()) == pytest.approx(-100)
    assert change["ap_partition_points"]["lost_recall_support"] == pytest.approx(-100)
    assert change["ap_partition_points"]["shared_recall_precision"] == 0
    assert change["same_recall_mean_delta_fp_before"] is None
    assert compare_recall_and_ranking(None, summarize_ranked_curve(_curve([]))) is None
    no_gt = summarize_ranked_curve(_curve([], num_gt=0))
    assert compare_recall_and_ranking(no_gt, no_gt) is None


@pytest.mark.parametrize("no_tp", [[], [False], [False, False]])
def test_first_or_last_tp_includes_zero_recall_bin_in_support_change(no_tp):
    loss = partition([False, True], no_tp, num_gt=1)
    gain = partition(no_tp, [False, True], num_gt=1)
    assert loss["ap_partition_points"] == pytest.approx({
        "shared_recall_precision": 0., "lost_recall_support": -50., "gained_recall_support": 0.})
    assert gain["ap_partition_points"] == pytest.approx({
        "shared_recall_precision": 0., "lost_recall_support": 0., "gained_recall_support": 50.})


def test_partition_closes_over_small_binary_streams_and_is_antisymmetric():
    from itertools import product
    streams = [list(bits) for length in range(5) for bits in product((False, True), repeat=length)
               if sum(bits) <= 2]
    for before in streams:
        for after in streams:
            forward, reverse = partition(before, after), partition(after, before)
            fp, rp = forward["ap_partition_points"], reverse["ap_partition_points"]
            assert sum(fp.values()) == pytest.approx(forward["delta_AP_points"], abs=1e-10)
            assert fp["lost_recall_support"] == pytest.approx(-rp["gained_recall_support"])
            assert fp["shared_recall_precision"] == pytest.approx(-rp["shared_recall_precision"])
