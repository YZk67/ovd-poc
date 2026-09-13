from collections import Counter
from copy import deepcopy
import json

import pytest

from tools.rare_transition_ops import build_transition_report


CATEGORIES = [
    {"id": 10, "name": "rare_a", "frequency": "r"},
    {"id": 20, "name": "base", "frequency": "f"},
    {"id": 30, "name": "rare_b", "frequency": "r"},
    {"id": 40, "name": "rare_empty", "frequency": "r"},
]


def _gt(gt_id, category_id=10, *, matched=False, ignored=False, candidates=None):
    detection_id = 100 + gt_id if matched else None
    if candidates is None:
        candidates = ([{"detection_id": detection_id, "score": 0.8, "iou": 0.9,
                        "matched_gt_id": gt_id, "ignored": ignored}] if matched else [])
    return {"gt_id": gt_id, "image_id": 1, "category_id": category_id,
            "gt_ignore": ignored, "matched": matched, "matched_detection_id": detection_id,
            "matched_detection_score": 0.8 if matched else None,
            "selected_iou_eligible_count": len(candidates), "selected_candidates": candidates}


def _matches(rows, *, fp=3, thresholds=("0.50",)):
    valid = [row for row in rows if not row["gt_ignore"]]
    counts = Counter(row["category_id"] for row in valid)
    hits = Counter(row["category_id"] for row in valid if row["matched"])
    summary = {"num_gt": len(valid), "true_positives": sum(hits.values()), "false_positives": fp,
               "valid_detections": sum(hits.values()) + fp,
               "macro_recall": sum(hits[c] / counts[c] for c in counts) / len(counts) if counts else None,
               "macro_recall_valid_categories": len(counts)}
    return {"summary_by_iou": {key: deepcopy(summary) for key in thresholds},
            "gt_by_iou": {key: deepcopy(rows) for key in thresholds}}


def _pairing(old, new, *, states=None):
    result = []
    classes = {row["id"]: index for index, row in enumerate(CATEGORIES)}
    states = states or {}
    for label, source in (("old", old), ("new", new)):
        for threshold, rows in source["gt_by_iou"].items():
            for row in rows:
                if row["gt_ignore"]:
                    continue
                eligible, pair = states.get((label, row["gt_id"]), (True, bool(row["matched"])))
                result.append({"gt_id": row["gt_id"], "image_id": row["image_id"],
                               "class_index": classes[row["category_id"]], "iou_threshold": float(threshold),
                               "variant": f"{label}_{label}", "eligible": eligible,
                               "pair_topk": pair, "best_iou": 0.9 if eligible else 0.2})
    return result


def _example():
    # Equal-class recall: rare_a has 3 GT, rare_b has 1, rare_empty has none.
    old = _matches([_gt(1, matched=True), _gt(2, matched=True), _gt(3), _gt(4, 30, matched=True)])
    new = _matches([_gt(1), _gt(2, matched=True), _gt(3, matched=True), _gt(4, 30)], fp=5)
    rows = _pairing(old, new, states={("new", 1): (False, False), ("new", 4): (True, False)})
    return old, new, rows


def test_macro_recall_closes_with_unequal_class_sizes_and_both_directions():
    old, new, rows = _example()
    report = build_transition_report(old, new, rows, CATEGORIES)
    summary = report["summary_by_iou"]["0.50"]
    assert summary["num_gt"] == 4 and summary["num_valid_classes"] == 2
    assert summary["valid_category_ids"] == [10, 30]  # No-hit rare_b still counts.
    assert summary["transitions"] == {"both_hit": 1, "both_miss": 0, "old_hit_new_miss": 2, "old_miss_new_hit": 1}
    assert summary["net_tp_change"] == -1 and summary["net_fp_change"] == 2
    assert summary["macro_recall_change_points"] == pytest.approx(-50.0)
    assert summary["loss_macro_recall_points"] == pytest.approx(-100 / 6 - 50)
    assert summary["gain_macro_recall_points"] == pytest.approx(100 / 6)
    assert summary["loss_reason_macro_recall_points"]["no_eligible_query"] == pytest.approx(-100 / 6)
    assert summary["loss_reason_macro_recall_points"]["correct_pair_below_topk"] == -50
    assert summary["gain_reason_counts"]["correct_pair_below_topk"] == 1
    assert abs(summary["invariants"]["macro_recall_closure_error_points"]) < 1e-8
    classes = {row["category_id"]: row for row in report["per_class_by_iou"]["0.50"]}
    assert classes[10]["macro_recall_contribution_points"] == 0
    assert classes[30]["macro_recall_contribution_points"] == -50
    assert classes[30]["new_tp"] == 0
    assert report["ap_comparison"] is None
    json.dumps(report, allow_nan=False)


def test_missing_side_distinguishes_competition_membership_and_unresolved():
    assigned = {"detection_id": 999, "score": 0.7, "iou": 0.8, "matched_gt_id": 80, "ignored": False}
    unassigned = {**assigned, "detection_id": 998, "matched_gt_id": None, "ignored": True}
    old = _matches([_gt(1, matched=True), _gt(2, matched=True), _gt(3, matched=True)])
    new = _matches([_gt(1, candidates=[assigned]), _gt(2), _gt(3, candidates=[unassigned])])
    rows = _pairing(old, new, states={("new", gt_id): (True, True) for gt_id in (1, 2, 3)})
    report = build_transition_report(old, new, rows, CATEGORIES)
    reasons = {row["gt_id"]: row["reason"] for row in report["gt_by_iou"]["0.50"]}
    assert reasons == {1: "matching_competition", 2: "candidate_membership_disagreement", 3: "candidate_matching_or_postprocess"}


def test_ignored_matches_do_not_contribute_and_need_no_pairing_rows():
    old = _matches([_gt(1, matched=True), _gt(2, 30, matched=True, ignored=True)])
    new = _matches([_gt(1), _gt(2, 30, ignored=True)])
    report = build_transition_report(old, new, _pairing(old, new), CATEGORIES)
    summary = report["summary_by_iou"]["0.50"]
    assert summary["num_gt"] == 1 and summary["num_ignored_gt"] == 1
    assert summary["num_valid_classes"] == 1
    assert summary["macro_recall_change_points"] == -100
    assert report["gt_by_iou"]["0.50"][1]["transition"] == "ignored"
    assert report["gt_by_iou"]["0.50"][1]["macro_recall_contribution_points"] == 0


def test_empty_valid_gt_keeps_undefined_macro_recall_json_safe():
    old = _matches([_gt(1, ignored=True, matched=True)])
    new = _matches([_gt(1, ignored=True)])
    report = build_transition_report(old, new, [], CATEGORIES)
    summary = report["summary_by_iou"]["0.50"]
    assert summary["num_valid_classes"] == 0
    assert summary["macro_recall_change_points"] is None
    assert report["per_class_by_iou"]["0.50"] == []
    json.dumps(report, allow_nan=False)


def test_close_iou_thresholds_and_input_order_do_not_merge_or_change_results():
    old = _matches([_gt(1, matched=True), _gt(2)], thresholds=("0.501", "0.504"))
    new = _matches([_gt(1), _gt(2)], thresholds=("0.501", "0.504"))
    rows = _pairing(old, new)
    expected = build_transition_report(old, new, rows, CATEGORIES)
    assert set(expected["summary_by_iou"]) == {"0.501", "0.504"}
    rows.reverse()
    for source in (old, new):
        for official_rows in source["gt_by_iou"].values():
            official_rows.reverse()
    assert build_transition_report(old, new, rows, CATEGORIES[::-1]) == expected


@pytest.mark.parametrize("mutation,match", [
    (lambda old, new, rows: rows.append(deepcopy(rows[0])), "Duplicate pairing"),
    (lambda old, new, rows: rows.pop(), "Missing pairing"),
    (lambda old, new, rows: rows[0].update(class_index=2), "class_index disagrees"),
    (lambda old, new, rows: rows[0].update(eligible=False, pair_topk=False), "official hit contradicts"),
    (lambda old, new, rows: rows[0].update(pair_topk=False), "official hit contradicts"),
    (lambda old, new, rows: new["gt_by_iou"]["0.50"][0].update(gt_ignore=True), "gt_ignore differs"),
    (lambda old, new, rows: new["gt_by_iou"]["0.50"].pop(), "GT universes differ"),
    (lambda old, new, rows: old["gt_by_iou"]["0.50"].append(deepcopy(old["gt_by_iou"]["0.50"][0])), "Duplicate official"),
    (lambda old, new, rows: old["summary_by_iou"]["0.50"].update(true_positives=999), "true_positives"),
    (lambda old, new, rows: old["summary_by_iou"]["0.50"].update(num_gt=999), "num_gt"),
    (lambda old, new, rows: old["summary_by_iou"]["0.50"].update(macro_recall=0.123), "macro_recall"),
    (lambda old, new, rows: old["gt_by_iou"]["0.50"][0].pop("selected_candidates"), "required keys"),
])
def test_contradictions_fail_instead_of_assigning_a_reason(mutation, match):
    old, new, rows = _example()
    mutation(old, new, rows)
    with pytest.raises(ValueError, match=match):
        build_transition_report(old, new, rows, CATEGORIES)


def _ap_reports():
    counts = {10: 3, 30: 1, 40: 0}
    return {
        label: {"official_apr": sum(values[:2]) / 2,
                "per_class": [{"category_id": c, "name": next(r["name"] for r in CATEGORIES if r["id"] == c),
                               "gt_annotations": counts[c], "AP": value} for c, value in zip((10, 30, 40), values)]}
        for label, values in (("old", [20.0, 80.0, None]), ("new", [30.0, 40.0, None]))
    }


def test_ap_join_is_category_level_and_not_instance_recall_attribution():
    old, new, rows = _example()
    original = deepcopy((old, new, rows))
    meta = {row["id"]: {**row, "class_index": index, "gt_annotations": {10: 3, 30: 1}.get(row["id"], 0)}
            for index, row in enumerate(CATEGORIES)}
    report = build_transition_report(old, new, rows, meta, ap_reports=_ap_reports())
    ap = report["ap_comparison"]
    assert ap["delta_apr"] == -15
    assert sum(row["apr_contribution_points"] or 0 for row in ap["per_class"]) == -15
    assert ap["valid_category_ids"] == [10, 30]
    classes = {row["category_id"]: row for row in report["per_class_by_iou"]["0.50"]}
    assert classes[10]["delta_AP"] == 10 and classes[10]["apr_contribution_points"] == 5
    assert classes[10]["macro_recall_contribution_points"] == 0
    assert classes[30]["apr_contribution_points"] == -20
    assert classes[30]["macro_recall_contribution_points"] == -50
    assert ap["per_class"][-1]["delta_AP"] is None
    assert all("apr_contribution_points" not in row for row in report["gt_by_iou"]["0.50"])
    assert (old, new, rows) == original


@pytest.mark.parametrize("mutate,match", [
    (lambda reports: reports["new"]["per_class"].pop(), "every rare category"),
    (lambda reports: reports["new"]["per_class"][0].update(gt_annotations=4), "raw GT count"),
    (lambda reports: reports["new"]["per_class"][0].update(AP=None), "valid AP categories"),
    (lambda reports: reports["old"].update(official_apr=30), "does not close"),
    (lambda reports: reports["old"]["per_class"][0].update(AP=float("nan")), "finite"),
])
def test_ap_join_rejects_mismatched_coverage_counts_and_unclosed_ap(mutate, match):
    old, new, rows = _example()
    reports = _ap_reports()
    mutate(reports)
    with pytest.raises(ValueError, match=match):
        build_transition_report(old, new, rows, CATEGORIES, ap_reports=reports)


def test_ap_raw_counts_include_ignored_gt_while_recall_weights_do_not():
    old = _matches([_gt(1, matched=True), _gt(2, ignored=True)])
    new = _matches([_gt(1), _gt(2, ignored=True)])
    reports = {label: {"official_apr": ap, "per_class": [
        {"category_id": 10, "gt_annotations": 2, "AP": ap},
        {"category_id": 30, "gt_annotations": 0, "AP": None},
        {"category_id": 40, "gt_annotations": 0, "AP": None},
    ]} for label, ap in (("old", 90.0), ("new", 20.0))}
    report = build_transition_report(old, new, _pairing(old, new), CATEGORIES, ap_reports=reports)
    assert report["ap_comparison"]["delta_apr"] == -70
    assert report["summary_by_iou"]["0.50"]["macro_recall_change_points"] == -100
