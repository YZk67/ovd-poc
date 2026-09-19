import copy
import json
import math

import pytest

from tools.rare_ap_concentration_ops import analyze, format_results


def reports():
    counts = [1, 1, 2, 4, 5, 9, 10, 19, 20, 0]
    changes = [50, -50, 10, -20, 5, -5, 15, 0, 5, None]
    result = []
    for side in (0, 1):
        rows = []
        for i, (gt, delta) in enumerate(zip(counts, changes), 1):
            ap = None if delta is None else 50.+side*delta
            rows.append(dict(category_id=i, name=f"category_{i}", gt_annotations=gt, AP=ap, AP50=ap, AP75=ap))
        result.append(dict(max_dets=300, rare_category_count=len(rows), per_class=rows,
                           official_apr=math.fsum(r["AP"] for r in rows if r["AP"] is not None)/9))
    return result


def test_macro_bins_and_signs_use_all_valid_classes_not_ontology_or_gt_weight():
    a, p = reports()
    report = analyze(a, p)
    assert report["valid_rare_categories"] == 9
    assert report["excluded_undefined_categories"] == 1
    assert report["delta_apr"] == pytest.approx(10/9)
    assert sum(r["apr_contribution"] for r in report["per_class"]) == pytest.approx(10/9)
    assert [s["classes"] for s in report["gt_strata"]] == [2, 2, 2, 2, 1]
    assert [s["apr_contribution"] for s in report["gt_strata"]] == pytest.approx([0, -10/9, 0, 15/9, 5/9])
    total = report["global"]
    assert (total["positive_classes"], total["negative_classes"], total["zero_classes"]) == (5, 3, 1)
    assert total["positive_apr_contribution"]+total["negative_apr_contribution"] == pytest.approx(report["delta_apr"])
    assert report["singleton"]["apr_contribution"] == 0
    assert report["singleton"]["positive_apr_contribution"] == pytest.approx(50/9)
    assert report["singleton"]["share_of_all_positive_contribution"] == pytest.approx(50/85)
    assert report["singleton"]["non_singleton"]["mean_delta_AP"] == pytest.approx(10/7)
    assert report["gt_at_least"]["5"]["classes"] == 5
    assert report["gt_at_least"]["10"]["median_delta_AP"] == 5


def test_tail_sensitivity_removes_both_signs_with_two_explicit_denominators():
    result = analyze(*reports())
    drop_gain, drop_loss, balanced = result["sensitivity"][:3]
    assert drop_gain["removed"][0]["category_id"] == 1
    assert drop_gain["retained"]["mean_delta_AP"] == -5
    assert drop_gain["retained"]["apr_contribution"] == pytest.approx(-40/9)
    assert drop_loss["retained"]["mean_delta_AP"] == 7.5
    assert balanced["retained"]["mean_delta_AP"] == pytest.approx(10/7)
    assert balanced["retained"]["apr_contribution"] == pytest.approx(10/9)
    # Removing all nonzero deltas leaves the unchanged class; no NaN.
    last_balanced = next(s for s in result["sensitivity"] if s["policy"] == "balanced_gains_and_losses" and s["requested_k_per_tail"] == 5)
    assert last_balanced["retained"]["mean_delta_AP"] == 0
    singleton_drop = next(s for s in result["sensitivity"] if s["policy"] == "single_GT_largest_gains")
    assert singleton_drop["retained"]["mean_delta_AP"] == -5
    assert all(r["gt_annotations"] == 1 for r in singleton_drop["removed"])
    json.dumps(result, allow_nan=False)
    text = format_results(result)
    assert "NOT official APr" in text and "no" in text.lower()
    assert "without ALL single-GT classes" in text


def test_zero_gain_empty_bins_and_small_dataset_never_emit_nan():
    a, _ = reports()
    a["per_class"] = [a["per_class"][0]]
    a["rare_category_count"] = 1
    a["official_apr"] = 50.
    r = analyze(a, a)
    assert r["singleton"]["share_of_all_positive_contribution"] is None
    assert r["singleton"]["non_singleton"]["mean_delta_AP"] is None
    assert r["gt_strata"][1]["mean_delta_AP"] is None
    p = copy.deepcopy(a)
    for key in ("AP", "AP50", "AP75"):
        p["per_class"][0][key] = 100.
    p["official_apr"] = 100.
    r = analyze(a, p)
    assert r["sensitivity"][0]["retained"]["classes"] == 0
    assert r["sensitivity"][0]["retained"]["mean_delta_AP"] is None
    json.dumps(r, allow_nan=False)


def test_all_178_valid_of_337_rare_used_not_only_focus_classes():
    a, p = reports()
    for side, report in enumerate((a, p)):
        report["per_class"] = [dict(category_id=i, name=str(i), gt_annotations=int(i < 178),
                                    **{k: (40.+side if i < 178 else None) for k in ("AP", "AP50", "AP75")})
                               for i in range(337)]
        report["rare_category_count"] = 337
        report["official_apr"] = 40.+side
    r = analyze(a, p)
    assert len(r["per_class"]) == 178
    assert r["per_class"][0]["apr_contribution"] == pytest.approx(1/178)
    assert r["delta_apr"] == 1


@pytest.mark.parametrize("mutation", ["category_set", "valid_mask", "gt", "zero_gt", "nan", "macro", "duplicate", "negative_ap", "max_dets"])
def test_incompatible_or_invalid_reports_fail_closed(mutation):
    a, p = reports()
    row = p["per_class"][0]
    if mutation == "category_set":
        row["category_id"] = 777
    elif mutation == "valid_mask":
        row["AP"] = None
        p["official_apr"] = sum(r["AP"] for r in p["per_class"] if r["AP"] is not None)/8
    elif mutation == "gt":
        row["gt_annotations"] = 5
    elif mutation == "zero_gt":
        row["gt_annotations"] = a["per_class"][0]["gt_annotations"] = 0
    elif mutation == "nan":
        row["AP"] = float("nan")
    elif mutation == "macro":
        p["official_apr"] += 1
    elif mutation == "duplicate":
        row["category_id"] = 2
    elif mutation == "negative_ap":
        row["AP"] = -1
    else:
        a["max_dets"] = p["max_dets"] = 100
    with pytest.raises(ValueError):
        analyze(a, p)


def test_arm_order_controls_sign_and_never_rounds_before_comparing():
    a, p = reports()
    assert analyze(p, a)["delta_apr"] == pytest.approx(-analyze(a, p)["delta_apr"])
    r = analyze(a, p)
    assert r["per_class"][0]["A_AP"] == 50
    assert r["per_class"][0]["P_AP"] == 100
