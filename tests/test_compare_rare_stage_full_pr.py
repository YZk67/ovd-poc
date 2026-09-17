from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import compare_rare_stage_full_pr as cli
from tools.decoder_aux_postmortem_ops import analyze
from test_decoder_aux_postmortem import reports, curve


@pytest.fixture
def args(tmp_path):
    old, new = reports()
    for label, report in (("old", old), ("new", new)):
        (tmp_path/f"{label}.json").write_text(json.dumps(report))
    return cli.parse_args([
        "--old-report", str(tmp_path/"old.json"), "--new-report", str(tmp_path/"new.json"),
        "--expected-old-apr", str(old["official_apr"]),
        "--expected-new-apr", str(new["official_apr"]),
        "--output", str(tmp_path/"out/report.json"),
    ])


def test_endpoint_labels_all_iou_macro_no_inherited_focus(args):
    before = {p: p.read_bytes() for p in (Path(args.old_report), Path(args.new_report))}
    result = cli.run(args)
    assert result["endpoint_labels"] == {"A": "8ep", "B": "12ep"}
    assert "original_focus" not in result
    assert "auxiliary decoder gradient blocked" not in json.dumps(result)
    assert result["valid_classes"] == 5 and result["all_rare_categories"] == 6
    assert len(result["by_iou"]) == 10 and len(result["per_class"]) == 5
    assert sum(result["global"]["partition_contribution"].values()) == pytest.approx(result["delta_apr"])
    assert result["gpu_inference"] is False and result["training_updates"] == 0
    assert json.loads(Path(args.output).read_text()) == result
    for p, content in before.items():
        assert p.read_bytes() == content


def test_intermediate_iou_effect_not_inferred_from_ap50_ap75(args):
    new = cli.load_json(args.new_report)
    # All other IoUs unchanged; .95 loses recall on this class.
    entry = new["focus"]["lasagna"]
    entry["iou_curves"]["0.95"] = curve([], 6)
    old_ap = entry["AP"]
    entry["AP"] *= .9
    next(r for r in new["per_class"] if r["name"] == "lasagna")["AP"] *= .9
    new["official_apr"] -= old_ap*.1/5
    cli.save_json(args.new_report, new)
    args.expected_new_apr = new["official_apr"]
    result = cli.run(args)
    row = next(r for r in result["per_class"] if r["name"] == "lasagna")
    assert row["delta_AP50"] == row["delta_AP75"] == 0
    assert row["ap_partition_points"]["lost_recall_support"] == pytest.approx(-old_ap*.1)


@pytest.mark.parametrize("mutation", ["missing_iou", "bad_ap", "bad_gt", "bad_raw", "focus_only"])
def test_incomplete_or_inconsistent_report_has_no_output(args, mutation):
    report = cli.load_json(args.new_report)
    if mutation == "missing_iou":
        del report["focus"]["keg"]["iou_curves"]["0.65"]
    elif mutation == "bad_ap":
        report["official_apr"] += .001
    elif mutation == "bad_gt":
        report["focus"]["lasagna"]["iou_curves"]["0.60"]["num_gt"] = 5
    elif mutation == "bad_raw":
        report["focus"]["lasagna"]["iou_curves"]["0.95"] = curve([], 6)
    else:
        report["curve_scope"] = "focus_only"
    cli.save_json(args.new_report, report)
    with pytest.raises(ValueError):
        cli.run(args)
    assert not Path(args.output).exists()


def test_leave_one_category_out_is_exhaustive_sensitivity_not_changed_official_ap():
    rows = [{"name": "positive", "gt_annotations": 1, "delta_AP": 100., "apr_contribution": 50.},
            {"name": "negative", "gt_annotations": 8, "delta_AP": -50., "apr_contribution": -25.}]
    before = deepcopy(rows)
    result = cli.category_sensitivity(rows, 25.)
    assert len(result["all_exclusions"]) == 2
    assert result["min"]["remaining_mean_delta_AP"] == -50.
    assert result["max"]["remaining_mean_delta_AP"] == 100.
    assert [r["excluded_category"] for r in result["opposite_sign_exclusions"]] == ["positive"]
    assert result["remaining_classes"] == 1 and rows == before
    assert cli.category_sensitivity(rows[:1], 100.)["defined"] is False


def test_shared_math_keeps_original_ab_defaults():
    default = analyze(*reports())
    generic = analyze(*reports(), focus=(), labels={"A": "8ep", "B": "12ep"})
    assert default["endpoint_labels"]["B"] == "auxiliary decoder gradient blocked"
    assert len(default["original_focus"]["classes"]) == 3
    for key in ("global", "per_class", "by_iou", "gt_strata", "delta_apr"):
        assert default[key] == generic[key]


@pytest.mark.parametrize("field,value", [("expected_old_apr", float("nan")), ("top_classes", 0),
                                        ("new_label", "8ep"), ("old_label", " ")])
def test_invalid_options_rejected(args, field, value):
    setattr(args, field, value)
    with pytest.raises(ValueError):
        cli.run(args)
    assert not Path(args.output).exists()


def test_input_and_existing_output_protected(args):
    args.output = args.old_report
    before = Path(args.old_report).read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        cli.run(args)
    assert Path(args.old_report).read_bytes() == before
    args.new_report = args.old_report
    with pytest.raises(ValueError, match="different files"):
        cli.run(args)


def test_standalone_without_site_packages(args):
    command = [sys.executable, "-S", cli.__file__,
               "--old-report", args.old_report, "--new-report", args.new_report,
               "--expected-old-apr", str(args.expected_old_apr),
               "--expected-new-apr", str(args.expected_new_apr), "--output", args.output]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "8ep -> 12ep" in result.stdout
    assert "No historical training source identified" in result.stdout
