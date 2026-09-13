from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools.compare_rare_pr_reports import parse_args, run, save_json


def raw_curve(false_positives):
    scores = [.95, .90, .85][:false_positives] + [.8]
    points = [
        {"score": score, "tp": 0, "fp": i + 1, "precision": 0., "recall": 0.}
        for i, score in enumerate(scores[:-1])
    ]
    points.append({"score": .8, "tp": 1, "fp": false_positives,
                   "precision": 1. / (false_positives + 1), "recall": 1.})
    return {"num_gt": 1, "valid_detections": len(points), "ignored_detections": 0,
            "true_positives": 1, "false_positives": false_positives, "curve": points}


@pytest.fixture
def inputs(tmp_path):
    dataset = {
        "images": [{"id": i, "width": 100, "height": 100,
                    "neg_category_ids": [1] if i != 1 else [],
                    "not_exhaustive_category_ids": []} for i in (1, 2, 3, 4)],
        "annotations": [{"id": 10, "image_id": 1, "category_id": 1,
                         "bbox": [0, 0, 10, 10], "area": 100}],
        "categories": [{"id": 1, "name": "koala", "frequency": "r"},
                       {"id": 2, "name": "unobserved", "frequency": "r"}],
    }
    save_json(tmp_path / "annotations.json", dataset)
    for label, fp in (("old", 0), ("new", 3)):
        predictions = [{"image_id": 1, "category_id": 1, "score": .8, "bbox": [0, 0, 10, 10]}]
        predictions += [
            {"image_id": i + 2, "category_id": 1, "score": score, "bbox": [0, 0, 10, 10]}
            for i, score in enumerate([.95, .90, .85][:fp])
        ]
        pred_path = tmp_path / (label + "_predictions.json")
        save_json(pred_path, predictions)
        ap = 100. / (fp + 1)
        report = {
            "predictions": str(pred_path), "prediction_bytes": pred_path.stat().st_size,
            "annotations": str(tmp_path / "annotations.json"),
            "official_apr": ap, "max_dets": 300, "rare_category_count": 2,
            "per_class": [
                {"category_id": 1, "name": "koala", "gt_annotations": 1,
                 "AP": ap, "AP50": ap, "AP75": ap, "AR": 100.},
                {"category_id": 2, "name": "unobserved", "gt_annotations": 0,
                 "AP": None, "AP50": None, "AP75": None, "AR": None},
            ],
            "focus": {},
        }
        save_json(tmp_path / (label + "_report.json"), report)
    return tmp_path


def arguments(path, *extra):
    return parse_args([
        "--old-report", str(path / "old_report.json"),
        "--new-report", str(path / "new_report.json"),
        "--annotations", str(path / "annotations.json"),
        "--focus", "koala", "--output", str(path / "comparison.json"), *extra,
    ])


def add_existing_curves(path):
    for label, fp in (("old", 0), ("new", 3)):
        file = path / (label + "_report.json")
        report = json.loads(file.read_text())
        row = report["per_class"][0]
        report["focus"]["koala"] = {
            **row, "iou_curves": {key: raw_curve(fp) for key in ("0.50", "0.75")},
        }
        save_json(file, report)


def test_reports_only_retains_ap_and_explicit_missing_curves(inputs, capsys):
    comparison = run(arguments(inputs))
    assert comparison["complete"] is False
    assert comparison["per_class"][0]["delta_AP50"] == -75.
    assert len(comparison["missing_curves"]) == 4
    assert comparison["per_class"][0]["curves"]["0.50"]["new"] is None
    assert "MISSING_CURVE" in capsys.readouterr().out
    assert json.loads((inputs / "comparison.json").read_text())["complete"] is False


def test_existing_curves_need_no_prediction_or_annotation_reads(inputs, monkeypatch, capsys):
    add_existing_curves(inputs)
    import builtins
    previous_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in ("lvis", "torch", "detectron2", "detrex"):
            raise AssertionError("Reports-only comparison must not load evaluation/model dependencies")
        return previous_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    args = arguments(inputs, "--fill-missing-curves")
    args.annotations = str(inputs / "not_present_annotations.json")
    args.old_predictions = str(inputs / "not_present_old_predictions.json")
    args.new_predictions = str(inputs / "not_present_new_predictions.json")
    report = run(args)
    assert report["complete"]
    assert report["curve_replay"] == {}
    curve = report["per_class"][0]["curves"]["0.50"]["new"]
    assert curve["first_tp_rank"] == 4 and curve["fp_before_first_tp"] == 3
    assert "1, 4, 3, 0.8" in capsys.readouterr().out


def test_optional_full_val_cpu_fill_and_report_immutability(inputs):
    pytest.importorskip("lvis")
    before = {p: p.read_bytes() for p in inputs.glob("*.json")}
    result = run(arguments(inputs, "--fill-missing-curves"))
    assert result["complete"]
    assert set(result["curve_replay"]) == {"old", "new"}
    for key in ("0.50", "0.75"):
        curves = result["per_class"][0]["curves"][key]
        assert curves["old"]["true_positives"] == curves["new"]["true_positives"] == 1
        assert curves["old"]["fp_before_first_tp"] == 0
        assert curves["new"]["fp_before_first_tp"] == 3
    for path, content in before.items():
        assert path.read_bytes() == content


def test_output_cannot_overwrite_source_report_or_prediction(inputs):
    for name in ("old_report.json", "new_predictions.json", "annotations.json"):
        with pytest.raises(ValueError, match="overwrite"):
            run(arguments(inputs, "--output", str(inputs / name)))


def test_overrides_do_not_allow_overwriting_recorded_original_sources(inputs):
    for original in ("old_predictions.json", "annotations.json"):
        with pytest.raises(ValueError, match="overwrite"):
            run(arguments(inputs,
                          "--annotations", str(inputs / "relocated_annotations.json"),
                          "--old-predictions", str(inputs / "relocated_predictions.json"),
                          "--output", str(inputs / original)))


def test_expected_apr_guard(inputs):
    with pytest.raises(ValueError, match="old report APr"):
        run(arguments(inputs, "--expected-old-apr", "45.2037"))


def test_real_cli_works_without_site_packages_with_complete_curves(inputs):
    add_existing_curves(inputs)
    root = Path(__file__).resolve().parents[1]
    foreign = inputs / "foreign"
    (foreign / "tools").mkdir(parents=True)
    (foreign / "tools/__init__.py").write_text('"""Conflicting tools package."""\n')
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(foreign), str(root)])}
    command = [sys.executable, "-S", str(root / "tools/compare_rare_pr_reports.py"),
               "--old-report", str(inputs / "old_report.json"),
               "--new-report", str(inputs / "new_report.json"),
               "--focus", "koala", "--output", str(inputs / "comparison.json")]
    completed = subprocess.run(command, cwd=inputs, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "complete=True" in completed.stdout


def test_real_cli_missing_curves_has_nonzero_exit(inputs):
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([
        sys.executable, "-S", str(root / "tools/compare_rare_pr_reports.py"),
        "--old-report", str(inputs / "old_report.json"),
        "--new-report", str(inputs / "new_report.json"),
        "--focus", "koala", "--output", str(inputs / "comparison.json"),
    ], cwd=inputs, capture_output=True, text=True)
    assert completed.returncode == 2, completed.stdout + completed.stderr
    assert "missing PR is NOT zero FP" in completed.stdout


def test_help_needs_only_stdlib():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, "-S", str(root / "tools/compare_rare_pr_reports.py"), "--help"],
                               capture_output=True, text=True)
    assert completed.returncode == 0 and "--fill-missing-curves" in completed.stdout
