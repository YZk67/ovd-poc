from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.analyze_rare_gt_transitions import (
    parse_args, run, sha256_file, validate_manifest, validate_prediction_scope,
    validate_replay, validate_source, write_json,
)
from tools.pairing_lvis_support import evaluate_panel_predictions, select_panel


def fixture_dataset():
    return {
        "images": [
            {"id": i, "height": 100, "width": 100,
             "neg_category_ids": [4, 30] if i == 3 else [],
             "not_exhaustive_category_ids": []}
            for i in (1, 2, 3)
        ],
        "categories": [
            {"id": 30, "name": "rare_two", "frequency": "r"},
            {"id": 17, "name": "frequent", "frequency": "f"},
            {"id": 4, "name": "rare_one", "frequency": "r"},
        ],
        "annotations": [
            {"id": 10 + i, "image_id": i, "category_id": category,
             "bbox": [0, 0, 10, 10], "area": 100}
            for i, category in ((1, 4), (2, 30))
        ],
    }


@pytest.fixture
def inputs(tmp_path):
    pytest.importorskip("lvis")
    dataset = fixture_dataset()
    annotations = tmp_path / "annotations.json"
    write_json(annotations, dataset)
    panel = select_panel(dataset, negative_images=1)
    predictions = [
        {"image_id": i, "category_id": category, "score": 0.9, "bbox": [0, 0, 10, 10]}
        for i, category in ((1, 4), (2, 30))
    ]
    source = {"panel": panel, "rows": [], "panel_pr": {}, "complete": True}
    for variant in ("old_old", "new_new"):
        selected = predictions if variant == "old_old" else predictions[:1]
        write_json(tmp_path / f"{variant}_predictions.json", selected)
        source["panel_pr"][variant] = evaluate_panel_predictions(dataset, panel["image_ids"], selected)
        for threshold in (.5, .75):
            for i, class_index in ((1, 0), (2, 2)):
                eligible = variant == "old_old" or i == 1
                source["rows"].append({
                    "image_id": i, "gt_id": 10 + i, "class_index": class_index,
                    "iou_threshold": threshold, "variant": variant,
                    "eligible": eligible, "pair_topk": eligible,
                })
    manifest_inputs = {
        "annotations_sha256": sha256_file(annotations), "image_ids": panel["image_ids"],
        "max_dets": 300, "iou_thresholds": [.5, .75],
    }
    source["fingerprint"] = hashlib.sha256(json.dumps(manifest_inputs, sort_keys=True).encode()).hexdigest()
    write_json(tmp_path / "pairing_cache" / "manifest.json", {
        "fingerprint": source["fingerprint"], "inputs": manifest_inputs,
    })
    write_json(tmp_path / "report.json", source)
    for label, values in (("old", (50., 60.)), ("new", (45., 40.))):
        write_json(tmp_path / f"{label}_ap.json", {
            "max_dets": 300, "rare_category_count": 2,
            "official_apr": sum(values) / 2,
            "per_class": [
                {"category_id": category, "name": name, "gt_annotations": 1, "AP": ap}
                for category, name, ap in zip((4, 30), ("rare_one", "rare_two"), values)
            ],
        })
    return tmp_path, dataset, source


def args_for(path, *extra):
    return parse_args([
        "--source-json", str(path / "report.json"),
        "--annotations", str(path / "annotations.json"),
        "--old-ap-report", str(path / "old_ap.json"),
        "--new-ap-report", str(path / "new_ap.json"),
        "--output", str(path / "transitions.json"), *extra,
    ])


def test_cpu_saved_artifact_pipeline_closes_matches_and_joins_ap(inputs, capsys):
    path, _, _ = inputs
    before = {p: p.read_bytes() for p in path.rglob("*.json")}
    report = run(args_for(path))
    assert report["complete"]
    assert report["annotations_manifest_verified"] and report["native_matching_replay_verified"]
    assert len(report["gt_by_iou"]["0.50"]) == 2
    summary = report["summary_by_iou"]["0.50"]
    assert summary["loss_reason_counts"]["no_eligible_query"] == 1
    assert summary["net_tp_change"] == -1
    assert summary["macro_recall_change_points"] == pytest.approx(-50.)
    by_class = {row["category_id"]: row for row in report["per_class_by_iou"]["0.50"]}
    assert by_class[30]["delta_AP"] == -20.
    assert by_class[30]["apr_contribution_points"] == -10.
    assert "[save]" in capsys.readouterr().out
    for file, content in before.items():
        assert file.read_bytes() == content
    assert json.loads((path / "transitions.json").read_text())["complete"]


def test_incomplete_report_and_missing_rare_image_fail(inputs):
    _, dataset, source = inputs
    changed = deepcopy(source)
    changed["complete"] = False
    with pytest.raises(ValueError, match="incomplete"):
        validate_source(changed, dataset)
    changed = deepcopy(source)
    changed["panel"]["rare_image_ids"] = [1]
    with pytest.raises(ValueError, match="every rare-GT image"):
        validate_source(changed, dataset)


def test_wrong_prediction_file_fails_native_replay(inputs):
    path, _, _ = inputs
    write_json(path / "new_new_predictions.json", [])
    with pytest.raises(ValueError, match="new_new.*replay"):
        run(args_for(path))
    assert not (path / "transitions.json").exists()


def test_changed_scores_fail_even_when_official_matches_do_not_change(inputs):
    path, _, _ = inputs
    file = path / "new_new_predictions.json"
    predictions = json.loads(file.read_text())
    predictions[0]["score"] = .8
    write_json(file, predictions)
    with pytest.raises(ValueError, match="ranked score/TP/FP curve differs"):
        run(args_for(path))


def test_non_hundredth_iou_keys_follow_saved_panel_format(inputs):
    _, dataset, source = inputs
    changed = deepcopy(source)
    for panel in changed["panel_pr"].values():
        panel["iou_thresholds"] = [.625]
        panel["summary_by_iou"] = {"0.625": panel["summary_by_iou"]["0.50"]}
    assert validate_source(changed, dataset)[1] == [.625]


def test_per_class_replay_checked_even_if_totals_match():
    expected = {
        "summary_by_iou": {"0.50": {"num_gt": 2, "true_positives": 1,
            "false_positives": 0, "macro_recall": .5, "macro_recall_valid_categories": 2}},
        "per_class": [
            {"category_id": cid, "iou_curves": {"0.50": {"num_gt": 1, "true_positives": tp}}}
            for cid, tp in ((4, 1), (30, 0))
        ],
    }
    actual = {"summary_by_iou": expected["summary_by_iou"], "gt_by_iou": {"0.50": [
        {"category_id": cid, "gt_ignore": False, "matched": matched}
        for cid, matched in ((4, False), (30, True))
    ]}}
    with pytest.raises(ValueError, match="category=4"):
        validate_replay(actual, expected, "old_old")


def test_annotation_hash_mismatch_fails_before_matching(inputs):
    path, dataset, _ = inputs
    dataset["annotations"][0]["bbox"] = [1, 1, 10, 10]
    write_json(path / "annotations.json", dataset)
    with pytest.raises(ValueError, match="Annotations SHA256"):
        run(args_for(path))


def test_output_cannot_overwrite_source(inputs):
    path, _, _ = inputs
    with pytest.raises(ValueError, match="overwrite"):
        run(args_for(path, "--output", str(path / "report.json")))


def test_both_ap_reports_required_together(inputs):
    path, _, _ = inputs
    args = args_for(path)
    args.old_ap_report = None
    with pytest.raises(ValueError, match="both"):
        run(args)


def test_prediction_scope_rejects_wrong_image_or_cap():
    with pytest.raises(ValueError, match="outside"):
        validate_prediction_scope([{"image_id": 7, "category_id": 1}], [1], {1}, 300)
    with pytest.raises(ValueError, match="exceed"):
        validate_prediction_scope([{"image_id": 1, "category_id": 1}] * 2, [1], {1}, 1)


def test_help_does_not_require_model_dependencies():
    script = Path(__file__).resolve().parents[1] / "tools/analyze_rare_gt_transitions.py"
    result = subprocess.run([sys.executable, "-S", str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--source-json" in result.stdout and "--old-ap-report" in result.stdout
