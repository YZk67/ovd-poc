from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import trace_single_rare_gt as cli


def dataset():
    return {"images": [{"id": 5, "width": 100, "height": 100,
                        "neg_category_ids": [], "not_exhaustive_category_ids": []}],
            "categories": [{"id": 1, "name": "scarecrow", "frequency": "r"},
                           {"id": 2, "name": "person", "frequency": "f"}],
            "annotations": [{"id": 91, "image_id": 5, "category_id": 1,
                             "bbox": [0., 0., 10., 10.], "area": 100.}]}


def pred(category=1, bbox=None, score=.9):
    return {"image_id": 5, "category_id": category, "bbox": bbox or [0., 0., 10., 10.], "score": score}


def expected(tp_old=1, tp_new=0):
    return {"category_id": 1, "name": "scarecrow", "gt_annotations": 1,
            "old_AP": 100., "new_AP": 0., "apr_contribution": -100.,
            "iou": {k: {"A": {"num_gt": 1, "tp": tp_old}, "B": {"num_gt": 1, "tp": tp_new}}
                    for k in cli.IOUS}}


def fake_match(source, images, predictions, *, iou_thresholds, max_dets, include_selected_predictions):
    assert images == [5] and include_selected_predictions
    gt = source["annotations"][0]
    selected = [{**p, "detection_id": i+1} for i, p in enumerate(predictions)]
    selected = sorted(selected, key=lambda p: -p["score"])[:max_dets]
    rows = {}
    for threshold in iou_thresholds:
        candidates = [{"detection_id": p["detection_id"], "score": p["score"],
                       "iou": cli.bbox_iou(p["bbox"], gt["bbox"]), "ignored": False,
                       "matched_gt_id": gt["id"] if j == 0 else None}
                      for j, p in enumerate(p for p in selected if p["category_id"] == gt["category_id"]
                                            and cli.bbox_iou(p["bbox"], gt["bbox"]) >= threshold)]
        rows[f"{threshold:.2f}"] = [{"gt_id": gt["id"], "gt_ignore": False, "matched": bool(candidates),
                                    "matched_detection_id": candidates[0]["detection_id"] if candidates else None,
                                    "selected_candidates": candidates}]
    return {"max_dets": max_dets, "selected_predictions": selected, "gt_by_iou": rows}


def describe(predictions, panel=None, exp=None, arm="B"):
    source = dataset()
    panel = panel or fake_match(source, [5], predictions, iou_thresholds=map(float, cli.IOUS),
                                max_dets=300, include_selected_predictions=True)
    return cli.describe_endpoint(source["annotations"][0], {c["id"]: c for c in source["categories"]},
                                 predictions, panel, exp or expected(), arm)


@pytest.mark.parametrize("chunk_size", [1, 7, 256])
@pytest.mark.parametrize("indent", [None, 2])
def test_streamed_array_preserves_image_order_and_escaped_strings(tmp_path, chunk_size, indent):
    records = [pred(score=.5), {**pred(), "image_id": 77},
               {**pred(category=2), "comment": 'unicode 稻草人, quote " and brackets ] }'}]
    path = tmp_path/"predictions.json"
    path.write_text(json.dumps(records, indent=indent, ensure_ascii=False))
    selected, count = cli.scan_image(path, 5, chunk_size=chunk_size)
    assert count == 3 and selected == [records[0], records[2]]


@pytest.mark.parametrize("text", ['{}', '[{},]', '[{"image_id":5}', '[{}', '[1]', '[] garbage', '[{broken}]'])
def test_invalid_stream_rejected(tmp_path, text):
    path = tmp_path/"broken.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        cli.scan_image(path, 5, chunk_size=3)


def test_empty_stream_and_empty_selected_image(tmp_path):
    path = tmp_path/"empty.json"
    path.write_text(' \n [ ]\n')
    assert cli.scan_image(path, 5, chunk_size=1) == ([], 0)
    assert describe([])["by_iou"]["0.50"]["reason"] == "no_eligible_geometry_in_saved_top300"


def test_other_category_geometry_does_not_claim_raw_true_class_rank():
    predictions = [pred(2), pred(1, bbox=[50., 50., 10., 10.], score=.4)]
    before = deepcopy(predictions)
    result = describe(predictions)
    row = result["by_iou"]["0.50"]
    assert row["reason"] == "eligible_geometry_survives_under_other_categories"
    assert row["selected_any_class_eligible"] == 1 and row["selected_true_class_eligible"] == 0
    assert row["raw_query_coverage"] is row["true_class_rank_before_top300"] is None
    assert result["best_any_class_iou"] == 1 and result["best_true_class_iou"] == 0
    assert predictions == before


def test_saved_localization_failure_does_not_claim_all_query_miss():
    result = describe([pred(bbox=[6., 0., 10., 10.])])
    assert result["best_true_class_iou"] == pytest.approx(.25)
    assert result["by_iou"]["0.50"]["reason"] == "no_eligible_geometry_in_saved_top300"
    assert result["by_iou"]["0.50"]["raw_query_coverage"] is None


def test_cap_loss_only_claimed_if_candidate_is_actually_in_input_file():
    predictions = [pred(2, bbox=[50., 50., 10., 10.], score=.99), pred(score=.1)]
    panel = fake_match(dataset(), [5], predictions, iou_thresholds=map(float, cli.IOUS),
                       max_dets=1, include_selected_predictions=True)
    row = describe(predictions, panel)["by_iou"]["0.50"]
    assert row["reason"] == "eligible_correct_pair_in_file_but_removed_by_lvis_cap"
    assert row["file_true_class_eligible"] == 1 and row["selected_true_class_eligible"] == 0


def test_matching_assignment_evidence_not_confused_with_missing_boxes():
    predictions = [pred()]
    panel = fake_match(dataset(), [5], predictions, iou_thresholds=map(float, cli.IOUS),
                       max_dets=300, include_selected_predictions=True)
    for rows in panel["gt_by_iou"].values():
        rows[0].update(matched=False, matched_detection_id=None)
        rows[0]["selected_candidates"][0]["matched_gt_id"] = 92
    result = describe(predictions, panel)
    assert result["by_iou"]["0.50"]["reason"] == "eligible_correct_detection_unmatched_check_assignments"
    assert result["by_iou"]["0.50"]["official"]["selected_candidates"][0]["matched_gt_id"] == 92


def test_correct_tp_and_tie_rank_intervals():
    result = describe([pred(score=.8), pred(2, score=.8)], exp=expected(tp_new=1))
    assert result["by_iou"]["0.50"]["reason"] == "official_true_positive"
    assert all(p["score_rank_interval"] == [1, 2] for p in result["selected_detections_by_gt_iou"])


def test_mismatch_against_stage_tp_stops_analysis():
    with pytest.raises(ValueError, match="TP differs"):
        describe([pred()])


def test_duplicate_gt_not_silently_selected():
    source = dataset()
    source["annotations"].append({**source["annotations"][0], "id": 92})
    with pytest.raises(ValueError, match="GT count"):
        cli.select_target(source, {"per_class": [expected()]}, "scarecrow")


@pytest.fixture
def stage_args(tmp_path):
    source = tmp_path/"source"
    source.mkdir()
    identities = {}
    for k, data in (("annotations", dataset()), ("old_predictions", [pred()]), ("new_predictions", [pred(2)])):
        path = source/(k+".json")
        cli.save_json(path, data)
        identities[k] = cli.file_identity(path)
    stage = tmp_path/"stage"
    stage.mkdir()
    pr = {}
    for side in ("old", "new"):
        path = stage/(side+"_report.json")
        cli.save_json(path, {"mock_pr": side})
        pr[side] = cli.file_identity(path)
    cli.save_json(stage/"report.json", {"complete": True, "endpoint_labels": {"A": "8ep", "B": "12ep"},
                  "A_apr": 100., "B_apr": 0., "delta_apr": -100., "per_class": [expected()],
                  "source_reports": pr, "training_updates": 0, "gpu_inference": False})
    cli.save_json(stage/"inputs.json", {"inputs": identities, "full_iou": True,
                  "stage_labels": {"old": "8ep", "new": "12ep"}, "expected_apr": {"old": 100., "new": 0.}})
    cli.save_json(stage/"COMPLETE.json", {"source_files_unchanged": True, "full_iou": True})
    return cli.parse_args(["--stage-dir", str(stage), "--output", str(tmp_path/"trace/report.json")])


def test_cpu_one_image_pipeline_no_input_writes(stage_args, monkeypatch):
    from tools import lvis_gt_matching
    snapshots = {p: p.read_bytes() for p in Path(stage_args.stage_dir).parent.rglob("*.json")}
    calls = []
    def matcher(*a, **kw):
        calls.append((a, kw))
        return fake_match(*a, **kw)
    monkeypatch.setattr(lvis_gt_matching, "match_panel_gt", matcher)
    result = cli.run(stage_args)
    assert result["complete"] and result["training_updates"] == 0 and result["gpu_inference"] is False
    assert len(calls) == 2 and all(a[1] == [5] for a, kw in calls)
    assert all(len(kw["iou_thresholds"]) == 10 and kw["max_dets"] == 300 for a, kw in calls)
    assert result["new_boxes_nearest_old_tp"][0]["iou_to_old_tp"] == 1
    for p, before in snapshots.items():
        assert p.read_bytes() == before


@pytest.mark.parametrize("target", ["old_predictions", "annotations", "pr"])
def test_changed_sources_fail_before_matching(stage_args, monkeypatch, target):
    from tools import lvis_gt_matching
    monkeypatch.setattr(lvis_gt_matching, "match_panel_gt", lambda *a, **k: pytest.fail("Must not match"))
    stage = Path(stage_args.stage_dir)
    path = stage/"old_report.json" if target == "pr" else stage.parent/"source"/(target+".json")
    path.write_text("changed")
    with pytest.raises(ValueError, match="changed since stage"):
        cli.run(stage_args)
    assert not Path(stage_args.output).exists()


def test_stage_output_protected(stage_args):
    stage_args.output = str(Path(stage_args.stage_dir)/"new.json")
    with pytest.raises(ValueError, match="new output outside"):
        cli.run(stage_args)


def test_help_does_not_import_model_or_lvis():
    result = subprocess.run([sys.executable, "-S", cli.__file__, "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "CPU only" in result.stdout


def test_real_official_matching_exposes_capped_all_class_predictions():
    pytest.importorskip("lvis")
    from tools.lvis_gt_matching import match_panel_gt
    source = dataset()
    predictions = [pred(2, score=.99), pred(score=.1)]
    before = deepcopy(predictions)
    result = match_panel_gt(source, [5], predictions, max_dets=1, include_selected_predictions=True)
    assert len(result["selected_predictions"]) == 1
    assert result["selected_predictions"][0]["category_id"] == 2
    assert result["gt_by_iou"]["0.50"][0]["matched"] is False
    assert predictions == before
