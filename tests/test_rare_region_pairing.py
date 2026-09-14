from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from tools.compare_rare_pr_reports import file_identity
from tools.diagnose_rare_fp_regions import (
    compare_banks, fill_missing_images, fingerprint, run, validate_sample,
)
from tools.rare_region_pairing_ops import (
    collect_regions, compare_region, fp_overlap_report, replay_sample,
)


PROTOCOL = {"alpha": 0., "beta": .3, "novel_scale": 3., "max_dets": 3}
CATEGORIES = {11: {"name": "koala"}, 22: {"name": "other"}}


def bank(side="new", signature="test"):
    return {
        "fingerprint": signature, "label": side, "category_ids": [11, 22],
        "prototypes": torch.eye(2).unsqueeze(1).repeat(1, 2, 1),
        "temperature": .07, "logit_scale": 5., "cls_bias": -3.,
        "vlm_text": torch.eye(2), "vlm_temperature": 1.,
        "novel_mask": torch.tensor([True, False]),
    }


def sample(side="new", signature="test", image_id=1):
    return {
        "fingerprint": signature, "label": side, "image_id": image_id,
        "features": torch.eye(2), "roi_features": torch.eye(2),
        "query_boxes": torch.tensor([[0., 0., 10., 10.], [40., 0., 50., 10.]]),
        "native_replay_check": {"logit_max_abs_error": 0., "score_max_abs_error": 0.},
    }


def region(replay):
    return {
        "category": "koala", "source_side": "new", "kind": "fp", "image_id": 1,
        "detection_id": 99, "box_xyxy": [0., 0., 10., 10.],
        "score": float(replay["scores"][0, 0]), "class_global_rank": 1,
    }


def test_pairs_other_checkpoint_by_geometry_not_query_index():
    current = replay_sample(sample(), bank(), PROTOCOL)
    old_sample = sample("old")
    for key in ("features", "roi_features", "query_boxes"):
        old_sample[key] = old_sample[key].flip(0)
    # Change the old detector branch only, on its geometrically corresponding q1.
    old_sample["features"][1] = torch.tensor([.7, .3])
    old = replay_sample(old_sample, bank("old"), PROTOCOL)
    target = region(current)
    target["nearest_other_labeled_gt"] = {"category_id": 22, "iou": .9}
    result = compare_region(target, {"old": old, "new": current}, CATEGORIES, PROTOCOL)
    assert result["source_candidates"][0]["query_id"] == 0
    assert result["other_nearest_box"]["query_id"] == 1
    assert result["other_best_score_box"]["query_id"] == 1
    assert result["other_best_score_box"]["region_iou"] == 1
    assert result["new_minus_old_log_score_components"]["detector"] > 0
    assert result["new_minus_old_log_score_components"]["clip"] == 0
    assert result["new_minus_old_log_score_components"]["closure_abs_error"] < 1e-5
    assert result["source_candidates"][0]["overlapping_annotation_class"]["name"] == "other"
    json.dumps(result, allow_nan=False)


def test_no_overlapping_other_box_does_not_invent_a_region_match():
    current = replay_sample(sample(), bank(), PROTOCOL)
    other_sample = sample("old")
    other_sample["query_boxes"] += 100
    old = replay_sample(other_sample, bank("old"), PROTOCOL)
    result = compare_region(region(current), {"old": old, "new": current}, CATEGORIES, PROTOCOL)
    assert result["other_nearest_box"]["region_iou"] == 0
    assert result["other_best_score_box"] is None
    assert result["other_eligible_query_count"] == 0
    assert result["new_minus_old_log_score_components"] is None


def test_recovery_requires_saved_score_and_preserves_query_ambiguity():
    current_sample = sample()
    for key in ("features", "roi_features", "query_boxes"):
        current_sample[key] = current_sample[key][0:1].repeat(2, 1)
    current = replay_sample(current_sample, bank(), PROTOCOL)
    old = replay_sample(sample("old"), bank("old"), PROTOCOL)
    target = region(current)
    result = compare_region(target, {"old": old, "new": current}, CATEGORIES, PROTOCOL)
    assert result["source_query_identity"] == "ambiguous"
    assert len(result["source_candidates"]) == 2
    assert result["new_minus_old_log_score_components"] is None
    target["score"] += .01
    with pytest.raises(ValueError, match="does not match source replay"):
        compare_region(target, {"old": old, "new": current}, CATEGORIES, PROTOCOL)


def test_fp_overlap_keeps_same_box_detections_but_excludes_tp_controls():
    base = {"kind": "fp", "source_side": "new", "category": "koala", "image_id": 1,
            "box_xyxy": [0., 0., 10., 10.], "detection_id": 1}
    regions = [base, {**base, "detection_id": 2},
               {**base, "detection_id": 3, "box_xyxy": [5., 0., 15., 10.]},
               {**base, "detection_id": 4, "kind": "tp"}]
    group = fp_overlap_report(regions)[0]
    assert group["detection_ids"] == [1, 2, 3]
    assert group["iou_matrix"][0][1] == 1
    assert group["iou_matrix"][0][2] == pytest.approx(1 / 3)
    assert group["pairs_iou_ge_050"] == group["pairs_iou_ge_075"] == 1


def _pipeline_fixture(tmp_path):
    dataset = {"images": [{"id": 1}, {"id": 2}], "annotations": [], "categories": [
        {"id": 11, "name": "koala"}, {"id": 22, "name": "other"},
    ]}
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps(dataset))
    annotation_id = file_identity(annotation_path)
    inputs = {"schema_version": 1, "annotations_sha256": annotation_id["sha256"],
              "image_ids": [1, 2], "tpa_tau": .004375, "cls_tau": .07, **PROTOCOL}
    signature = fingerprint(inputs)
    parent_dir = tmp_path / "parent_cache"
    parent_dir.mkdir()
    (parent_dir / "manifest.json").write_text(json.dumps({"inputs": inputs, "fingerprint": signature}))
    for side in ("old", "new"):
        (parent_dir / side).mkdir()
        torch.save(bank(side, signature), parent_dir / side / "bank.pt")
        for image_id in (1, 2):
            torch.save(sample(side, signature, image_id), parent_dir / side / f"{image_id}.pt")
    replay = replay_sample(sample(), bank(), PROTOCOL)
    fp = {"image_id": 1, "detection_id": 99, "rank": 1, "bbox": [0., 0., 10., 10.],
          "score": float(replay["scores"][0, 0])}
    tp = {**fp, "image_id": 2, "detection_id": 199, "rank": 2, "matched_gt_id": 1001}
    details = {"max_dets": 3, "sources": {"annotations": annotation_id}, "classes": {"koala": {
        side: {"0.50": {"leading_false_positives": [fp] if side == "new" else [],
                         "true_positives": [tp]}} for side in ("old", "new")
    }}}
    details_path = tmp_path / "fp_details.json"
    details_path.write_text(json.dumps(details))
    return SimpleNamespace(
        fp_details=str(details_path), pairing_cache=str(parent_dir),
        output=str(tmp_path / "output" / "report.json"),
        categories=["koala"], match_iou=.5, cpu_threads=1, max_images=10,
        fill_missing=False, device="cpu", annotations=None,
    ), parent_dir, details


def test_cached_pipeline_never_runs_forward_and_preserves_parent(tmp_path, monkeypatch):
    args, parent, details = _pipeline_fixture(tmp_path)
    before = {path: path.read_bytes() for path in parent.rglob("*") if path.is_file()}
    def forbidden(*args, **kwargs):
        raise AssertionError("Complete cache must not forward")
    monkeypatch.setattr("tools.diagnose_rare_fp_regions.fill_missing_images", forbidden)
    report = run(args)
    assert report["complete"] and report["selected_image_ids"] == [1, 2]
    assert len(report["regions"]) == 3  # New FP and old/new TP controls.
    assert all(row["source_query_identity"] == "unique" for row in report["regions"])
    assert before == {path: path.read_bytes() for path in before}
    assert json.loads(open(args.output).read()) == report
    assert len(collect_regions(details, ["koala"])) == 3


def test_missing_cache_lists_exact_images_and_only_fills_them(tmp_path, monkeypatch):
    args, parent, _ = _pipeline_fixture(tmp_path)
    (parent / "old" / "1.pt").unlink()
    incomplete = run(args)
    assert not incomplete["complete"]
    assert incomplete["missing_images"] == {"old": [1], "new": []}
    calls = []
    def fill(args, parent_manifest, supplement, missing, dataset, annotation_path, cache_dir):
        calls.append(deepcopy(missing))
        (cache_dir / "old").mkdir(parents=True)
        signature = supplement["fingerprint"]
        torch.save(bank("old", signature), cache_dir / "old" / "bank.pt")
        torch.save(sample("old", signature, 1), cache_dir / "old" / "1.pt")
    monkeypatch.setattr("tools.diagnose_rare_fp_regions.fill_missing_images", fill)
    args.fill_missing = True
    complete = run(args)
    assert complete["complete"]
    assert calls == [{"old": [1], "new": []}]
    assert not (parent / "old" / "1.pt").exists()
    # Subsequent cached replay must reuse that supplemental image.
    args.fill_missing = False
    assert run(args)["complete"]
    assert len(calls) == 1


def test_native_fill_passes_only_missing_ids_to_existing_dump(tmp_path, monkeypatch):
    import sys
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"fixture")
    inputs = {"code_sha256": "same-code", "asset_sha256": {}, "seed": 42,
              "old_checkpoint": str(checkpoint), "new_checkpoint": str(checkpoint),
              "old_sha256": file_identity(checkpoint)["sha256"],
              "tpa_tau": .004375, "cls_tau": .07, **PROTOCOL}
    args = SimpleNamespace(config_file="config.py", old_checkpoint=None, new_checkpoint=None, dump_device="cpu")
    calls = []
    def dump(dump_args, side, panel, dataset, signature, cache_dir):
        calls.append((side, panel["image_ids"], signature))
        assert dump_args.beta == .3 and dump_args.max_dets == 3
    monkeypatch.setattr("tools.diagnose_rare_fp_regions.model_code_hash", lambda _: "same-code")
    monkeypatch.setitem(sys.modules, "tools.diagnose_detector_tpa_pairing", SimpleNamespace(dump_checkpoint=dump))
    fill_missing_images(args, {"inputs": inputs}, {"fingerprint": "extra"},
                        {"old": [7, 9], "new": []}, {}, tmp_path / "annotations.json", tmp_path / "cache")
    assert calls == [("old", [7, 9], "extra")]


def test_supplemental_bank_must_preserve_model_values():
    parent = bank("old", "parent")
    extra = bank("old", "extra")
    compare_banks(parent, extra)  # The separate cache fingerprint may differ.
    extra["prototypes"][0, 0, 0] += .01
    with pytest.raises(ValueError, match="Supplemental bank changed: prototypes"):
        compare_banks(parent, extra)


def test_nonfinite_native_replay_check_is_rejected():
    invalid = sample()
    invalid["native_replay_check"]["score_max_abs_error"] = float("nan")
    with pytest.raises(ValueError, match="native classifier/fusion replay"):
        validate_sample(invalid, bank(), "test", "new", 1)


def test_output_cannot_overwrite_source_or_parent_cache(tmp_path):
    args, parent, _ = _pipeline_fixture(tmp_path)
    for target in (args.fp_details, str(parent / "manifest.json")):
        args.output = target
        with pytest.raises(ValueError, match="Output must not overwrite"):
            run(args)
