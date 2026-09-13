from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from tools.diagnose_detector_tpa_pairing import (
    VARIANTS, analyze_cache, check_replay, locked_manifest, save_tensor_file,
    summarize_pair_rows, tensor_digest, to_predictions,
    validate_existing_bank, validate_parameter_state,
)


def make_rows(gt_id, class_index, *, old=True, new=True, rank=1, threshold=0.5):
    rows = []
    for label, variants in VARIANTS.items():
        eligible = old if label == "old" else new
        for variant in variants:
            rows.append({"image_id": 1, "gt_id": gt_id, "class_index": class_index,
                         "iou_threshold": threshold, "variant": variant, "eligible": eligible,
                         "detector_rank": rank if eligible else None, "margin": 1.0 if eligible else None,
                         "geometry_anchor": {"detector_rank": rank, "margin": 1.0},
                         "pair_topk": eligible, "threshold_ratio": 1.0 if eligible else None})
    return rows


def test_pair_summary_equal_class_not_instance_weighting_and_exclusions():
    rows = make_rows(1, 0, rank=1) + make_rows(2, 0, rank=1) + make_rows(3, 1, rank=10)
    rows += make_rows(4, 2, old=False) + make_rows(5, 3, new=False) + make_rows(6, 4, old=False, new=False)
    data = summarize_pair_rows(rows)["0.50"]
    assert data["localization_partition"] == {"all_gt": 6, "both_eligible": 3, "old_only_eligible": 1,
                                               "new_only_eligible": 1, "neither_eligible": 1}
    assert data["all_gt_classes"] == 5
    assert data["variants"]["new_new"]["paired_classes"] == 2
    assert data["variants"]["new_new"]["macro"]["native_anchor_top1"] == 0.5
    assert data["per_class_localization"]["4"]["both_eligible"] == 0
    assert data["terminal_swap_macro_deltas"]["new_old_minus_new_new"]["native_anchor_top1"] == 0


def test_pair_summary_handles_none_ratio_and_distinct_thresholds():
    rows = make_rows(1, 0, threshold=0.501) + make_rows(1, 0, threshold=0.504)
    for row in rows:
        row["threshold_ratio"] = None
    report = summarize_pair_rows(rows)
    assert set(report) == {"0.501", "0.504"}
    per_class = report["0.501"]["variants"]["new_new"]["per_class"][0]
    assert per_class["native_anchor_threshold_ratio"] is None
    assert per_class["threshold_ratio_valid_gt"] == 0
    json.dumps(report, allow_nan=False)


def test_pair_summary_does_not_drop_no_intersection_case():
    report = summarize_pair_rows(make_rows(1, 0, old=False))["0.50"]
    assert report["localization_partition"]["new_only_eligible"] == 1
    assert report["variants"]["new_new"]["paired_gt"] == 0
    assert report["variants"]["new_new"]["macro"]["native_anchor_top1"] is None
    with pytest.raises(ValueError, match="do not pair"):
        summarize_pair_rows(make_rows(1, 0)[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_pair_rows(make_rows(1, 0) * 2)


def test_native_replay_rejects_errors_and_tolerates_boundary_roundoff():
    logits = torch.tensor([[1.0, 2.0]])
    scores = torch.tensor([[0.5000001, 0.5]])
    replay_scores = scores.flip(-1)
    assert check_replay(logits, scores, logits.clone(), replay_scores, max_dets=1)["topk_symmetric_difference"] == 2
    with pytest.raises(ValueError, match="replay failed"):
        check_replay(logits, scores, logits + 1, scores, max_dets=1)
    with pytest.raises(ValueError, match="Non-finite"):
        check_replay(logits * float("nan"), scores, logits, scores, max_dets=1)


def test_checkpoint_shape_validation_prevents_loader_silent_skip():
    model = torch.nn.Linear(3, 2)
    validate_parameter_state(model, model.state_dict())
    with pytest.raises(ValueError, match="misses"):
        validate_parameter_state(model, {})
    wrong = model.state_dict()
    wrong["weight"] = torch.zeros(3, 3)
    with pytest.raises(ValueError, match="shapes/types"):
        validate_parameter_state(model, wrong)


def test_manifest_and_partial_bank_are_locked(tmp_path):
    path = tmp_path / "manifest.json"
    first = locked_manifest(path, {"version": 1})
    assert locked_manifest(path, {"version": 1}) == first
    with pytest.raises(ValueError, match="differs"):
        locked_manifest(path, {"version": 2})
    bank = {"prototypes": torch.ones(2, 3, 4), "prompt_hash": "original"}
    bank_path = tmp_path / "bank.pt"
    save_tensor_file(bank_path, bank)
    validate_existing_bank(bank_path, bank)
    changed = deepcopy(bank)
    changed["prototypes"][0, 0, 0] = 2
    with pytest.raises(ValueError, match="bank changed"):
        validate_existing_bank(bank_path, changed)
    assert tensor_digest(torch.ones(2, 3)) != tensor_digest(torch.ones(3, 2))


def test_predictions_keep_category_mapping_and_drop_empty_without_refill():
    top = [
        {"class_index": 1, "score": 0.8, "box_xyxy": [3, 4, 13, 24]},
        {"class_index": 0, "score": 0.7, "box_xyxy": [2, 4, 2, 8]},
    ]
    assert to_predictions(top, 99, [10, 20]) == [{"image_id": 99, "category_id": 20,
                                               "bbox": [3, 4, 10, 20], "score": 0.8}]


@pytest.mark.parametrize("official", [False, True])
def test_cached_end_to_end_replay_without_detectron2(tmp_path, monkeypatch, official):
    """Exercise bank loading, all five combinations, GT pairing and FP handoff."""
    if official:
        pytest.importorskip("lvis")
    dataset = {"images": [{"id": 1, "width": 100, "height": 100, "neg_category_ids": [], "not_exhaustive_category_ids": []}],
               "annotations": [{"id": 11, "image_id": 1, "category_id": 10, "bbox": [0, 0, 10, 10], "area": 100}],
               "categories": [{"id": 10, "frequency": "r"}, {"id": 20, "frequency": "f"}]}
    panel = {"image_ids": [1], "rare_category_ids": [10]}
    cache_dir = tmp_path / "cache"
    for label in VARIANTS:
        prototypes = torch.eye(2).unsqueeze(1).repeat(1, 2, 1)
        if label == "new":
            prototypes = prototypes.flip(0)
        bank = {"fingerprint": "test", "label": label, "iteration": 1,
                "prototypes": prototypes, "vlm_text": torch.eye(2), "prompt_sha256": "same",
                "novel_mask": torch.tensor([True, False]), "category_ids": [10, 20],
                "temperature": 0.07, "logit_scale": 5.0, "cls_bias": -3.0,
                "vlm_temperature": 1.0, "slot_prior_strength": 0.0,
                "prototype_mode_strength": 0.0, "tpa_tau": 0.004375}
        sample = {"fingerprint": "test", "label": label, "image_id": 1,
                  "features": torch.eye(2), "roi_features": torch.eye(2),
                  "query_boxes": torch.tensor([[0., 0., 10., 10.], [50., 50., 60., 60.]]),
                  "native_replay_check": {"logit_max_abs_error": 0., "score_max_abs_error": 0., "topk_symmetric_difference": 0}}
        save_tensor_file(cache_dir / label / "bank.pt", bank)
        save_tensor_file(cache_dir / label / "1.pt", sample)
    calls = []
    def mock_lvis(source, image_ids, predictions, **kwargs):
        calls.append(predictions)
        assert image_ids == [1] and kwargs["max_dets"] == 2
        assert len(predictions) == 2
        return {"summary_by_iou": {"0.50": {"true_positives": 1, "false_positives": 1}}}
    if not official:
        monkeypatch.setattr("tools.diagnose_detector_tpa_pairing.evaluate_panel_predictions", mock_lvis)
    args = SimpleNamespace(device="cpu", alpha=0., beta=.3, novel_scale=3., query_chunk_size=2,
                           max_dets=2, iou_thresholds=[.5], log_interval=5, output=str(tmp_path / "report.json"))
    analyze_cache(args, panel, dataset, "test", cache_dir)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["complete"] and len(report["panel_pr"]) == 5
    assert len(report["rows"]) == 5
    assert report["paired"]["0.50"]["localization_partition"]["both_eligible"] == 1
    if not official:
        assert len(calls) == 5
        assert report["panel_pr_deltas"]["new_old_minus_new_new"]["0.50"]["false_positives"] == 0
    else:
        assert report["panel_pr_deltas"]["new_old_minus_new_new"]["0.50"] == {"true_positives": 1, "false_positives": -1}
