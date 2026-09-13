import json

import pytest
import torch
import torch.nn.functional as F

from lami_dino.pairing_diagnostic_ops import analyze_image, replay_classifier
from lami_dino.prototype_ops import calibrated_logmeanexp_similarity


def _sample(**overrides):
    values = dict(
        gt_boxes=torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        gt_classes=torch.tensor([0]),
        gt_ids=[71],
        query_boxes=torch.tensor([[0.0, 0.0, 9.0, 10.0], [0.0, 0.0, 10.0, 10.0]]),
        variant_logits={
            "native": torch.tensor([[3.0, 1.0, 0.0], [2.0, 1.0, 0.0]]),
            "swap": torch.tensor([[-3.0, 1.0, 0.0], [4.0, 1.0, 0.0]]),
        },
        vlm_logits=torch.tensor([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]),
        novel_mask=torch.tensor([True, False, False]),
        native_variant="native",
        alpha=0.0,
        beta=0.0,
        novel_scale=1.0,
        max_dets=1,
        iou_thresholds=[0.5],
    )
    values.update(overrides)
    return values


def test_replay_matches_native_equation_and_chunk_sizes():
    generator = torch.Generator().manual_seed(12)
    features = torch.randn(11, 8, generator=generator)
    prototypes = torch.randn(4, 5, 8, generator=generator)
    expected = calibrated_logmeanexp_similarity(
        F.normalize(features, dim=-1), F.normalize(prototypes, dim=-1),
        temperature=0.13, logit_scale=17.0,
    ) - 4.6
    for chunk_size in (1, 3, 128):
        actual = replay_classifier(
            features, prototypes, temperature=0.13, logit_scale=17.0,
            cls_bias=-4.6, query_chunk_size=chunk_size,
        )
        torch.testing.assert_close(actual, expected)


def test_pmean_duplicate_invariance_and_mean_before_normalization():
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    bank = torch.tensor([[[4.0, 0.0], [0.0, 1.0]], [[0.0, -2.0], [-3.0, 0.0]]])
    mean = bank.mean(1, keepdim=True)
    singleton = replay_classifier(features, mean)
    duplicate = replay_classifier(features, mean.expand(-1, 5, -1))
    torch.testing.assert_close(singleton, duplicate)
    expected = 50.0 * features @ F.normalize(mean[:, 0], dim=-1).T
    torch.testing.assert_close(singleton, expected)
    normalized_first = replay_classifier(features, F.normalize(bank, dim=-1).mean(1, keepdim=True))
    assert not torch.allclose(singleton, normalized_first)


def test_native_query_is_fixed_and_geometry_and_oracle_are_separate():
    output = analyze_image(**_sample())
    native, swap = output["rows"]
    assert native["gt_id"] == swap["gt_id"] == 71
    assert native["query_id"] == swap["query_id"] == 0
    assert native["detector_rank"] == 1 and native["margin"] == 2.0
    assert swap["detector_rank"] == 3 and swap["margin"] == -4.0
    assert native["geometry_anchor"]["query_id"] == swap["geometry_anchor"]["query_id"] == 1
    assert swap["geometry_anchor"]["detector_rank"] == 1
    assert native["oracle"]["query_id"] == 0
    assert swap["oracle"]["query_id"] == 1
    assert native["in_topk"] and not swap["in_topk"]
    assert swap["pair_topk"]  # Coverage can recover while fixed-query scoring worsens.
    assert swap["threshold_ratio"] < 1.0
    json.dumps(output, allow_nan=False)


def test_fusion_and_global_topk_include_base_classes():
    values = _sample(alpha=0.2, beta=0.7, novel_scale=3.0, max_dets=3)
    values["variant_logits"] = {"native": torch.tensor([[-4.0, 6.0, 5.0], [-3.0, 7.0, 4.0]])}
    output = analyze_image(**values)
    probability = values["variant_logits"]["native"].sigmoid()
    clip = values["vlm_logits"].softmax(-1)
    expected = probability.clone()
    expected[:, 1:] = probability[:, 1:] ** 0.8 * clip[:, 1:] ** 0.2
    expected[:, 0] = probability[:, 0] ** 0.3 * clip[:, 0] ** 0.7 * 3.0
    scores, indices = expected.flatten().topk(3)
    predictions = output["top_predictions"]["native"]
    assert [p["query_id"] * 3 + p["class_index"] for p in predictions] == indices.tolist()
    torch.testing.assert_close(torch.tensor([p["score"] for p in predictions]), scores)
    assert any(p["class_index"] != 0 for p in predictions)
    assert output["topk_thresholds"]["native"] == pytest.approx(float(scores[-1]))


def test_topk_ties_use_actual_indices_not_cutoff_comparison():
    values = _sample(
        variant_logits={"native": torch.zeros(2, 3)},
        vlm_logits=torch.zeros(2, 3),
        max_dets=1,
    )
    output = analyze_image(**values)
    chosen = torch.full((6,), 0.5).topk(1).indices.tolist()
    prediction = output["top_predictions"]["native"][0]
    assert prediction["query_id"] * 3 + prediction["class_index"] == chosen[0]
    row = output["rows"][0]
    assert row["threshold_ratio"] == 1.0
    assert row["in_topk"] == (row["query_id"] * 3 in chosen)
    assert row["pair_topk"] == any(index in chosen for index in (0, 3))


def test_query_and_gt_permutations_preserve_paired_results():
    values = _sample(
        gt_boxes=torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
        gt_classes=torch.tensor([0, 2]), gt_ids=[71, 83],
    )
    original = analyze_image(**values)
    permutation = torch.tensor([1, 0])
    values["query_boxes"] = values["query_boxes"][permutation]
    values["vlm_logits"] = values["vlm_logits"][permutation]
    values["variant_logits"] = {name: logits[permutation] for name, logits in values["variant_logits"].items()}
    values["gt_boxes"] = values["gt_boxes"].flip(0)
    values["gt_classes"] = values["gt_classes"].flip(0)
    values["gt_ids"] = values["gt_ids"][::-1]
    permuted = analyze_image(**values)
    indexed = {(row["gt_id"], row["variant"]): row for row in permuted["rows"]}
    for row in original["rows"]:
        other = indexed[(row["gt_id"], row["variant"])]
        for key in ("eligible", "best_iou", "iou", "detector_rank", "margin", "clip_rank", "fused_score", "pair_topk"):
            assert row[key] == other[key]
        if row["query_id"] is not None:
            assert row["query_id"] == int(permutation[other["query_id"]])


def test_iou_thresholds_change_eligibility_without_dropping_gt():
    output = analyze_image(**_sample(iou_thresholds=[0.5, 0.95]))
    assert len(output["rows"]) == 4
    assert [row["query_id"] for row in output["rows"]] == [0, 0, 1, 1]
    missing = analyze_image(**_sample(gt_boxes=torch.tensor([[20.0, 20.0, 30.0, 30.0]])))
    for row in missing["rows"]:
        assert not row["eligible"] and row["num_eligible"] == 0
        assert row["query_id"] is None and row["detector_rank"] is None
        assert not row["pair_topk"] and row["oracle"] is None
        assert row["best_iou"] == 0.0
        assert row["geometry_anchor"] is not None
        assert not row["geometry_anchor"]["eligible"]


def test_empty_queries_gt_and_zero_topk_are_json_safe():
    output = analyze_image(**_sample(
        query_boxes=torch.empty(0, 4), variant_logits={"native": torch.empty(0, 3)},
        vlm_logits=torch.empty(0, 3),
    ))
    assert output["top_predictions"] == {"native": []}
    assert output["topk_thresholds"] == {"native": None}
    assert output["rows"][0]["geometry_anchor"] is None
    assert output["rows"][0]["best_iou"] is None
    json.dumps(output, allow_nan=False)
    output = analyze_image(**_sample(gt_boxes=torch.empty(0, 4), gt_classes=torch.empty(0, dtype=torch.long), gt_ids=[]))
    assert output["rows"] == [] and len(output["top_predictions"]["native"]) == 1
    output = analyze_image(**_sample(max_dets=0))
    assert output["rows"][0]["threshold_ratio"] is None
    assert not output["rows"][0]["pair_topk"]
    assert replay_classifier(torch.empty(0, 2), torch.ones(3, 5, 2)).shape == (0, 3)


@pytest.mark.parametrize("overrides", [
    {"vlm_logits": torch.full((2, 3), float("nan"))},
    {"variant_logits": {"native": torch.zeros(2, 4)}},
    {"gt_classes": torch.tensor([3])},
    {"gt_classes": torch.tensor([0.0])},
    {"gt_ids": []},
    {"novel_mask": torch.tensor([1, 0, 0])},
    {"alpha": float("nan")},
    {"beta": 1.1},
    {"max_dets": -1},
    {"iou_thresholds": [0.5, 0.5]},
    {"iou_thresholds": [float("inf")]},
    {"query_boxes": torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])},
])
def test_analyze_rejects_invalid_inputs(overrides):
    with pytest.raises(ValueError):
        analyze_image(**_sample(**overrides))


@pytest.mark.parametrize("overrides", [
    {"features": torch.full((1, 2), float("inf"))},
    {"prototypes": torch.ones(3, 0, 2)},
    {"prototypes": torch.ones(3, 1, 4)},
    {"temperature": 0.0},
    {"cls_bias": float("nan")},
    {"query_chunk_size": 0},
])
def test_replay_rejects_invalid_inputs(overrides):
    values = {"features": torch.ones(1, 2), "prototypes": torch.ones(3, 1, 2)}
    values.update(overrides)
    with pytest.raises(ValueError):
        replay_classifier(**values)
