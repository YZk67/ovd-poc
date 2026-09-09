import torch
import torch.nn.functional as F

from lami_dino.diagnostic_ops import (
    detection_stage_hits,
    fuse_detector_vlm_scores,
    fuse_sparse_detector_vlm_scores,
    prototype_variant_logits,
    sparse_fusion_candidate_pairs,
    true_class_mode_weights,
)


def test_power_fusion_matches_probability_formula():
    detector = torch.tensor([[0.2, -0.7, 1.1, -0.1]])
    vlm = torch.tensor([[1.0, 0.5, -0.5, 0.0]])
    novel = torch.tensor([False, False, True, True])

    actual = fuse_detector_vlm_scores(
        detector,
        vlm,
        novel,
        fusion="power",
        base_weight=0.2,
        novel_weight=0.3,
        novel_scale=3.0,
    ).exp()

    detector_probability = detector.sigmoid()
    vlm_probability = vlm.softmax(dim=-1)
    weights = torch.tensor([0.2, 0.2, 0.3, 0.3])
    expected = detector_probability ** (1.0 - weights) * vlm_probability ** weights
    expected[:, novel] *= 3.0
    torch.testing.assert_close(actual, expected)


def test_sparse_pool_contains_every_profiles_top_pairs():
    torch.manual_seed(0)
    detector = torch.randn(7, 11)
    vlm = torch.randn(7, 11)
    novel = torch.arange(11) >= 6
    profiles = [
        {
            "fusion": "power",
            "base_weight": 0.0,
            "novel_weight": 0.3,
            "novel_scale": 3.0,
        },
        {
            "fusion": "logprob_add",
            "base_weight": 0.1,
            "novel_weight": 0.5,
            "novel_scale": 3.0,
        },
    ]
    queries, classes = sparse_fusion_candidate_pairs(
        detector, vlm, novel, topk=9, profiles=profiles
    )
    cached = set(zip(queries.tolist(), classes.tolist()))

    for profile in profiles:
        scores = fuse_detector_vlm_scores(detector, vlm, novel, **profile)
        top = scores.flatten().topk(9).indices
        expected = set(zip((top // 11).tolist(), (top % 11).tolist()))
        assert expected <= cached


def test_sparse_replay_matches_dense_fusion_on_cached_pairs():
    torch.manual_seed(2)
    detector = torch.randn(5, 7)
    vlm = torch.randn(5, 7)
    novel = torch.arange(7) >= 4
    profile = {
        "fusion": "logprob_add",
        "base_weight": 0.1,
        "novel_weight": 0.5,
        "novel_scale": 3.0,
    }
    queries, classes = sparse_fusion_candidate_pairs(
        detector, vlm, novel, topk=11, profiles=[profile]
    )
    dense = fuse_detector_vlm_scores(detector, vlm, novel, **profile)
    sparse = fuse_sparse_detector_vlm_scores(
        detector[queries, classes],
        vlm[queries, classes],
        torch.logsumexp(vlm, dim=-1),
        queries,
        classes,
        novel,
        **profile,
    )
    torch.testing.assert_close(sparse, dense[queries, classes])


def test_sparse_pool_can_union_different_detector_sources():
    torch.manual_seed(4)
    detector_sources = {
        "multi": torch.randn(5, 7),
        "mean": torch.randn(5, 7),
    }
    vlm = torch.randn(5, 7)
    novel = torch.arange(7) >= 4
    profile = {
        "fusion": "power",
        "base_weight": 0.0,
        "novel_weight": 0.3,
        "novel_scale": 3.0,
    }

    pairs = set()
    for detector in detector_sources.values():
        queries, classes = sparse_fusion_candidate_pairs(
            detector, vlm, novel, topk=8, profiles=[profile]
        )
        pairs.update(zip(queries.tolist(), classes.tolist()))

    for detector in detector_sources.values():
        scores = fuse_detector_vlm_scores(detector, vlm, novel, **profile)
        top = scores.flatten().topk(8).indices
        expected = set(zip((top // 7).tolist(), (top % 7).tolist()))
        assert expected <= pairs


def test_duplicate_prototypes_make_lme_and_prototype_mean_identical():
    torch.manual_seed(1)
    features = torch.randn(5, 8)
    directions = F.normalize(torch.randn(4, 8), dim=-1)
    prototypes = directions[:, None].expand(-1, 5, -1).contiguous()
    prompts = directions[:, None].expand(-1, 8, -1).contiguous()

    logits = prototype_variant_logits(
        features,
        prototypes,
        prompts,
        temperature=0.07,
        logit_scale=50.0,
    )
    torch.testing.assert_close(logits["logmeanexp"], logits["prototype_mean"])
    torch.testing.assert_close(logits["logmeanexp"], logits["prompt_mean"])


def test_mode_weights_select_matching_true_class_mode():
    prototypes = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[-1.0, 0.0], [0.0, -1.0]],
        ]
    )
    features = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    classes = torch.tensor([0, 1])
    weights = true_class_mode_weights(
        features, prototypes, classes, temperature=0.01
    )

    assert weights[0].argmax().item() == 0
    assert weights[1].argmax().item() == 1
    assert torch.all(weights.max(dim=-1).values > 0.99)


def test_detection_stage_hits_separates_proposals_best_pair_and_any_pair():
    gt_boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]
    )
    gt_classes = torch.tensor([0, 1])
    query_boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
            [21.0, 21.0, 29.0, 29.0],
        ]
    )
    # The second GT has a correct-class selected query at IoU=.64, but its
    # best-IoU query/class pair is not selected.
    selected_queries = torch.tensor([0, 2])
    selected_classes = torch.tensor([0, 1])

    best_iou, stages = detection_stage_hits(
        query_boxes,
        gt_boxes,
        gt_classes,
        selected_queries,
        selected_classes,
        thresholds=[0.5, 0.75],
    )

    torch.testing.assert_close(best_iou, torch.ones(2))
    assert stages[0.5]["proposal"].tolist() == [True, True]
    assert stages[0.5]["best_query_pair"].tolist() == [True, False]
    assert stages[0.5]["class_aware_topk"].tolist() == [True, True]
    assert stages[0.75]["class_aware_topk"].tolist() == [True, False]
