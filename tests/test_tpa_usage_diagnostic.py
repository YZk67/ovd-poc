from collections import Counter, defaultdict

import pytest
import torch

from tools.diagnose_tpa_usage import (
    complementarity_verdict,
    component_branch_verdict,
    finalize_complementarity_accumulator,
    finalize_component_accumulator,
    finalize_ranking_stats,
    finalize_rare_recall_accumulator,
    finalize_selection_accumulator,
    greedy_gt_topk_matches,
    make_complementarity_accumulator,
    make_component_accumulator,
    make_selection_accumulator,
    select_dataset_records,
    semantic_best_queries,
    true_class_topk_hits,
    update_complementarity_accumulator,
    update_component_accumulator,
    update_ranking_stats,
    update_rare_recall_accumulator,
    update_selection_accumulator,
)


def test_fused_rank_and_global_selection_partition():
    logits = torch.tensor(
        [
            [9.0, 1.0, 0.0],
            [3.0, 4.0, 5.0],
            [1.0, 3.0, 2.0],
            [4.0, 3.0, 2.0],
        ]
    )
    classes = torch.tensor([0, 0, 2, 2])
    frequencies = ["r"] * 4
    ranking = defaultdict(lambda: defaultdict(float))
    ranks = update_ranking_stats(ranking, logits, classes, frequencies)
    assert ranks.tolist() == [1, 3, 2, 3]
    finalized_ranking = finalize_ranking_stats(ranking)
    assert finalized_ranking["r"]["top1"] == 0.25
    assert finalized_ranking["r"]["top5"] == 1.0
    assert finalized_ranking["r"]["median_rank"] == 2.5
    assert finalized_ranking["r"]["p90_rank"] == 3.0

    selection = make_selection_accumulator()
    update_selection_accumulator(
        selection,
        torch.tensor([1, 7, 3, 9]),
        torch.tensor([True, False, False, True]),
        frequencies,
        cutoff=5,
    )
    report = finalize_selection_accumulator(selection, cutoff=5)["r"]
    assert report["global_topk_recall"] == 0.5
    assert report["miss_due_to_category_rank_fraction"] == 0.5
    assert report["miss_despite_rank_cutoff_fraction"] == 0.5


def test_rare_sampling_keeps_every_rare_image_when_num_images_is_zero():
    records = [
        {"image_id": 1, "annotations": [{"category_id": 0}]},
        {"image_id": 2, "annotations": [{"category_id": 1}]},
        {
            "image_id": 3,
            "annotations": [{"category_id": 0}, {"category_id": 1}],
        },
    ]
    selected = select_dataset_records(
        records,
        num_images=0,
        seed=0,
        sampling="rare",
        frequencies=["r", "f"],
    )
    assert [record["image_id"] for record in selected] == [1, 3]


def test_component_report_separates_hit_and_miss_branch_scores():
    accumulator = make_component_accumulator()
    update_component_accumulator(
        accumulator,
        detector_probabilities=torch.tensor([0.8, 0.2]),
        vlm_probabilities=torch.tensor([0.4, 0.1]),
        fused_scores=torch.tensor([0.5, 0.05]),
        topk_threshold=0.1,
        selected_ious=torch.tensor([0.8, 0.6]),
        category_ranks=torch.tensor([1, 10]),
        survives=torch.tensor([True, False]),
        frequencies=["r", "r"],
    )
    report = finalize_component_accumulator(accumulator)
    assert report["r"]["hit"]["detector_probability"]["median"] == pytest.approx(0.8)
    assert report["r"]["miss"]["detector_probability"]["median"] == pytest.approx(0.2)
    assert report["r"]["hit"]["log_score_margin"]["median"] == pytest.approx(
        torch.log(torch.tensor(5.0)).item()
    )

    verdict = component_branch_verdict(
        report, detector_weight=0.7, vlm_weight=0.3
    )
    assert verdict["verdict"] == "DETECTOR_COMPONENT_DOMINANT"
    assert verdict["detector_weighted_log_separation"] > verdict[
        "vlm_weighted_log_separation"
    ]


def test_semantic_best_uses_highest_true_score_among_iou_valid_queries():
    overlaps = torch.tensor(
        [
            [0.9, 0.7, 0.2],
            [0.1, 0.4, 0.3],
        ]
    )
    classes = torch.tensor([1, 0])
    scores = torch.tensor(
        [
            [0.1, 0.2],
            [0.2, 0.8],
            [0.9, 0.1],
        ]
    )
    valid, queries, selected_ious = semantic_best_queries(
        overlaps, classes, scores, min_iou=0.5
    )
    assert valid.tolist() == [True, False]
    # Query 0 has the best IoU, but query 1 has the strongest true-class score.
    assert queries.tolist() == [1]
    assert selected_ious.tolist() == pytest.approx([0.7])


def test_detector_and_clip_select_their_own_best_eligible_query():
    overlaps = torch.tensor([[0.8, 0.7, 0.2]])
    classes = torch.tensor([1])
    detector = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.0, 1.0]])
    clip = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.0, 1.0]])
    det_valid, det_queries, _ = semantic_best_queries(
        overlaps, classes, detector, min_iou=0.5
    )
    clip_valid, clip_queries, _ = semantic_best_queries(
        overlaps, classes, clip, min_iou=0.5
    )
    assert det_valid.tolist() == clip_valid.tolist() == [True]
    assert det_queries.tolist() == [0]
    assert clip_queries.tolist() == [1]


def test_rare_failure_partition_distinguishes_topk_from_one_to_one():
    accumulator = Counter()
    update_rare_recall_accumulator(
        accumulator,
        frequencies=["r", "r", "r", "r", "f"],
        eligible_box=torch.tensor([False, True, True, True, True]),
        pair_in_topk=torch.tensor([False, False, True, True, True]),
        one_to_one_hit=torch.tensor([False, False, False, True, True]),
    )
    report = finalize_rare_recall_accumulator(
        accumulator, min_iou=0.5, image_topk=300
    )
    assert report["rare_gt"] == 4
    assert report["no_eligible_box"] == 1
    assert report["eligible_but_pair_below_topk"] == 1
    assert report["pair_in_topk_but_one_to_one_miss"] == 1
    assert report["one_to_one_hit"] == 1
    with pytest.raises(ValueError, match="not nested"):
        update_rare_recall_accumulator(
            accumulator,
            frequencies=["r"],
            eligible_box=torch.tensor([False]),
            pair_in_topk=torch.tensor([True]),
            one_to_one_hit=torch.tensor([False]),
        )


def test_true_class_topk_hits_and_complementarity_partition_actual_misses():
    eligible = torch.tensor(
        [
            [True, True, False],
            [False, True, True],
            [True, False, False],
            [False, False, False],
        ]
    )
    classes = torch.tensor([1, 0, 1, 0])
    # Flat pair ids use query * C + class with C=2.
    current_hits = true_class_topk_hits(
        eligible, classes, torch.tensor([3]), num_classes=2
    )
    detector_hits = true_class_topk_hits(
        eligible, classes, torch.tensor([1]), num_classes=2
    )
    vlm_hits = true_class_topk_hits(
        eligible, classes, torch.tensor([2]), num_classes=2
    )
    assert current_hits.tolist() == [True, False, False, False]
    assert detector_hits.tolist() == [True, False, True, False]
    assert vlm_hits.tolist() == [False, True, False, False]

    accumulator = make_complementarity_accumulator()
    update_complementarity_accumulator(
        accumulator,
        valid=eligible.any(dim=1),
        current_hits=current_hits,
        detector_hits=detector_hits,
        vlm_hits=vlm_hits,
        frequencies=["r", "r", "r", "r"],
    )
    report = finalize_complementarity_accumulator(accumulator)
    assert report["r"]["proposal_valid"] == 3
    assert report["r"]["current_hit"] == 1
    assert report["r"]["vlm_only"] == 1
    assert report["r"]["detector_only"] == 1
    assert report["r"]["neither"] == 0
    assert complementarity_verdict(report) == "COMPLEMENTARY_COMPONENTS"


def test_greedy_topk_matching_is_one_to_one_and_score_ordered():
    overlaps = torch.tensor(
        [
            [0.9, 0.8],
            [0.85, 0.7],
        ]
    )
    classes = torch.tensor([0, 0])
    # q0/class0 is higher scored than q1/class0.
    hits, matched_queries = greedy_gt_topk_matches(
        overlaps,
        classes,
        torch.tensor([0, 2]),
        num_classes=2,
        min_iou=0.5,
    )
    assert hits.tolist() == [True, True]
    assert matched_queries.tolist() == [0, 1]

    one_hit, _ = greedy_gt_topk_matches(
        overlaps,
        classes,
        torch.tensor([0]),
        num_classes=2,
        min_iou=0.5,
    )
    assert one_hit.tolist() == [True, False]
