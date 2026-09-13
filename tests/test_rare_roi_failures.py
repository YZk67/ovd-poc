import pytest

from tools.analyze_rare_roi_failures import (
    analyze,
    inferred_detector_probability,
    print_report,
    semantic_partition,
)


def test_invert_actual_rare_power_fusion():
    detector_probability = 0.4
    clip_probability = 0.01
    beta = 0.3
    scale = 3.0
    fused = scale * detector_probability ** (1 - beta) * clip_probability ** beta
    row = {
        "fused_true_score": fused,
        "fused_query_clip_true_probability": clip_probability,
    }
    assert inferred_detector_probability(row, beta=beta, novel_scale=scale) == pytest.approx(0.4)
    row["fused_query_clip_true_probability"] = 0.0
    assert inferred_detector_probability(row, beta=beta, novel_scale=scale) is None


@pytest.mark.parametrize(
    "gt_rank,query_rank,expected",
    [
        (1, 2, "both_topk"),
        (2, 8, "gt_only_topk"),
        (8, 2, "query_only_topk"),
        (8, 10, "neither_topk"),
    ],
)
def test_semantic_partition(gt_rank, query_rank, expected):
    assert semantic_partition({"gt_rank": gt_rank, "clip_best_query_rank": query_rank}) == expected


def make_row(image_id, status, gt_rank, query_rank=2, score=0.1):
    row = {
        "image_id": image_id,
        "gt_index": 0,
        "category_id": 7,
        "status": status,
        "gt_xyxy": [0, 0, 40, 40],
        "gt_rank": gt_rank,
        "gt_true_probability": 0.1,
    }
    if status != "no_eligible_box":
        row.update(
            {
                "clip_best_query_rank": query_rank,
                "fused_query_clip_rank": query_rank,
                "fused_query_clip_true_probability": 0.1,
                "fused_query_clip_top1_category_id": 8,
                "fused_query_iou": 0.8,
                "fused_true_score": score,
                "image_topk_threshold": 0.2,
            }
        )
    return row


def test_analysis_counts_paired_roi_states_and_thresholds(capsys):
    rows = [
        make_row(1, "current_miss", 2, 8, score=0.18),
        make_row(2, "current_miss", 9, 8, score=0.04),
        make_row(3, "current_hit", 1, 1, score=0.3),
        make_row(4, "no_eligible_box", 12),
    ]
    report = analyze(
        rows,
        {"beta": 0.3, "novel_scale": 3.0},
        seen_class_ids={8},
    )
    assert report["status_counts"] == {
        "current_hit": 1,
        "current_miss": 2,
        "no_eligible_box": 1,
    }
    assert report["miss_semantic_partition"]["gt_only_topk"] == 1
    assert report["miss_semantic_partition"]["neither_topk"] == 1
    assert report["miss_near_top300_threshold_ratio_ge_0p8"]["count"] == 1
    assert report["by_area"]["medium"]["miss_fraction"] == 0.5
    assert report["detector_inversion_unavailable"] == 0
    assert report["wrong_clip_top1_seen_fraction"] == 1.0
    category = report["categories_with_misses"][0]
    assert category["hit_count"] == 1
    assert category["miss_fused_best_components"]["count"] == 2
    assert category["miss_fused_best_components"]["near_top300_threshold_count"] == 1
    assert category["hit_fused_best_components"]["score_over_top300_threshold"]["median"] == pytest.approx(1.5)
    assert category["miss_semantic_partition"] == {
        "gt_only_topk": 1,
        "neither_topk": 1,
    }
    print_report(report)
    assert "Rare ROI-path failure analysis" in capsys.readouterr().out


def test_miss_above_top300_threshold_is_rejected():
    row = make_row(1, "current_miss", 8, score=0.3)
    with pytest.raises(ValueError, match="exceed the stored top-300 threshold"):
        analyze([row], {"beta": 0.3, "novel_scale": 3.0})
