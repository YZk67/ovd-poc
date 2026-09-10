import math

import torch

from tools.analyze_ovd_clip_confidence_gates import (
    build_clip_gate_specs,
    clip_confidence_gate_selection,
)


def make_payload():
    return {
        "candidate_query_ids": torch.tensor([0, 1], dtype=torch.int16),
        "candidate_class_ids": torch.tensor([0, 0], dtype=torch.int16),
        "component_query_summary": {
            "vlm_top_probabilities": torch.tensor(
                [[0.90, 0.05], [0.96, 0.02], [0.85, 0.10]]
            ),
            "vlm_top_classes": torch.tensor(
                [[1, 0], [2, 0], [2, 1]], dtype=torch.int16
            ),
        },
    }


def test_confidence_gate_recovers_qualifying_top1_outside_sparse_pool():
    payload = make_payload()
    current = torch.tensor([-2.0, -3.0])
    novel = torch.tensor([False, True, True])
    result = clip_confidence_gate_selection(
        payload,
        current,
        novel,
        min_probability=0.9,
        min_margin=0.8,
        vlm_multiplier=0.1,
        novel_scale=3.0,
        max_dets=2,
    )

    pairs = set(zip(result["query_ids"].tolist(), result["class_ids"].tolist()))
    assert pairs == {(0, 1), (1, 2)}
    assert result["recovered_outside_sparse_pool"] == 2
    expected = sorted(
        [math.log(0.90 * 0.3), math.log(0.96 * 0.3)], reverse=True
    )
    assert torch.allclose(
        result["scores"], torch.tensor(expected), atol=1e-6
    )


def test_confidence_gate_requires_probability_margin_and_novel_top1():
    payload = make_payload()
    current = torch.tensor([-0.1, -0.2])
    novel = torch.tensor([False, True, False])
    result = clip_confidence_gate_selection(
        payload,
        current,
        novel,
        min_probability=0.95,
        min_margin=0.9,
        vlm_multiplier=0.1,
        novel_scale=3.0,
        max_dets=2,
    )

    # q1/q2 top-1 is not novel and q0 misses the probability threshold.
    assert result["recovered_outside_sparse_pool"] == 0
    assert torch.allclose(result["scores"], current)


def test_clip_gate_grid_deduplicates_logically_equivalent_thresholds():
    specs = build_clip_gate_specs([0.8, 0.9, 0.95], [0.7, 0.8, 0.9], 0.1)
    assert len(specs) == 9
    assert specs[0]["name"] == "gate_d0_v0"
    confidence = specs[1:]
    assert {
        (row["min_probability"], row["min_margin"]) for row in confidence
    } == {
        (max(probability, margin), margin)
        for probability in (0.8, 0.9, 0.95)
        for margin in (0.7, 0.8, 0.9)
    }
    assert all(row["vlm_multiplier"] == 0.1 for row in confidence)
