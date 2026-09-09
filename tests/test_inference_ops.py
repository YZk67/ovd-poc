import pytest
import torch

from lami_dino.inference_ops import select_query_class_topk


def test_zero_cap_exactly_matches_original_global_topk():
    torch.manual_seed(4)
    scores = torch.randn(2, 5, 7)
    expected_scores, expected_ids = scores.reshape(2, -1).topk(9, dim=1)

    actual_scores, queries, classes = select_query_class_topk(
        scores, max_detections=9, per_query_class_topk=0
    )

    torch.testing.assert_close(actual_scores, expected_scores)
    assert torch.equal(queries, expected_ids // 7)
    assert torch.equal(classes, expected_ids % 7)


def test_positive_cap_limits_each_query_before_global_selection():
    scores = torch.tensor(
        [
            [
                [10.0, 9.0, 8.0, 7.0],
                [6.0, 5.0, 4.0, 3.0],
                [2.0, 1.0, 0.0, -1.0],
            ]
        ]
    )
    selected_scores, queries, classes = select_query_class_topk(
        scores, max_detections=3, per_query_class_topk=1
    )

    torch.testing.assert_close(selected_scores, torch.tensor([[10.0, 6.0, 2.0]]))
    assert queries.tolist() == [[0, 1, 2]]
    assert classes.tolist() == [[0, 0, 0]]


def test_cap_larger_than_class_count_matches_global_topk():
    torch.manual_seed(9)
    scores = torch.randn(1, 4, 3)
    global_result = select_query_class_topk(
        scores, max_detections=8, per_query_class_topk=0
    )
    capped_result = select_query_class_topk(
        scores, max_detections=8, per_query_class_topk=99
    )
    for actual, expected in zip(capped_result, global_result):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("cap", [-1, -3])
def test_negative_cap_is_rejected(cap):
    with pytest.raises(ValueError, match="non-negative"):
        select_query_class_topk(
            torch.ones(1, 2, 3),
            max_detections=2,
            per_query_class_topk=cap,
        )
