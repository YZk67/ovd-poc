from collections import defaultdict

import torch

from tools.diagnose_tpa_usage import (
    finalize_ranking_stats,
    finalize_selection_accumulator,
    make_selection_accumulator,
    select_dataset_records,
    update_ranking_stats,
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
