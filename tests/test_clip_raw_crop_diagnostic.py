import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from tools.diagnose_clip_raw_crops import (
    raw_crop_clip_logits,
    select_rare_rows,
    square_context_crop,
    summarize_raw_crop_rows,
)


def test_square_context_crop_preserves_center_and_pads_at_border():
    image = Image.new("RGB", (10, 8), (255, 0, 0))
    crop = square_context_crop(image, [0, 0, 2, 2], scale=2.0, size=8)
    array = np.asarray(crop)
    assert crop.size == (8, 8)
    assert np.all(array[4, 4] == [255, 0, 0])
    # Bicubic resizing can change a border pixel by one or two intensity units.
    assert np.allclose(array[0, 0], [124, 116, 104], atol=2)
    with pytest.raises(ValueError, match="invalid GT box"):
        square_context_crop(image, [2, 2, 1, 1], scale=1.0, size=8)


def test_rare_row_selection_keeps_misses_and_bounds_controls():
    rows = [
        {"image_id": index, "gt_index": 0, "status": status}
        for index, status in enumerate(
            ["current_miss"] * 4 + ["current_hit"] * 4 + ["no_eligible_box"] * 4
        )
    ]
    selected = select_rare_rows(
        rows, max_misses=0, max_hit_controls=2, max_no_box_controls=1, seed=42
    )
    assert [row["status"] for row in selected].count("current_miss") == 4
    assert [row["status"] for row in selected].count("current_hit") == 2
    assert [row["status"] for row in selected].count("no_eligible_box") == 1
    selected_again = select_rare_rows(
        rows, max_misses=0, max_hit_controls=2, max_no_box_controls=1, seed=42
    )
    assert selected == selected_again
    misses_only = select_rare_rows(
        rows, max_misses=0, max_hit_controls=0, max_no_box_controls=0, seed=42
    )
    assert len(misses_only) == 4


def test_raw_crop_uses_visual_head_and_existing_text_bank():
    class Backbone:
        def __call__(self, x):
            return {}, {"p3": x.mean(dim=(2, 3), keepdim=True)}

    class Model:
        score_ensemble = True
        vlm_temperature = 2.0
        vlm_content_query_embedding = torch.eye(3)
        backbone = Backbone()
        identical = nn.Identity()
        thead = nn.Flatten(start_dim=1)
        head = nn.Identity()
        normalizer = staticmethod(lambda x: x / 255.0)

    crops = torch.zeros((2, 3, 4, 4))
    crops[0, 0] = 255
    crops[1, 1] = 255
    logits = raw_crop_clip_logits(Model(), crops)
    torch.testing.assert_close(
        logits, torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    )


def test_summary_keeps_both_rescues_and_regressions():
    rows = [
        {
            "status": "current_miss",
            "gt_rank": 10,
            "gt_true_probability": 0.01,
            "raw_crop": {"1.0": {"rank": 2, "true_probability": 0.2}},
        },
        {
            "status": "current_miss",
            "gt_rank": 2,
            "gt_true_probability": 0.2,
            "raw_crop": {"1.0": {"rank": 10, "true_probability": 0.01}},
        },
    ]
    summary = summarize_raw_crop_rows(rows, [1.0])["1.0"]["current_miss"]
    assert summary["count"] == 2
    assert summary["roi_top5"] == summary["raw_top5"] == 0.5
    assert summary["raw_rescues_roi_top5"] == 1
    assert summary["raw_worsens_roi_top5"] == 1
