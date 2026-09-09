import torch

from tools.analyze_ovd_error_decomposition import classify_rare_detections
from tools.analyze_ovd_rescue_gates import (
    build_gate_grid,
    choose_gates,
    gated_log_scores,
)


def test_gate_changes_only_novel_candidates():
    current = torch.tensor([-2.0, -2.0, -3.0])
    detector = torch.tensor([-1.0, -1.0, -1.0])
    vlm = torch.tensor([-4.0, -0.5, -4.0])
    classes = torch.tensor([0, 1, 2])
    novel = torch.tensor([False, True, True])

    result = gated_log_scores(
        current,
        detector,
        vlm,
        classes,
        novel,
        detector_multiplier=0.5,
        vlm_multiplier=0.5,
    )

    assert result[0] == current[0]
    assert torch.allclose(result[1], vlm[1] + torch.log(torch.tensor(0.5)))
    assert torch.allclose(result[2], detector[2] + torch.log(torch.tensor(0.5)))


def test_gate_grid_is_unique_and_contains_single_and_joint_branches():
    rows = build_gate_grid([0.1, 0.5], [0.1, 0.5])
    pairs = {
        (row["detector_multiplier"], row["vlm_multiplier"])
        for row in rows
    }
    assert len(rows) == len(pairs) == 9
    assert (0.0, 0.0) in pairs
    assert (0.5, 0.0) in pairs
    assert (0.0, 0.5) in pairs
    assert (0.5, 0.5) in pairs


def test_gate_selection_respects_precision_budget_and_pareto_frontier():
    def row(name, net_tp, delta_fp, lost_tp):
        return {
            "name": name,
            "detector_multiplier": 0.1,
            "vlm_multiplier": 0.0,
            "net_tp": net_tp,
            "delta_fp": delta_fp,
            "lost_tp": lost_tp,
            "added_fp_per_net_tp": delta_fp / net_tp if net_tp > 0 else None,
        }

    rows = [
        row("good", 4, 2, 0),
        row("dominated", 3, 3, 0),
        row("too_many_fp", 5, 20, 0),
        row("loses_tp", 5, 2, 4),
    ]
    selected = choose_gates(rows, max_added_fp_per_net_tp=1.0, max_lost_tp=1, limit=3)
    assert [item["name"] for item in selected] == ["good"]


def test_added_detection_can_treat_baseline_gt_as_already_matched():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    records = classify_rare_detections(
        boxes,
        torch.tensor([0.9]),
        torch.tensor([0]),
        boxes,
        torch.tensor([0]),
        negative_classes=set(),
        not_exhaustive_classes=set(),
        iou_threshold=0.5,
        background_iou=0.1,
        initially_matched_gt=torch.tensor([True]),
    )
    assert records[0][2] == "duplicate"
