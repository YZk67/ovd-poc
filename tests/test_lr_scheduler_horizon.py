import pytest
import torch

from detectron2.solver import LRMultiplier
from fvcore.common.param_scheduler import MultiStepParamScheduler


def _make_scheduler(horizon):
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=1e-4)
    multiplier = MultiStepParamScheduler(
        values=[1.0, 0.1],
        milestones=[78_100, 85_200],
    )
    scheduler = LRMultiplier(optimizer, multiplier, max_iter=horizon)
    return optimizer, scheduler


def test_four_epoch_stop_uses_twelve_epoch_lr_prefix():
    optimizer, scheduler = _make_scheduler(85_200)
    scheduler.last_epoch = 28_399
    assert scheduler.get_lr() == pytest.approx([1e-4])

    scheduler.last_epoch = 78_100
    assert scheduler.get_lr() == pytest.approx([1e-5])


def test_using_stop_iteration_as_horizon_would_compress_decay():
    _, scheduler = _make_scheduler(28_400)
    scheduler.last_epoch = 28_399
    assert scheduler.get_lr() == pytest.approx([1e-5])
