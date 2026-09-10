import pytest
import torch
from torch import nn

from tools.train_net import Trainer


class _LinearLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.forward_calls = 0

    def forward(self, coefficient):
        self.forward_calls += 1
        return {"loss_linear": self.weight * coefficient}


def _trainer(*, accumulation_steps):
    model = _LinearLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        dataloader=[torch.tensor(2.0), torch.tensor(4.0)],
        optimizer=optimizer,
        gradient_accumulation_steps=accumulation_steps,
    )
    trainer._write_tpa_metrics = lambda: None
    return model, trainer


def test_two_micro_batches_make_one_averaged_optimizer_update(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model, trainer = _trainer(accumulation_steps=2)
    recorded = {}
    trainer._write_metrics = lambda losses, data_time: recorded.update(losses)

    trainer.run_step()

    assert model.forward_calls == 2
    # Mean gradient of 2*w and 4*w is 3, hence 1 - 0.1*3 = 0.7.
    torch.testing.assert_close(model.weight, torch.tensor(0.7))
    torch.testing.assert_close(recorded["loss_linear"], torch.tensor(3.0))


def test_accumulation_setting_is_checkpointed_and_resume_locked():
    _, trainer = _trainer(accumulation_steps=2)
    state = trainer.state_dict()
    assert state["gradient_accumulation_steps"] == 2

    _, incompatible = _trainer(accumulation_steps=1)
    with pytest.raises(ValueError, match="different gradient accumulation"):
        incompatible.load_state_dict(state)


def test_apr_gradient_tuples_accumulate_optional_entries():
    first = (torch.tensor([1.0, 2.0]), None)
    second = (torch.tensor([3.0, 4.0]), torch.tensor([5.0]))

    accumulated = Trainer._accumulate_gradient_tuples(None, first)
    accumulated = Trainer._accumulate_gradient_tuples(accumulated, second)

    torch.testing.assert_close(accumulated[0], torch.tensor([4.0, 6.0]))
    torch.testing.assert_close(accumulated[1], torch.tensor([5.0]))
