import pytest
import torch
from torch import nn

from tools import tpa_formula_screen_ops as ops
from tools.train_tpa_formula_arm import training_options


class FakeTPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.prototype_queries = nn.Parameter(torch.randn(5, 3))
        self.key_proj = nn.Linear(4, 3)
        self.value_proj = nn.Linear(4, 4)
        self.register_buffer("slot_prior_strength", torch.tensor(.2))
        self.register_buffer("prototype_mode_strength", torch.tensor(0.))


class Head(nn.Module):
    def __init__(self, tpa):
        super().__init__()
        self.tpa = tpa
        self.tpa_train_aggregation = "calibrated"


class Toy(nn.Module):
    def __init__(self, *, shared=True):
        super().__init__()
        self.detector = nn.Linear(4, 1)
        self.transformer = nn.Module()
        self.transformer.decoder = nn.Module()
        first = FakeTPA()
        second = first if shared else FakeTPA()
        if not shared:
            second.load_state_dict(first.state_dict())
        self.transformer.decoder.class_embed = nn.ModuleList([Head(first), Head(second)])
        self.novel_idx = torch.zeros(5, dtype=torch.bool)


def test_training_formula_is_separate_from_unchanged_tpa_geometry():
    model = Toy(shared=False)
    before = ops.geometry_digest(model)
    ops.set_training_formula(model, "legacy")
    ops.verify_training_formula(model, "legacy")
    assert ops.geometry_digest(model) == before
    assert len(ops.tpa_parameters(model)) == 5 * 2
    with pytest.raises(ValueError, match="Unknown TPA"):
        ops.set_training_formula(model, "not-a-formula")


def test_live_geometry_rejects_unequal_head_aliases():
    model = Toy(shared=False)
    with torch.no_grad():
        model.transformer.decoder.class_embed[1].tpa.prototype_queries.add_(1)
    with pytest.raises(ValueError, match="Unequal live TPA aliases"):
        ops.live_tpa_geometry(model)


def test_pairing_allows_only_intended_loss_value_difference():
    base = {
        "iteration": ops.START,
        "mapped": [1],
        "fedloss": [2],
        "rng_after": {"cpu": "same"},
        "lrs": [1e-4],
        "amp_scale": 1024,
        "loss_keys": ["loss_class"],
        "losses": {"loss_class": 1.0},
    }
    changed_loss = {**base, "losses": {"loss_class": 2.0}}
    ops.verify_formula_pair(changed_loss, base)
    changed_data = {**changed_loss, "mapped": [9]}
    with pytest.raises(ValueError, match="Formula arms differ"):
        ops.verify_formula_pair(changed_data, base)


class MinimalNative:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.separate_tpa_grad_clip = True
        self.tpa_conflict_projection = True
        self.gradient_accumulation_steps = 2
        self.clip_grad_params = {"max_norm": .5, "norm_type": 2}


def test_optimizer_wrapper_freezes_weights_and_adamw_state(tmp_path):
    for arm in ops.ARMS:
        (tmp_path / arm).mkdir()
    model = Toy(shared=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    manifest = {
        "output_dir": str(tmp_path),
        "updates": 1,
        "seed": 42,
        "fingerprint": "unit-test",
    }
    trainer_type = ops.make_trainer_class(MinimalNative, manifest, "A", 0)
    trainer = trainer_type(model, optimizer)
    tpa_before = {id(parameter): parameter.detach().clone() for parameter in trainer.frozen_tpa_parameters}
    detector_before = model.detector.weight.detach().clone()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert trainer.actual_updates == 1
    assert not torch.equal(model.detector.weight, detector_before)
    for parameter in trainer.frozen_tpa_parameters:
        torch.testing.assert_close(parameter, tpa_before[id(parameter)], rtol=0, atol=0)
        assert parameter not in optimizer.state
    trainer.pair_stream.close()
    trainer.update_stream.close()


def test_worker_forces_common_calibrated_evaluation_controls():
    manifest = {"output_dir": "/tmp/formula", "checkpoint": {"path": "/tmp/source.pth"}, "updates": 3}
    for arm, aggregation in ops.ARMS.items():
        options = training_options(manifest, arm)
        assert f'model.classifier.tpa_train_aggregation="{aggregation}"' in options
        assert "model.classifier.tpa_eval_legacy_logsumexp=False" in options
        assert "model.classifier.tpa_eval_logit_bias=0.0" in options
