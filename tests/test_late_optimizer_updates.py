"""Finite native AdamW/AMP tests on CPU; not a CUDA/NCCL integration certification."""
from contextlib import nullcontext
from copy import deepcopy
import math
from types import SimpleNamespace

import pytest
import torch

from tools import late_optimizer_audit_ops as ops
from tools import audit_late_optimizer_updates as runner
from tools.decoder_aux_ablation_ops import state_digest, validate_resume
from test_decoder_aux_ablation import Toy, native_class, full_checkpoint


class Model(Toy):
    def __init__(self):
        super().__init__()
        self.register_buffer("transient", torch.tensor([2.]), persistent=False)
        self.unused = torch.nn.Parameter(torch.tensor([1.]))

    def forward(self, data):
        losses = super().forward(data)
        losses["loss_class_enc"] = losses["loss_class"]*.2
        return losses


def initialized():
    torch.manual_seed(27)
    model = Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sum(p.square().sum() for p in model.parameters()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def trainer(model, optimizer, data=()):
    if not hasattr(torch.amp, "GradScaler"):
        pytest.skip("CPU AMP requires recent torch; server uses torch.cuda.amp")
    return native_class()(model, data, optimizer, amp=True,
                          grad_scaler=torch.amp.GradScaler("cpu", init_scale=1024.),
                          clip_grad_params={"max_norm": .5, "norm_type": 2},
                          separate_tpa_grad_clip=True, tpa_conflict_projection=True,
                          lr_scheduler_max_iter=85200, gradient_accumulation_steps=2)


def capture(tr, batches):
    named = [(n, p) for n, p in tr.model.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    parts = {s: [torch.zeros_like(p) for p in parameters] for s in ops.SOURCES}
    apr = [torch.zeros_like(p) for p in parameters]
    ids = {id(p): i for i, p in enumerate(parameters)}
    tr.optimizer.zero_grad(set_to_none=True)
    for batch in batches:
        losses = tr.model(batch)
        for p, g in zip(tr._get_tpa().parameters(), tr._compute_apr_gradients(losses, gradient_scale=.5)):
            if g is not None:
                apr[ids[id(p)]] += g.detach()
        for source, keys in ops.grouped_losses(losses).items():
            grads = torch.autograd.grad(tr.grad_scaler.scale(sum(losses[k] for k in keys)/2), parameters,
                                        retain_graph=True, allow_unused=True)
            for i, g in enumerate(grads):
                if g is not None:
                    parts[source][i] += g.detach()/tr.grad_scaler.get_scale()
        tr.grad_scaler.scale(sum(losses.values())/2).backward()
    tr.grad_scaler.unscale_(tr.optimizer)
    payload = dict(names=[n for n, _ in named], present=[p.grad is not None for p in parameters],
                   full=[p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in parameters],
                   components=parts, apr_reference=apr)
    return named, payload


def test_10ep_complete_resume_not_weights_only_or_8ep():
    checkpoint = full_checkpoint()
    with pytest.raises(ValueError, match="70999"):
        validate_resume(checkpoint, start=71000)
    checkpoint["iteration"] = checkpoint["trainer"]["iteration"] = 70999
    checkpoint["trainer"]["hooks"]["LRScheduler"]["last_epoch"] = 71000
    assert validate_resume(checkpoint, start=71000)["lrs"] == [1e-4]
    for key in ("optimizer", "grad_scaler", "hooks"):
        invalid = deepcopy(checkpoint)
        invalid["trainer"].pop(key)
        with pytest.raises(ValueError):
            validate_resume(invalid, start=71000)


def test_coarse_sources_exhaustive_and_preserve_encoder_dn_aux():
    model, _ = initialized()
    losses = model([{"image": torch.ones(2)}])
    groups = ops.grouped_losses(losses)
    assert set(sum(groups.values(), [])) == set(losses)
    assert set(groups["classification"]) == {"loss_class", "loss_class_enc", "loss_class_dn",
                                           *[f"loss_class_{i}" for i in range(5)]}
    losses["loss_something_new"] = losses["loss_class"]
    with pytest.raises(ValueError, match="Unclassified"):
        ops.grouped_losses(losses)


def test_actual_native_amp_accumulated_step_matches_trainer(monkeypatch):
    # Execute the repository Trainer.run_step itself, stubbing only CUDA availability/autocast.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda.amp, "autocast", lambda **kw: nullcontext())
    batches = [[{"image": torch.tensor([.4, -.3])}], [{"image": torch.tensor([-.2, .8])}]]
    model, optimizer = initialized()
    snapshot, opt = ops.model_snapshot(model), ops.cpu_copy(optimizer.state_dict())
    native = trainer(model, optimizer, batches)
    torch.manual_seed(812)
    native.run_step()
    expected = ops.model_snapshot(model)
    expected_opt = ops.cpu_copy(optimizer.state_dict())
    expected_scaler = native.grad_scaler.state_dict()
    ops.restore_model(model, snapshot)
    optimizer.load_state_dict(deepcopy(opt))
    tr = trainer(model, optimizer)
    torch.manual_seed(812)
    named, payload = capture(tr, batches)
    assert ops.validate_gradients(payload, named) < 1e-6
    # fresh restored AMP scaler: old per-optimizer unscale state must not leak.
    tr = trainer(model, optimizer)
    stats = ops.actual_step(tr, named, payload, "native")
    assert stats["optimizer_steps"] == 1
    for k, value in model.state_dict().items():
        torch.testing.assert_close(value, expected["state"][k], rtol=0, atol=0)
    assert state_digest(optimizer.state_dict()) == state_digest(expected_opt)
    assert tr.grad_scaler.state_dict() == expected_scaler


@pytest.mark.parametrize("arm", ops.ARMS)
def test_every_arm_restores_parameters_moments_and_nonpersistent_buffers_on_error(arm):
    model, optimizer = initialized()
    snapshot, opt = ops.model_snapshot(model), ops.cpu_copy(optimizer.state_dict())
    named, payload = capture(trainer(model, optimizer), [[{"image": torch.ones(2)}]]*2)
    with pytest.raises(RuntimeError, match="fake eval failure"):
        with ops.restored_branch(model, optimizer, snapshot, opt, buffer_changes={"transient": torch.tensor([7.])}):
            assert model.transient.item() == 7.
            result = ops.actual_step(trainer(model, optimizer), named, payload, arm)
            assert result["optimizer_steps"] == 1
            assert state_digest(model.state_dict()) != state_digest(snapshot["state"])
            # Globally unused parameters keep None: even weight decay must not step them.
            assert model.unused.grad is None
            torch.testing.assert_close(model.unused, snapshot["state"]["unused"], rtol=0, atol=0)
            raise RuntimeError("fake eval failure")
    assert state_digest(ops.model_snapshot(model)) == state_digest(snapshot)
    assert state_digest(optimizer.state_dict()) == state_digest(opt)


def test_gradient_layout_nonfinite_and_reconstruction_fail_closed():
    model, optimizer = initialized()
    named, payload = capture(trainer(model, optimizer), [[{"image": torch.ones(2)}]]*2)
    for key in ("full", "apr_reference"):
        bad = deepcopy(payload)
        bad[key][0].fill_(float("nan"))
        with pytest.raises(ValueError, match="nonfinite"):
            ops.validate_gradients(bad, named)
    bad = deepcopy(payload)
    bad["full"][1].add_(100.)
    with pytest.raises(ValueError, match="reconstruct"):
        ops.validate_gradients(bad, named)
    bad = deepcopy(payload)
    bad["names"] = list(reversed(bad["names"]))
    with pytest.raises(ValueError, match="inventory"):
        ops.validate_gradients(bad, named)


def fake_analysis(margin):
    return {"image_cutoff": .5, "by_iou": {"0.50": {
        "raw_eligible_queries": 3, "retained_true_class_queries": int(margin > 0),
        "best_fused_true_class_eligible": {"score_threshold_ratio": math.exp(margin), "fused_score": .5*math.exp(margin)}}}}


def fake_window(i, native, removed):
    return dict(window=i, buffer_only=fake_analysis(1.),
                arms={k: {"analysis": fake_analysis(native if k == "native" else removed)} for k in ops.ARMS})


def test_improvement_without_native_degradation_is_not_a_source_claim():
    result = ops.summarize([fake_window(0, 1.1, 1.2), fake_window(1, 1.05, 1.2)])
    assert result["consistent_local_candidates"] == []
    assert result["verdict"] == "NO_CONSISTENT_LOCAL_SOURCE"


def test_local_candidates_still_neither_historical_causality_nor_global_ap():
    result = ops.summarize([fake_window(0, .8, .9), fake_window(1, .7, .8)])
    assert result["consistent_local_candidates"] == list(ops.SOURCES)
    assert not result["historical_causality_proven"]
    assert not result["global_AP_measured"]
    assert not result["next_training_recommended"]


def test_hard_budget_checked_before_reading_files(tmp_path):
    for windows, gpus in ((0, 4), (5, 4), (2, 1)):
        with pytest.raises(ValueError, match="Hard budget"):
            runner.prepare(SimpleNamespace(windows=windows, num_gpus=gpus, seed=42))


def test_partial_artifact_never_silently_recaptured(tmp_path):
    (tmp_path/"window_00.json").write_text("{}")
    with pytest.raises(ValueError, match="Incomplete window"):
        runner.read_artifact(tmp_path, 0, "fingerprint")
