"""CPU tests of the actual native Trainer class with only D2 shell/IO stubbed.

These tests do not certify the server's CUDA custom kernels or four-rank NCCL.
"""
import ast
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lami_dino.prototype_ops import route_conflicting_task_gradient
from tools import decoder_aux_ablation_ops as ops
from tools import run_decoder_aux_ablation as runner
from tools import train_decoder_aux_arm as worker


def full_checkpoint():
    return {"iteration": ops.START-1, "trainer": {
        "iteration": ops.START-1, "lr_scheduler_max_iter": ops.HORIZON,
        "gradient_accumulation_steps": 2,
        "optimizer": {"state": {0: {"exp_avg": torch.ones(1)}},
                      "param_groups": [{"params": [0], "lr": 1e-4, "betas": (.9, .999), "weight_decay": 0.}]},
        "hooks": {"LRScheduler": {"last_epoch": ops.START, "base_lrs": [1e-4]}},
        "grad_scaler": {"scale": 1024., "_growth_tracker": 100},
    }}


def test_full_resume_accepts_native_zero_decay_norm_groups():
    checked = ops.validate_resume(full_checkpoint())
    assert checked["lrs"] == [1e-4]


@pytest.mark.parametrize("key,value", [
    ("iteration", 0), ("lr_scheduler_max_iter", 28400), ("gradient_accumulation_steps", 1),
    ("optimizer", {}), ("grad_scaler", {}), ("hooks", {}),
])
def test_incomplete_resume_fails_closed(key, value):
    checkpoint = full_checkpoint()
    checkpoint["trainer"][key] = value
    with pytest.raises(ValueError):
        ops.validate_resume(checkpoint)


def test_wrong_lr_and_nonfinite_scaler_rejected():
    checkpoint = full_checkpoint()
    checkpoint["trainer"]["optimizer"]["param_groups"][0]["lr"] = 1e-5
    with pytest.raises(ValueError, match="plateau"):
        ops.validate_resume(checkpoint)
    checkpoint = full_checkpoint()
    checkpoint["trainer"]["grad_scaler"]["scale"] = float("nan")
    with pytest.raises(ValueError, match="GradScaler"):
        ops.validate_resume(checkpoint)


def test_digest_tracks_dtype_shape_and_optimizer_moments():
    a = {"x": torch.tensor([1.]), "step": 1}
    assert ops.state_digest(a) == ops.state_digest(deepcopy(a))
    for b in ({"x": torch.tensor([[1.]]), "step": 1}, {"x": a["x"].double(), "step": 1},
              {"x": torch.tensor([2.]), "step": 1}, {"x": a["x"], "step": 2}):
        assert ops.state_digest(a) != ops.state_digest(b)


def test_only_auxiliary_core_derivative_removed_after_full_clip():
    core, other = nn.Parameter(torch.ones(2)), nn.Parameter(torch.ones(2))
    aux = (core * torch.tensor([8., 1.])).sum() + other.sum()*3
    remaining = (core * torch.tensor([-7., 5.])).sum() + other.sum()*2
    losses = {f"loss_class_{i}": aux/5 for i in range(5)}
    losses.update(loss_class=remaining/2, loss_class_dn=remaining/4, loss_class_enc=remaining/4)
    g = ops.auxiliary_gradient(losses, (core,), 2)
    torch.testing.assert_close(g, torch.tensor([4., .5]))
    (sum(losses.values())/2).backward()
    original_other = other.grad.clone()
    norm = nn.utils.clip_grad_norm_([core, other], .5)
    factor = (.5/(norm+1e-6)).clamp(max=1)
    clipped = core.grad.clone()
    ops.subtract_clipped_auxiliary((core,), g, norm, .5, remove=False)
    torch.testing.assert_close(core.grad, clipped, rtol=0, atol=0)
    ops.subtract_clipped_auxiliary((core,), g, norm, .5, remove=True)
    torch.testing.assert_close(core.grad, torch.tensor([-3.5, 2.5])*factor)
    torch.testing.assert_close(other.grad, original_other*factor, rtol=0, atol=0)
    # Cancellation removal may increase the norm: no second clipping is allowed.
    assert core.grad.norm() > clipped.norm()


def test_grad_partition_requires_five_aux_not_encoder_or_dn():
    p = nn.Parameter(torch.ones(()))
    with pytest.raises(ValueError, match="five"):
        ops.auxiliary_gradient({"loss_class": p, "loss_class_dn_0": p, "loss_class_enc": p}, (p,), 2)


def test_allreduce_before_subtraction(monkeypatch):
    monkeypatch.setattr(ops.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(ops.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(ops.dist, "all_reduce", lambda g: g.add_(torch.tensor([3., 6.])))
    torch.testing.assert_close(ops.synchronize_auxiliary(torch.tensor([1., 2.])), torch.tensor([1., 2.]))
    with pytest.raises(FloatingPointError):
        ops.synchronize_auxiliary(torch.tensor([float("nan"), 2.]))


def test_layout_failure_does_not_mutate_gradients():
    p = nn.Parameter(torch.ones(2))
    p.grad = torch.tensor([3., 4.])
    with pytest.raises(ValueError, match="layout"):
        ops.subtract_clipped_auxiliary((p,), torch.ones(3), torch.tensor(5.), .5, remove=True)
    torch.testing.assert_close(p.grad, torch.tensor([3., 4.]))


class Shell:
    """Minimal D2 SimpleTrainer shell; native Trainer methods are compiled unchanged."""
    def __init__(self, model, data_loader, optimizer):
        self.model, self.data_loader, self.optimizer = model, data_loader, optimizer
        self._data_loader_iter_obj = None
        self.iter = ops.START-1
        self.hooks = {"LRScheduler": {"last_epoch": ops.START, "base_lrs": [1e-4]}}
    @property
    def _data_loader_iter(self):
        if self._data_loader_iter_obj is None:
            self._data_loader_iter_obj = iter(self.data_loader)
        return self._data_loader_iter_obj
    def state_dict(self):
        return {"iteration": self.iter, "optimizer": self.optimizer.state_dict(), "hooks": self.hooks}
    def load_state_dict(self, state):
        self.iter, self.hooks = state["iteration"], deepcopy(state["hooks"])
        self.optimizer.load_state_dict(state["optimizer"])
    def _write_metrics(self, *a):
        pass


def native_class():
    path = Path(__file__).resolve().parents[1] / "tools/train_net.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "Trainer")
    namespace = {"SimpleTrainer": Shell, "torch": torch, "nn": nn, "time": time,
                 "DistributedDataParallel": nn.parallel.DistributedDataParallel,
                 "DataParallel": nn.DataParallel, "nullcontext": nullcontext,
                 "dist": torch.distributed, "route_conflicting_task_gradient": route_conflicting_task_gradient}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    cls = namespace["Trainer"]
    cls._write_tpa_metrics = lambda self: None
    return cls


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        decoder = self.transformer.decoder = nn.Module()
        decoder.layers = nn.ModuleList([nn.Linear(2, 1)])
        decoder.class_embed = nn.ModuleList([nn.Module()])
        decoder.class_embed[0].tpa = nn.Linear(2, 1, bias=False)
        self.classifier = nn.Linear(1, 1)
        self.novel_idx = torch.zeros(5, dtype=torch.bool)
    def filter_content_info(self, data):
        return torch.randperm(5), data
    def forward(self, data):
        self.filter_content_info(data)
        x = torch.stack([d["image"] for d in data]).mean(0)
        y = self.transformer.decoder.layers[0](x).sum()
        z = sum(p.sum() for p in self.classifier.parameters())
        tpa = self.transformer.decoder.class_embed[0].tpa.weight
        random_weight = torch.rand(())*.1 + 1
        loss = {f"loss_class_{i}": ((y+z-i*.1)**2)*random_weight/5 for i in range(5)}
        loss.update(loss_class=(y-2)**2, loss_class_dn=(y+1)**2, loss_bbox=(y-.3)**2,
                    loss_giou=(y+.4)**2, loss_apr=tpa.square().sum()*.1, loss_rpsa=z.square()*.03)
        return loss


class Data:
    def __iter__(self):
        while True:
            yield [{"image_id": i, "image": torch.rand(2), "instances": SimpleNamespace(
                gt_classes=torch.tensor([i%5]), gt_boxes=SimpleNamespace(tensor=torch.rand(1,4)))} for i in range(4)]


def experiment(tmp_path):
    if not hasattr(torch.amp, "GradScaler"):
        pytest.skip("Native-step CPU AMP test requires torch.amp.GradScaler (CUDA server uses torch.cuda.amp)")
    torch.manual_seed(71)
    cls = native_class()
    model = Toy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # Nonzero inherited AdamW moments are part of the treatment definition.
    sum(p.square().sum() for p in model.parameters()).backward()
    optimizer.step()
    optimizer.zero_grad()
    scaler = torch.amp.GradScaler("cpu", init_scale=1024.)
    kwargs = dict(amp=True, clip_grad_params={"max_norm": .5, "norm_type": 2},
                  separate_tpa_grad_clip=True, tpa_conflict_projection=True,
                  lr_scheduler_max_iter=ops.HORIZON, gradient_accumulation_steps=2)
    trainer = cls(model, Data(), optimizer, grad_scaler=scaler, **kwargs)
    state = deepcopy(trainer.state_dict())
    model_state = deepcopy(model.state_dict())
    manifest = {"output_dir": str(tmp_path), "updates": 2, "seed": 42, "fingerprint": "toy",
                "decoder_keys": sorted(ops.canonical_name(k) for k, p in model.named_parameters()
                                       if ops.key_group(ops.canonical_name(k)) == "decoder_core"),
                "model_digest": ops.state_digest(model_state),
                "resume": ops.validate_resume({"iteration": ops.START-1, "trainer": state})}
    for arm in ops.ARMS:
        (tmp_path/arm).mkdir()
    def build(arm, subclass=True):
        new = Toy()
        new.load_state_dict(model_state)
        opt = torch.optim.AdamW(new.parameters(), lr=1e-4, weight_decay=1e-4)
        impl = ops.make_trainer_class(cls, manifest, arm, 0) if subclass else cls
        t = impl(new, Data(), opt, grad_scaler=torch.amp.GradScaler("cpu", init_scale=1.), **kwargs)
        t.load_state_dict(deepcopy(state))
        return t
    return manifest, build, model_state


def test_actual_native_accumulated_amp_step_preserves_other_modules_and_moments(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda.amp, "autocast", lambda **kw: nullcontext())
    manifest, build, initial = experiment(tmp_path)
    a, b = build("A"), None
    a.iter = ops.START
    a.run_step()
    a.pair_stream.close()
    a.update_stream.close()
    b = build("B")
    b.iter = ops.START
    b.run_step()
    assert a.actual_updates == b.actual_updates == 1
    assert a.initial_state == b.initial_state
    assert a.last_intervention["clip_coefficient"] == b.last_intervention["clip_coefficient"]
    assert a.grad_scaler.get_scale() == b.grad_scaler.get_scale() == 1024.
    changed_core = []
    for k, value in a.raw_model.state_dict().items():
        other = b.raw_model.state_dict()[k]
        if ops.key_group(ops.canonical_name(k)) != "decoder_core":
            torch.testing.assert_close(value, other, rtol=0, atol=0)
        else:
            changed_core.append(not torch.equal(value, other))
    assert any(changed_core)
    for parameter, state in b.optimizer.state.items():
        assert int(state["step"]) == 2  # restored step1, plus exactly one actual step
    assert b.state_dict()["decoder_aux_trial"]["updates"] == 1
    b.pair_stream.close()
    b.update_stream.close()
    b.reference_stream.close()


def test_two_update_pairing_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda.amp, "autocast", lambda **kw: nullcontext())
    _, build, _ = experiment(tmp_path)
    for arm in ops.ARMS:
        trainer = build(arm)
        for i in range(2):
            trainer.iter = ops.START+i
            trainer.run_step()
        assert trainer.actual_updates == 2
        trainer.pair_stream.close()
        trainer.update_stream.close()
        if trainer.reference_stream:
            assert not trainer.reference_stream.readline()
            trainer.reference_stream.close()


def test_arm_a_is_native_update_with_same_inputs_and_rng(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda.amp, "autocast", lambda **kw: nullcontext())
    _, build, _ = experiment(tmp_path)
    a, native = build("A"), build("A", subclass=False)
    a.iter = native.iter = ops.START
    a.run_step()
    iterator, data = iter(Data()), []
    for seed in (42, 44):
        with ops.isolated_rng(seed):
            data.append(next(iterator))
    native._data_loader_iter_obj = iter(data)
    call = native.model.forward
    seeds = iter((43, 45))
    def forward(batch):
        with ops.isolated_rng(next(seeds)):
            return call(batch)
    native.model.forward = forward
    native.run_step()
    for key, value in a.model.state_dict().items():
        torch.testing.assert_close(value, native.model.state_dict()[key], rtol=0, atol=0)
    assert ops.state_digest(a.optimizer.state_dict()) == ops.state_digest(native.optimizer.state_dict())
    a.pair_stream.close()
    a.update_stream.close()


def _ddp_aux_check(rank, rendezvous, directory):
    """Real Gloo DDP/autograd test, independent of D2 and CUDA availability."""
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2)
    try:
        torch.manual_seed(1)
        model = nn.Linear(2, 1)
        ddp = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        parameters = tuple(model.parameters())
        auxiliary = None
        for micro in range(2):
            with (ddp.no_sync() if micro == 0 else nullcontext()):
                x = torch.tensor([[rank+1., micro+1.]])
                y = ddp(x).sum()
                losses = {f"loss_class_{i}": y*(i+1) for i in range(5)}
                losses["loss_class"] = y*2
                g = ops.auxiliary_gradient(losses, parameters, 2)
                auxiliary = g if auxiliary is None else auxiliary+g
                (sum(losses.values())/2).backward()
        auxiliary = ops.synchronize_auxiliary(auxiliary)
        full = torch.cat([p.grad.flatten() for p in parameters])
        torch.testing.assert_close(full, torch.tensor([25.5, 25.5, 17.]))
        torch.testing.assert_close(auxiliary, torch.tensor([22.5, 22.5, 15.]))
        norm = nn.utils.clip_grad_norm_(parameters, .5)
        coefficient = ops.subtract_clipped_auxiliary(parameters, auxiliary, norm, .5, remove=True)
        remaining = torch.cat([p.grad.flatten() for p in parameters])
        torch.testing.assert_close(remaining, torch.tensor([3., 3., 2.])*coefficient)
        Path(directory, f"rank{rank}.ok").write_text("ok")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo unavailable")
def test_real_ddp_aux_autograd_accumulation_and_reduction(tmp_path):
    torch.multiprocessing.spawn(_ddp_aux_check, args=(str(tmp_path/"rendezvous"), str(tmp_path)),
                                nprocs=2, join=True)
    assert all((tmp_path/f"rank{rank}.ok").exists() for rank in range(2))


def test_skipped_step_not_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda.amp, "autocast", lambda **kw: nullcontext())
    _, build, _ = experiment(tmp_path)
    trainer = build("A")
    trainer.iter = ops.START
    monkeypatch.setattr(trainer.grad_scaler, "step", lambda opt: None)
    with pytest.raises(ValueError, match="Skipped"):
        trainer.run_step()
    assert trainer.actual_updates == 0
    trainer.pair_stream.close()
    trainer.update_stream.close()


@pytest.mark.parametrize("key,value", [("mapped", ["changed"]), ("fedloss", [2]),
                                     ("rng_after", "changed"), ("lrs", [1e-5]), ("amp_scale", 2)])
def test_pairing_mismatch_rejected(key, value):
    a = {"iteration": ops.START, "mapped": [], "fedloss": [1], "rng_after": "x", "lrs": [1e-4],
         "amp_scale": 1024., "loss_keys": ["loss_class"], "losses": {"loss_class": 1.}}
    b = deepcopy(a)
    b[key] = value
    with pytest.raises(ValueError, match="A/B"):
        ops.verify_pair(b, a)


def test_losses_only_required_equal_before_intervention():
    a = {"iteration": ops.START, "losses": {"x": 1.}}
    b = {"iteration": ops.START, "losses": {"x": 3.}}
    with pytest.raises(ValueError, match="BEFORE"):
        ops.verify_pair(b, a)
    a["iteration"] += 1
    b["iteration"] += 1
    ops.verify_pair(b, a)


def test_training_and_evaluation_commands_keep_protocol():
    m = {"output_dir": "/tmp/trial", "updates": 500, "checkpoint": {"path": "/tmp/8ep.pth"}}
    options = worker.training_options(m, "B")
    assert "train.max_iter=57300" in options
    assert "train.lr_scheduler_max_iter=85200" in options
    assert "train.eval_after_train=False" in options
    assert "dataloader.train.num_workers=0" in options
    assert not any("weight_dict" in x or "lr=" in x for x in options)
    cmd = runner.evaluation_command("config.py", "checkpoint.pth", Path("/tmp/eval"), 4)
    assert "--eval-only" in cmd and "--resume" not in cmd
    assert "model.classifier.tpa_prototype_mode_strength=0.0" in cmd


def test_evaluation_cannot_silently_succeed(tmp_path):
    (tmp_path/"console.log").write_text("Skipping evaluation - training will continue")
    with pytest.raises(RuntimeError, match="failed"):
        runner.collect_evaluation(tmp_path)
    (tmp_path/"console.log").write_text("")
    (tmp_path/"log.txt").write_text("copypaste: " + ",".join(["40.0000"]*9))
    with pytest.raises(ValueError, match="predictions"):
        runner.collect_evaluation(tmp_path)
    (tmp_path/"lvis_instances_results.json").write_text('[{"score": 0.5}]')
    assert runner.collect_evaluation(tmp_path)["metrics"]["APr"] == 40.


def test_prepare_only_never_executes_gpu(monkeypatch):
    args = runner.parse_args(["--classification-audit", "unused", "--output-dir", "unused", "--prepare-only"])
    monkeypatch.setattr(runner, "prepare", lambda a: {"prepared": True})
    monkeypatch.setattr(runner, "execute", lambda m: pytest.fail("GPU launched"))
    assert runner.run(args) == {"prepared": True}


def test_incomplete_run_does_not_resume_or_overwrite(tmp_path, monkeypatch):
    for arm in ops.ARMS:
        (tmp_path/arm).mkdir()
        (tmp_path/arm/"last_checkpoint").write_text("original")
    (tmp_path/"A"/"console.log").write_text("prior failed training")
    monkeypatch.setattr(runner, "verified_identity", lambda *a: None)
    monkeypatch.setattr(runner, "run_evaluation", lambda *a: pytest.fail("training launched"))
    with pytest.raises(ValueError, match="Incomplete A"):
        runner.execute({"output_dir": str(tmp_path), "checkpoint": {}})
    assert (tmp_path/"A"/"console.log").read_text() == "prior failed training"


def test_prepare_writes_independent_pointers_without_changing_source(tmp_path, monkeypatch):
    source, audit = tmp_path/"source", tmp_path/"audit"
    source.mkdir()
    audit.mkdir()
    path = source/"model_0056799.pth"
    checkpoint = full_checkpoint()
    checkpoint["model"] = {"x": torch.ones(1)}
    torch.save(checkpoint, path)
    before = path.read_bytes()
    (audit/"report.json").write_text("{}")
    (source/"config.py").write_text("# mock")
    report = {"inputs": {"code": {}, "sources": {
        "old_checkpoint": runner.file_identity(path), "config_file": runner.file_identity(source/"config.py")}}}
    monkeypatch.setattr(runner, "validate_split", lambda p: (report, {"inventory": {"decoder_core": {"keys": ["x"]}}}))
    monkeypatch.setattr(runner, "endpoint_state", lambda c, i: c["model"])
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda p: SimpleNamespace(free=10**12))
    args = runner.parse_args(["--classification-audit", str(audit/"report.json"), "--output-dir", str(tmp_path/"trial")])
    manifest = runner.prepare(args)
    assert worker.read_manifest(tmp_path/"trial"/"manifest.json") == manifest
    for arm in ops.ARMS:
        assert (tmp_path/"trial"/arm/"last_checkpoint").read_text() == str(path)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="NEW output"):
        runner.prepare(args)


def test_exactly_two_trainings_then_two_evaluations_and_full_delta(tmp_path, monkeypatch):
    for arm in ops.ARMS:
        (tmp_path/arm).mkdir()
        (tmp_path/arm/"last_checkpoint").write_text("original")
    m = {"output_dir": str(tmp_path), "checkpoint": {}, "updates": 500, "fingerprint": "test",
         "config": {"path": "config.py"}, "scope": ["test"]}
    calls = []
    monkeypatch.setattr(runner, "verified_identity", lambda *a: None)
    monkeypatch.setattr(runner, "run_evaluation", lambda cmd, path: calls.append(path.name))
    monkeypatch.setattr(runner, "verify_arm", lambda m, arm: {"checkpoint": {"path": arm+".pth"}})
    monkeypatch.setattr(runner, "collect_evaluation", lambda d: {
        "metrics": {k: 42. if d.name == "eval_A" else 41. for k in runner.METRICS}, "predictions": {}})
    result = runner.execute(m)
    assert calls == ["A", "B", "eval_A", "eval_B"]
    assert result["delta_B_minus_A"]["APr"] == -1.
    assert result["updates_per_arm"] == 500
    assert (tmp_path/"summary.json").is_file()


def test_unverified_b_never_gets_formal_eval(tmp_path, monkeypatch):
    for arm in ops.ARMS:
        (tmp_path/arm).mkdir()
        (tmp_path/arm/"last_checkpoint").write_text("original")
    calls = []
    monkeypatch.setattr(runner, "verified_identity", lambda *a: None)
    monkeypatch.setattr(runner, "run_evaluation", lambda cmd, path: calls.append(path.name))
    def verify(m, arm):
        if arm == "B":
            raise ValueError("mismatched B")
        return {"checkpoint": {}}
    monkeypatch.setattr(runner, "verify_arm", verify)
    with pytest.raises(ValueError, match="mismatched B"):
        runner.execute({"output_dir": str(tmp_path), "checkpoint": {}, "updates": 500})
    assert calls == ["A", "B"]
    assert not (tmp_path/"summary.json").exists()
