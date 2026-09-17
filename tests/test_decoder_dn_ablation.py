from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tools import decoder_dn_ablation_ops as ops
from tools import run_decoder_dn_ablation as runner
from tools import train_decoder_dn_arm as worker


def test_dn_partition_is_final_plus_five_auxiliary_dn_terms_only():
    core = nn.Parameter(torch.ones(2))
    other = nn.Parameter(torch.ones(2))
    source = (core*torch.tensor([6., 12.])).sum() + other.sum()*9
    remaining = (core*torch.tensor([-4., 3.])).sum() + other.sum()*2
    losses = {"loss_class_dn": source/6}
    losses.update({f"loss_class_dn_{i}": source/6 for i in range(5)})
    losses.update({"loss_class": remaining/2, "loss_class_0": remaining/2,
                   "loss_bbox_dn": source*0, "loss_giou_dn_0": source*0})
    gradient = ops.dn_classification_gradient(losses, (core,), 2)
    torch.testing.assert_close(gradient, torch.tensor([3., 6.]))
    (sum(losses.values())/2).backward()
    full_other = other.grad.clone()
    norm = nn.utils.clip_grad_norm_([core, other], .5)
    coefficient = (.5/(norm+1e-6)).clamp(max=1)
    ops.subtract_clipped_auxiliary((core,), gradient, norm, .5, remove=True)
    torch.testing.assert_close(core.grad, torch.tensor([-2., 1.5])*coefficient)
    torch.testing.assert_close(other.grad, full_other*coefficient)


@pytest.mark.parametrize("losses", [
    {"loss_class_dn": torch.ones((), requires_grad=True)},
    {**{"loss_class_dn": torch.ones((), requires_grad=True)},
     **{f"loss_class_dn_{i}": torch.ones((), requires_grad=True) for i in range(6)}},
])
def test_dn_partition_fails_closed_on_missing_or_extra_terms(losses):
    parameter = nn.Parameter(torch.ones(()))
    with pytest.raises(ValueError, match="final plus five"):
        ops.dn_classification_gradient(losses, (parameter,), 2)


class FakeScaler:
    def state_dict(self):
        return {"scale": 1024.}
    def get_scale(self):
        return 1024.


class Native:
    def __init__(self, model, data, optimizer):
        self.model, self.optimizer = model, optimizer
        self._native_iterator = iter(data)
        self.iter = ops.START-1
        self.separate_tpa_grad_clip = self.tpa_conflict_projection = True
        self.gradient_accumulation_steps = 2
        self.clip_grad_params = {"max_norm": .5, "norm_type": 2}
        self.grad_scaler, self.amp = FakeScaler(), False
        self.hooks = {"LRScheduler": {"last_epoch": ops.START, "base_lrs": [1e-4]}}
    @property
    def _data_loader_iter(self):
        return self._native_iterator
    def state_dict(self):
        return {"iteration": self.iter, "optimizer": self.optimizer.state_dict(),
                "hooks": deepcopy(self.hooks), "grad_scaler": self.grad_scaler.state_dict()}
    def load_state_dict(self, state):
        self.iter = state["iteration"]
        self.hooks = deepcopy(state["hooks"])
        self.optimizer.load_state_dict(state["optimizer"])
    def _compute_apr_gradients(self, loss_dict, *, gradient_scale=1.):
        return None
    def clip_model_grads(self):
        return (nn.utils.clip_grad_norm_(self.model.parameters(), .5), None)
    def run_step(self):
        self.optimizer.zero_grad()
        for _ in range(2):
            losses = self.model(next(self._data_loader_iter))
            self._compute_apr_gradients(losses, gradient_scale=.5)
            (sum(losses.values())/2).backward()
        self.clip_model_grads()
        self.optimizer.step()


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.decoder = nn.Module()
        self.transformer.decoder.layers = nn.ModuleList([nn.Linear(2, 1, bias=False)])
        self.other = nn.Parameter(torch.tensor([.4]))
        self.novel_idx = torch.zeros(5, dtype=torch.bool)
    def filter_content_info(self, data):
        return torch.tensor([0, 1]), data
    def forward(self, data):
        self.filter_content_info(data)
        x = torch.stack([row["image"] for row in data]).mean(0)
        value = self.transformer.decoder.layers[0](x).sum()
        dn = (value+self.other).square()
        ordinary = (value-self.other-1).square()
        losses = {"loss_class_dn": dn/6}
        losses.update({f"loss_class_dn_{i}": dn/6 for i in range(5)})
        losses.update({"loss_class": ordinary/2, "loss_class_0": ordinary/2,
                       "loss_bbox": (value-.2).square(), "loss_giou": (value+.3).square()})
        return losses


def batches():
    batch = [
        {"image_id": index, "image": torch.tensor([index+1., 1.]),
         "instances": SimpleNamespace(gt_classes=torch.tensor([index%5]),
                                      gt_boxes=SimpleNamespace(tensor=torch.ones(1, 4)))}
        for index in range(4)
    ]
    return [deepcopy(batch), deepcopy(batch)]


def test_actual_step_changes_only_decoder_relative_to_native(tmp_path, monkeypatch):
    torch.manual_seed(4)
    initial = Toy()
    native_model, blocked_model = deepcopy(initial), deepcopy(initial)
    native_optimizer = torch.optim.SGD(native_model.parameters(), lr=1e-4)
    blocked_optimizer = torch.optim.SGD(blocked_model.parameters(), lr=1e-4)
    native = Native(native_model, batches(), native_optimizer)
    state = native.state_dict()
    model_digest = ops.state_digest(initial.state_dict())
    reference = tmp_path/"reference.jsonl"
    reference.write_text("{}\n{}\n")
    output = tmp_path/"trial"
    (output/"B").mkdir(parents=True)
    manifest = {
        "output_dir": str(output), "updates": 1, "seed": 42, "fingerprint": "toy",
        "decoder_keys": sorted(
            ops.canonical_name(name) for name, parameter in blocked_model.named_parameters()
            if ops.key_group(ops.canonical_name(name)) == "decoder_core"),
        "model_digest": model_digest,
        "resume": {
            "optimizer": ops.state_digest(blocked_optimizer.state_dict()),
            "scheduler": ops.state_digest(state["hooks"]["LRScheduler"]),
            "scaler": ops.state_digest(FakeScaler().state_dict()),
        },
        "reference_A": {"transcripts": {"0": {"path": str(reference)}}},
    }
    monkeypatch.setattr(ops, "verify_pair", lambda current, prior: None)
    cls = ops.make_trainer_class(Native, manifest, 0)
    blocked = cls(blocked_model, batches(), blocked_optimizer)
    blocked.load_state_dict(state)
    native.iter = blocked.iter = ops.START
    native.run_step()
    blocked.run_step()
    assert blocked.last_intervention["dn_norm"] > 0
    torch.testing.assert_close(native_model.other, blocked_model.other, rtol=0, atol=0)
    assert not torch.equal(
        native_model.transformer.decoder.layers[0].weight,
        blocked_model.transformer.decoder.layers[0].weight,
    )
    assert blocked.actual_updates == 1
    blocked.pair_stream.close()
    blocked.update_stream.close()
    blocked.reference_stream.close()


def test_training_options_keep_locked_protocol():
    manifest = {"output_dir": "/tmp/dn", "updates": 500,
                "checkpoint": {"path": "/tmp/8ep.pth"}}
    options = worker.training_options(manifest)
    assert "train.max_iter=57300" in options
    assert "train.lr_scheduler_max_iter=85200" in options
    assert "train.gradient_accumulation_steps=2" in options
    assert "dataloader.train.total_batch_size=16" in options
    assert not any("weight_dict" in value for value in options)


def _ddp_dn_check(rank, rendezvous, directory):
    import torch.distributed as dist
    from contextlib import nullcontext

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=Path(rendezvous).as_uri(),
                            rank=rank, world_size=2)
    try:
        torch.manual_seed(3)
        model = nn.Linear(2, 1)
        ddp = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        parameters = tuple(model.parameters())
        source = None
        for micro in range(2):
            with (ddp.no_sync() if micro == 0 else nullcontext()):
                value = ddp(torch.tensor([[rank+1., micro+1.]])).sum()
                losses = {"loss_class_dn": value/6}
                losses.update({f"loss_class_dn_{i}": value/6 for i in range(5)})
                losses["loss_class"] = value*2
                gradient = ops.dn_classification_gradient(losses, parameters, 2)
                source = gradient if source is None else source+gradient
                (sum(losses.values())/2).backward()
        source = ops.synchronize_auxiliary(source)
        full = torch.cat([parameter.grad.flatten() for parameter in parameters])
        torch.testing.assert_close(source, torch.tensor([1.5, 1.5, 1.]))
        torch.testing.assert_close(full, torch.tensor([4.5, 4.5, 3.]))
        norm = nn.utils.clip_grad_norm_(parameters, .5)
        coefficient = ops.subtract_clipped_auxiliary(
            parameters, source, norm, .5, remove=True)
        remaining = torch.cat([parameter.grad.flatten() for parameter in parameters])
        torch.testing.assert_close(remaining, torch.tensor([3., 3., 2.])*coefficient)
        Path(directory, f"dn_rank{rank}.ok").write_text("ok")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo unavailable")
def test_real_ddp_dn_accumulation_and_reduction(tmp_path):
    torch.multiprocessing.spawn(
        _ddp_dn_check, args=(str(tmp_path/"rendezvous"), str(tmp_path)), nprocs=2, join=True)
    assert all((tmp_path/f"dn_rank{rank}.ok").exists() for rank in range(2))


def test_execute_reuses_a_and_runs_only_b_and_one_evaluation(tmp_path, monkeypatch):
    (tmp_path/"B").mkdir()
    (tmp_path/"B"/"last_checkpoint").write_text("source")
    prediction = tmp_path/"a.json"
    prediction.write_text("[]")
    checkpoint = tmp_path/"a.pth"
    checkpoint.write_text("a")
    manifest = {
        "output_dir": str(tmp_path), "fingerprint": "x", "config": {"path": "cfg.py"},
        "checkpoint": {"path": "source"}, "scope": [],
        "intervention": "dn_classification_to_decoder_core",
        "reference_A": {
            "directory": "/reference", "checkpoint": runner.file_identity(checkpoint),
            "evaluation": {"metrics": {key: 42. for key in runner.METRICS},
                           "predictions": runner.file_identity(prediction)},
        },
    }
    calls = []
    monkeypatch.setattr(runner, "verify_reference_inputs", lambda m: None)
    monkeypatch.setattr(runner, "run_evaluation", lambda cmd, path: calls.append(path.name))
    monkeypatch.setattr(runner, "verify_b", lambda m: {"checkpoint": {"path": "b.pth"}})
    monkeypatch.setattr(runner, "collect_evaluation", lambda path: {
        "metrics": {key: 41. for key in runner.METRICS}, "predictions": {}})
    result = runner.execute(manifest)
    assert calls == ["B", "eval_B"]
    assert result["training"]["A"]["reused"] is True
    assert result["delta_B_minus_A"]["APr"] == -1.


def test_prepare_creates_only_b_and_records_immutable_a(tmp_path, monkeypatch):
    reference = tmp_path/"reference"
    reference.mkdir()
    source_dir = tmp_path/"source"
    source_dir.mkdir()
    source = source_dir/"model_0056799.pth"
    source.write_text("checkpoint")
    config = source_dir/"config.py"
    config.write_text("# locked")
    transcript_identities = {}
    receipts = []
    for rank in range(4):
        path = reference/f"pairing_rank{rank}.jsonl"
        path.write_text("{}\n")
        identity = runner.file_identity(path)
        transcript_identities[str(rank)] = identity
        receipts.append({"rank": rank, "transcript": identity})
    prediction = reference/"predictions.json"
    prediction.write_text("[]")
    checkpoint_a = reference/"model_final.pth"
    checkpoint_a.write_text("A")
    for name in ("manifest.json", "summary.json"):
        (reference/name).write_text("{}")
    prior = {
        "checkpoint": runner.file_identity(source), "config": runner.file_identity(config),
        "resume": {"optimizer": "o", "scheduler": "s", "scaler": "g", "lrs": [1e-4]},
        "model_digest": ops.state_digest({"x": torch.ones(1)}),
        "decoder_keys": ["transformer.decoder.layers.0.weight"],
        "seed": 42, "torch_version": str(torch.__version__), "code": {}, "fingerprint": "old",
    }
    stage = {"checkpoint": runner.file_identity(checkpoint_a), "receipts": receipts}
    evaluation = {"metrics": {key: 42. for key in runner.METRICS},
                  "predictions": runner.file_identity(prediction)}
    monkeypatch.setattr(runner, "validate_reference",
                        lambda path: (reference, prior, {}, stage, evaluation))
    monkeypatch.setattr(runner, "verified_identity", lambda *args: None)
    monkeypatch.setattr(runner, "load_trusted_torch_file", lambda path: {"model": {}})
    monkeypatch.setattr(runner, "endpoint_state", lambda checkpoint, iteration: {"x": torch.ones(1)})
    monkeypatch.setattr(runner.shutil, "disk_usage",
                        lambda path: SimpleNamespace(free=10**12))
    output = tmp_path/"dn_trial"
    args = runner.parse_args([
        "--reference-trial", str(reference), "--output-dir", str(output),
        "--prepare-only",
    ])
    manifest = runner.prepare(args)
    assert (output/"B"/"last_checkpoint").read_text() == str(source)
    assert not (output/"A").exists()
    assert manifest["reference_A"]["transcripts"] == transcript_identities
    assert manifest["reference_A"]["evaluation"] == evaluation


def test_incomplete_b_is_not_resumed(tmp_path, monkeypatch):
    (tmp_path/"B").mkdir()
    (tmp_path/"B"/"last_checkpoint").write_text("source")
    (tmp_path/"B"/"console.log").write_text("failed")
    monkeypatch.setattr(runner, "verify_reference_inputs", lambda m: None)
    monkeypatch.setattr(runner, "run_evaluation", lambda *args: pytest.fail("training launched"))
    with pytest.raises(ValueError, match="Incomplete DN B"):
        runner.execute({"output_dir": str(tmp_path)})


def test_prepare_only_never_executes_gpu(monkeypatch):
    args = runner.parse_args([
        "--reference-trial", "reference", "--output-dir", "output", "--prepare-only"])
    monkeypatch.setattr(runner, "prepare", lambda value: {"prepared": True})
    monkeypatch.setattr(runner, "execute", lambda value: pytest.fail("GPU launched"))
    assert runner.run(args) == {"prepared": True}
