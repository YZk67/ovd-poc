from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tools import gt_normalization_trial_ops as ops
from tools import run_gt_normalization_trial as runner
from tools import train_gt_normalization_arm as worker
from tools.decoder_aux_ablation_ops import validate_resume, state_digest
from test_decoder_aux_ablation import native_class


def test_reweight_only_audited_detection_losses_without_mutating_forward():
    p = nn.Parameter(torch.tensor(2.))
    detection = ["loss_class", "loss_bbox", "loss_giou", "loss_class_0", "loss_bbox_enc",
                 "loss_giou_dn", "loss_class_dn_4"]
    losses = {key: p.square() for key in detection}
    losses.update(loss_apr=p*3, loss_rpsa=p*5, metric=torch.tensor(9.))
    a, b = ops.reweight_losses(losses, 1.), ops.reweight_losses(losses, 1.3)
    assert all(a[k] is losses[k] for k in losses)
    for k in ("loss_apr", "loss_rpsa", "metric"):
        assert b[k] is losses[k]
    for k in detection:
        torch.testing.assert_close(b[k], losses[k]*1.3)
        assert losses[k].item() == 4
    grad = torch.autograd.grad(sum(v for k,v in b.items() if k.startswith("loss")), p)[0]
    torch.testing.assert_close(grad, torch.tensor(4*len(detection)*1.3+8))
    with pytest.raises(ValueError, match="Unknown"):
        ops.reweight_losses({**losses, "loss_new": p}, 1.3)
    for bad in (0., float("nan"), -1.):
        with pytest.raises(ValueError):
            ops.reweight_losses(losses, bad)


@pytest.mark.parametrize("counts", [[0,0], [0,200], [1,1], [366,200], [250,124]])
def test_pooled_algebra_includes_empty_clamp_and_ddp_mean(counts):
    plan = ops.normalization_plan(counts, 4, 8)
    numerators = torch.tensor([17., 29.])
    actual = sum(numerator / denom * factor / 2 for numerator,denom,factor in zip(
        numerators, plan["micro_global_denominators"], plan["detection_loss_multipliers"]))
    torch.testing.assert_close(actual, numerators.sum()/max(sum(counts), 8))


def record(iteration=ops.START):
    return {"iteration": iteration, "micro": 0, "mapped": [1], "fedloss": [1,2],
            "lrs": [.001], "amp_scale": 1024., "rng_after": {"cpu": "same"},
            "normalization": {"counts": [8,16]}, "loss_keys": ["loss_class"],
            "losses": {"loss_class": 1.}, "multiplier": 1.}


def test_pair_checks_raw_initial_losses_but_allows_later_learning():
    a = record()
    ops.verify_pair({**a, "multiplier": 1.2}, a)
    with pytest.raises(ValueError, match="First-window"):
        ops.verify_pair({**a, "losses": {"loss_class": 2.}}, a)
    a = record(ops.START+1)
    ops.verify_pair({**a, "losses": {"loss_class": 2.}}, a)
    for key,value in (("mapped", [2]), ("fedloss", [1,3]), ("amp_scale", 2048.),
                      ("lrs", [.002]), ("rng_after", {}), ("normalization", {})):
        with pytest.raises(ValueError, match="Unpaired"):
            ops.verify_pair({**a, key:value}, a)


class TPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.prototype_queries = nn.Parameter(torch.eye(5))
        self.key_proj = nn.Linear(5,5)
        self.value_proj = nn.Linear(5,5)
        with torch.no_grad():
            self.key_proj.weight.copy_(torch.eye(5))
            self.value_proj.weight.copy_(torch.eye(5))
            self.key_proj.bias.zero_()
            self.value_proj.bias.zero_()
        self.register_buffer("slot_prior_strength", torch.tensor(.2))
        self.register_buffer("prototype_mode_strength", torch.tensor(0.))


PROMPTS = torch.eye(5).unsqueeze(0).expand(5,-1,-1).clone()


def test_health_reconstruction_is_rng_and_state_neutral_and_catches_collapse():
    tpa = TPA()
    mask = torch.tensor([False,False,False,True,True])
    before, rng = state_digest(tpa.state_dict()), torch.get_rng_state().clone()
    health = ops.bank_health(tpa.state_dict(), PROMPTS, mask)
    assert health["guard_pass"]
    assert health["all"]["mean_rank"] == pytest.approx(5, abs=1e-5)
    assert state_digest(tpa.state_dict()) == before
    assert torch.equal(rng, torch.get_rng_state())
    with torch.no_grad():
        tpa.value_proj.weight.zero_()
        tpa.value_proj.bias.fill_(1)
    health = ops.bank_health(tpa.state_dict(), PROMPTS, mask)
    assert not health["guard_pass"]
    assert health["rare"]["rank_below_2_count"] == 2


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        dec = self.transformer.decoder = nn.Module()
        dec.layers = nn.ModuleList([nn.Linear(2,1)])
        head = nn.Module()
        head.tpa = TPA()
        head.tpa_train_aggregation = "calibrated"
        dec.class_embed = nn.ModuleList([head, head])
        self.novel_idx = torch.tensor([False,False,False,True,True])

    def filter_content_info(self, data):
        return torch.randperm(3), data

    def forward(self, data):
        self.filter_content_info(data)
        y = self.transformer.decoder.layers[0](torch.stack([d["image"] for d in data]).mean(0)).sum()
        tpa = self.transformer.decoder.class_embed[0].tpa
        z = sum(p.sum()*.01 for p in tpa.parameters())
        denominator = max(sum(len(d["instances"].gt_classes) for d in data)/4, 1.)
        # Shared detector/TPA paths, independent APR/RPSA; native routing and
        # BOTH clipping groups are exercised, unlike a scalar loss-only test.
        loss = (y+z-torch.rand(()))**2/denominator
        return {"loss_class": loss, "loss_bbox": (y-z)**2/denominator,
                "loss_giou": (y+1)**2/denominator, "loss_class_0": loss*.2,
                "loss_class_dn": loss*.1, "loss_bbox_enc": loss*.15,
                "loss_apr": sum(p.square().sum() for p in tpa.parameters())*.01,
                "loss_rpsa": (y+z).square()*.01}


class Data:
    def __iter__(self):
        index = 0
        while True:
            count = 1 if index%2 == 0 else 3
            index += 1
            yield [{"image_id": i, "image": torch.rand(2), "instances": SimpleNamespace(
                gt_classes=torch.zeros(count,dtype=torch.long),
                gt_boxes=SimpleNamespace(tensor=torch.rand(count,4)))} for i in range(4)]


def build_experiment(tmp_path, monkeypatch, *, rank=0, distributed=False):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda.amp, "autocast", lambda **kw: nullcontext())
    torch.manual_seed(23)
    native = native_class()
    model = Model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sum(p.square().sum() for p in model.parameters()).backward()
    opt.step()
    opt.zero_grad()
    kwargs = dict(amp=True, clip_grad_params={"max_norm": .5, "norm_type": 2},
                  separate_tpa_grad_clip=True, tpa_conflict_projection=True,
                  lr_scheduler_max_iter=ops.HORIZON, gradient_accumulation_steps=2)
    trainer = native(model, Data(), opt, grad_scaler=torch.amp.GradScaler("cpu", init_scale=1024.), **kwargs)
    state, model_state = deepcopy(trainer.state_dict()), deepcopy(model.state_dict())
    manifest = {"output_dir": str(tmp_path), "updates": 2, "seed": 42, "fingerprint": "toy",
                "model_digest": state_digest(model_state),
                "resume": validate_resume({"iteration": ops.START-1, "trainer": state})}
    if not distributed:
        for arm in ops.ARMS:
            (tmp_path/arm).mkdir()
    def build(arm, intervention=True):
        new = Model()
        new.load_state_dict(model_state)
        opt = torch.optim.AdamW(new.parameters(), lr=1e-4, weight_decay=1e-4)
        if distributed:
            new = nn.parallel.DistributedDataParallel(new, find_unused_parameters=True)
        cls = ops.make_trainer_class(native, manifest, arm, rank, PROMPTS) if intervention else native
        t = cls(new, Data(), opt, grad_scaler=torch.amp.GradScaler("cpu", init_scale=1.), **kwargs)
        t.load_state_dict(deepcopy(state))
        return t
    return manifest, build, model_state


def test_actual_native_training_step_pairs_restores_and_keeps_tpa_trainable(tmp_path, monkeypatch):
    manifest, build, initial = build_experiment(tmp_path, monkeypatch)
    a = build("A")
    for i in range(2):
        a.iter = ops.START+i
        a.run_step()
    a.close_streams()
    b = build("B")
    for i in range(2):
        b.iter = ops.START+i
        b.run_step()
    b.close_streams()
    assert a.initial_state == b.initial_state
    assert a.actual_updates == b.actual_updates == 2
    a_rows = runner.read_rows(tmp_path/"A/pairing_rank0.jsonl")
    b_rows = runner.read_rows(tmp_path/"B/pairing_rank0.jsonl")
    assert len(a_rows) == len(b_rows) == 4
    for left,right in zip(a_rows,b_rows):
        ops.verify_pair(right, left)
    assert [r["multiplier"] for r in b_rows] == [.5,1.5,.5,1.5]
    assert len(a.health_records) == len(b.health_records) == 2
    assert all(r["guard_pass"] for r in a.health_records+b.health_records)
    tpa_name = "transformer.decoder.class_embed.0.tpa.prototype_queries"
    for t in (a,b):
        assert not torch.equal(t.model.state_dict()[tpa_name], initial[tpa_name])
        assert all(int(s["step"]) == 3 for s in t.optimizer.state.values())
        assert t.state_dict()["gt_normalization_trial"]["updates"] == 2
    assert any(not torch.equal(v, b.model.state_dict()[k]) for k,v in a.model.state_dict().items())


def test_control_is_exact_native_step_on_identical_inputs_and_rng(tmp_path, monkeypatch):
    _, build, _ = build_experiment(tmp_path, monkeypatch)
    a = build("A")
    reference = build("A", intervention=False)
    a.iter = reference.iter = ops.START
    a.run_step()
    # Native reference iterator maps and forwards under the same explicit seeds;
    # no reweighting, no copied implementation of native backward/routing/clip.
    data = []
    source = iter(Data())
    for micro in range(2):
        with ops.isolated_rng(42+micro*2):
            data.append(next(source))
    reference._data_loader_iter_obj = iter(data)
    forward = reference.model.forward
    step = [0]
    def paired_forward(batch):
        with ops.isolated_rng(43+2*step[0]):
            result = forward(batch)
        step[0] += 1
        return result
    reference.model.forward = paired_forward
    reference.run_step()
    for key,value in a.model.state_dict().items():
        torch.testing.assert_close(value, reference.model.state_dict()[key], rtol=0, atol=0)
    assert state_digest(a.optimizer.state_dict()) == state_digest(reference.optimizer.state_dict())
    a.close_streams()


def test_global_count_reduction_occurs_before_either_forward(tmp_path, monkeypatch):
    _, build, _ = build_experiment(tmp_path, monkeypatch)
    a = build("A")
    a.iter = ops.START
    calls = []
    monkeypatch.setattr(ops.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(ops.dist, "get_world_size", lambda: 4)
    def reduce(tensor):
        calls.append(tensor.tolist())
        tensor.add_(torch.tensor([8,4]))
    monkeypatch.setattr(ops.dist, "all_reduce", reduce)
    a._prefetch_window()
    assert calls == [[4,12]]
    assert a.plan["global_gt_counts"] == [12,16]
    assert a._micro == 0
    a.close_streams()


def test_health_failure_aborts_without_update(tmp_path, monkeypatch):
    _, build, _ = build_experiment(tmp_path, monkeypatch)
    a = build("A")
    a.iter = ops.START
    monkeypatch.setattr(ops, "bank_health", lambda *a: {"guard_pass":False})
    with pytest.raises(ValueError, match="health guard failed"):
        a.run_step()
    assert a.actual_updates == 0
    assert not (tmp_path/"A/complete_rank0.json").exists()
    a.close_streams()


def test_options_leave_formula_no_radius_lr_and_inference_identical():
    m = {"output_dir":"/tmp/trial", "checkpoint":{"path":"/tmp/8ep.pth"}, "updates":500}
    a,b = [worker.training_options(m,arm) for arm in ops.ARMS]
    assert [x for x in a if "output_dir=" not in x] == [x for x in b if "output_dir=" not in x]
    for option in ("train.lr_scheduler_max_iter=85200", "train.max_iter=57300",
                   "train.checkpointer.period=57301",
                   "train.gradient_accumulation_steps=2", "dataloader.train.total_batch_size=16",
                   "model.classifier.tpa_prototype_mode_strength=0.0",
                   "model.classifier.tpa_train_aggregation=calibrated", "train.eval_after_train=False"):
        assert option in a
    assert all((runner.ROOT/path).is_file() for path in runner.CODE_FILES)


def test_failed_evaluation_is_not_accepted_as_success(tmp_path):
    (tmp_path/"console.log").write_text("Skipping evaluation\n")
    with pytest.raises(RuntimeError):
        runner.collect_evaluation(tmp_path)


def test_incomplete_or_mutating_source_audit_rejected():
    base = {"complete":True, "live_optimizer_steps":0, "model_state_unchanged":True,
            "live_optimizer_state_unchanged":True, "source_checkpoint_unchanged":True,
            "decision":"CONDITIONAL_ONE_STEP_DIFFERENCES_NOT_APR_ATTRIBUTION", "inputs":{},
            "protocol":{"iteration":ops.START}, "windows":[{"paired_native_forward_verified":True}]}
    base["fingerprint"] = hashlib.sha256(json.dumps({},sort_keys=True).encode()).hexdigest()
    runner.validate_update_report(base)
    for key,value in (("complete",False), ("live_optimizer_steps",1), ("model_state_unchanged",False),
                      ("fingerprint","changed"), ("windows",[])):
        with pytest.raises(ValueError):
            runner.validate_update_report({**base,key:value})


def test_existing_output_is_rejected_before_loading_or_launching(tmp_path):
    args = SimpleNamespace(num_gpus=4, seed=42, updates=500, cpu_threads=2,
                           output_dir=str(tmp_path), update_audit="must-not-open")
    with pytest.raises(ValueError, match="NEW output"):
        runner.prepare(args)


def test_receipts_certify_updates_final_checkpoint_health_and_pairs(tmp_path, monkeypatch):
    m = {"output_dir":str(tmp_path), "updates":1, "fingerprint":"test",
         "resume":{"optimizer":"o", "scheduler":"s", "scaler":"g", "lrs":[.001]},
         "model_digest":"m"}
    initial = {"optimizer":"o", "scheduler":"s", "scaler":"g", "model":"m"}
    health = [{"update":i, "guard_pass":True} for i in (0,1)]
    plan = ops.normalization_plan([8,16])
    saved = {}
    for arm in ops.ARMS:
        directory = tmp_path/arm
        directory.mkdir()
        (directory/"model_final.pth").write_bytes(b"mock-checkpoint")
        (directory/"rank_health.jsonl").write_text("".join(json.dumps(r)+"\n" for r in health))
        for rank in range(4):
            rows = [{**record(), "micro":i, "normalization":plan,
                     "multiplier":plan["detection_loss_multipliers"][i] if arm == "B" else 1.}
                    for i in range(2)]
            updates = [{"iteration":ops.START, "update":1, "arm":arm,
                        "normalization":plan, "preclip_norms":{"detector":1.,"tpa":2.}}]
            transcript, log = directory/f"pairing_rank{rank}.jsonl", directory/f"updates_rank{rank}.jsonl"
            transcript.write_text("".join(json.dumps(r)+"\n" for r in rows))
            log.write_text("".join(json.dumps(r)+"\n" for r in updates))
            runner.save_json(directory/f"complete_rank{rank}.json", {
                "complete":True, "arm":arm, "rank":rank, "start":ops.START, "stop":ops.START+1,
                "updates":1, "initial_state":initial, "manifest_fingerprint":"test", "health_records":health,
                "transcript":runner.file_identity(transcript), "update_log":runner.file_identity(log)})
        saved[arm] = {"iteration":ops.START, "trainer":{
            "iteration":ops.START, "lr_scheduler_max_iter":ops.HORIZON, "gradient_accumulation_steps":2,
            "hooks":{"LRScheduler":{"last_epoch":ops.START+1}}, "optimizer":{"state":{0:"moments"}},
            "grad_scaler":{"scale":1024}, "gt_normalization_trial":{
                "arm":arm,"normalization":ops.ARMS[arm],"start":ops.START,"updates":1,"manifest_fingerprint":"test"}}}
    monkeypatch.setattr(runner,"load_trusted_torch_file",lambda p: saved[Path(p).parent.name])
    def endpoint(checkpoint, iteration):
        assert checkpoint["iteration"] == iteration
    monkeypatch.setattr(runner,"endpoint_state",endpoint)
    for arm in ops.ARMS:
        result = runner.verify_arm(m,arm)
        assert result["final_rank"] == health[-1]
    saved["B"]["trainer"]["lr_scheduler_max_iter"] = ops.START+1
    with pytest.raises(ValueError, match="Final checkpoint"):
        runner.verify_arm(m,"B")
    saved["B"]["trainer"]["lr_scheduler_max_iter"] = ops.HORIZON
    receipt_path = tmp_path/"B/complete_rank3.json"
    receipt = runner.load_json(receipt_path)
    receipt["initial_state"]["optimizer"] = "wrong"
    runner.save_json(receipt_path,receipt)
    with pytest.raises(ValueError, match="receipt"):
        runner.verify_arm(m,"B")


def test_incomplete_training_directory_cannot_silently_resume(tmp_path, monkeypatch):
    source = tmp_path/"source.json"
    source.write_text("{}")
    identity = runner.file_identity(source)
    output = tmp_path/"trial"
    (output/"A").mkdir(parents=True)
    (output/"A/last_checkpoint").write_text("unchanged")
    (output/"A/console.log").write_text("interrupted")
    m = {"output_dir":str(output), "checkpoint":identity, "update_audit":identity,
         "objective_audit":identity, "train_annotations":identity, "val_annotations":identity, "assets":[]}
    monkeypatch.setattr(runner,"run_evaluation",lambda *a: pytest.fail("Must not launch GPU work"))
    with pytest.raises(ValueError, match="Incomplete A"):
        runner.execute(m)


def _distributed_trial(rank, rendezvous, directory):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=4)
    patch = pytest.MonkeyPatch()
    try:
        manifest, build, _ = build_experiment(Path(directory), patch, rank=rank, distributed=True)
        for arm in ops.ARMS:
            trainer = build(arm)
            for i in range(2):
                trainer.iter = ops.START+i
                trainer.run_step()
                assert trainer.plan["global_gt_counts"] == [16,48]
            trainer.close_streams()
            hashes = [None]*4
            dist.all_gather_object(hashes, state_digest(trainer.raw_model.state_dict()))
            assert len(set(hashes)) == 1
            assert trainer.actual_updates == 2
            assert all(row["guard_pass"] for row in trainer.health_records)
            dist.barrier()
        Path(directory, f"rank{rank}.ok").write_text("paired DDP passed")
    finally:
        patch.undo()
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo unavailable")
def test_real_four_rank_native_trial_pairs_counts_backward_routing_and_health(tmp_path):
    for arm in ops.ARMS:
        (tmp_path/arm).mkdir()
    torch.multiprocessing.spawn(_distributed_trial,
        args=(str(tmp_path/"rendezvous"),str(tmp_path)), nprocs=4, join=True)
    assert all((tmp_path/f"rank{rank}.ok").exists() for rank in range(4))
