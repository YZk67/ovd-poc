"""Native Trainer updates with real AdamW/AMP/DDP; D2 shell and file IO isolated."""
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tools import apr_projection_extension_ops as ext
from tools import apr_projection_trial_ops as base
from tools import extend_apr_projection_trial as runner
from tools import train_apr_projection_extension as worker
from tools.compare_rare_pr_reports import file_identity
from tools.decoder_aux_ablation_ops import START, HORIZON, state_digest, validate_resume
from tools.gt_normalization_trial_ops import normalization_plan
from test_apr_projection_trial import Model, PROMPTS
from test_decoder_aux_ablation import native_class
from test_gt_normalization_extension import StatefulData, advance


def dirs(root):
    for group in ("continuous", "parent", "extended"):
        for arm in ext.ARMS:
            (root/group/arm).mkdir(parents=True)


def setup(root, patch, rank=0, distributed=False):
    if not hasattr(getattr(torch, "amp", None), "GradScaler"):
        pytest.skip("CPU native-step test requires torch.amp.GradScaler; server worker uses CUDA AMP")
    patch.setattr(torch.cuda, "is_available", lambda: True)
    patch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    patch.setattr(torch.cuda.amp, "autocast", lambda **kw: nullcontext())
    torch.manual_seed(23)
    native = native_class()
    model = Model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sum(p.square().sum() for p in model.parameters()).backward()
    opt.step(); opt.zero_grad()
    kwargs = dict(amp=True, separate_tpa_grad_clip=True,
                  clip_grad_params={"max_norm": .5, "norm_type": 2},
                  lr_scheduler_max_iter=HORIZON, gradient_accumulation_steps=2)
    trainer = native(model, StatefulData(), opt, tpa_conflict_projection=True,
                     grad_scaler=torch.amp.GradScaler("cpu", init_scale=1024.), **kwargs)
    weights, state = deepcopy(model.state_dict()), deepcopy(trainer.state_dict())
    parent = {"output_dir": str(root/"continuous"), "updates": 4, "seed": 42, "fingerprint": "parent",
              "resume": validate_resume({"iteration": START-1, "trainer": state}), "model_digest": state_digest(weights)}

    def build(arm, manifest=None, snapshot=None):
        new = Model()
        new.load_state_dict(weights if snapshot is None else snapshot[0])
        optimizer = torch.optim.AdamW(new.parameters(), lr=1e-4, weight_decay=1e-4)
        if distributed:
            new = nn.parallel.DistributedDataParallel(new, find_unused_parameters=True)
        cls = (base.make_trainer_class(native, parent, arm, rank, PROMPTS) if manifest is None
               else ext.make_trainer_class(native, manifest, arm, rank, PROMPTS))
        t = cls(new, StatefulData(), optimizer, tpa_conflict_projection=ext.ARMS[arm]["projection"],
                grad_scaler=torch.amp.GradScaler("cpu", init_scale=1.), **kwargs)
        t.load_state_dict(deepcopy(state if snapshot is None else snapshot[1]))
        return t

    sources, snapshots, finals = {}, {}, {}
    for arm in ext.ARMS:
        t = build(arm)
        advance(t, 0, 2)
        t.check_health()
        snapshots[arm] = (deepcopy(t.raw_model.state_dict()), deepcopy(t.state_dict()))
        transcript = root/"parent"/arm/f"pairing_rank{rank}.jsonl"
        transcript.write_text((root/"continuous"/arm/f"pairing_rank{rank}.jsonl").read_text())
        if distributed:
            torch.distributed.barrier()
        sources[arm] = {"checkpoint": {"path": str(root/"parent"/arm/"model_final.pth")},
                        "resume": ext.resume_identity(snapshots[arm][1], START+1, parent["resume"]["lrs"]),
                        "model_digest": state_digest(snapshots[arm][0]), "final_rank": t.health_records[-1],
                        "transcripts": [file_identity(root/"parent"/arm/f"pairing_rank{r}.jsonl")
                                        for r in range(4 if distributed else 1)]}
        advance(t, 2, 4)
        finals[arm] = {"weights": deepcopy(t.raw_model.state_dict()),
                       "optimizer": state_digest(t.optimizer.state_dict()),
                       "scaler": deepcopy(t.grad_scaler.state_dict())}
        t.close_streams()
        if distributed:
            torch.distributed.barrier()
    m = {"output_dir": str(root/"extended"), "completed_updates": 2, "total_updates": 4,
         "additional_updates": 2, "start": START+2, "seed": 42,
         "parent_fingerprint": "parent", "fingerprint": "extension", "sources": sources}
    return m, build, snapshots, finals


def test_own_resume_and_data_only_replay_match_uninterrupted_updates(tmp_path, monkeypatch):
    dirs(tmp_path)
    m, build, snapshots, finals = setup(tmp_path, monkeypatch)
    assert m["sources"]["A"]["model_digest"] != m["sources"]["P"]["model_digest"]
    for arm in ext.ARMS:
        t = build(arm, m, snapshots[arm])
        assert t.actual_updates == 2 and t.initial_state["model"] == m["sources"][arm]["model_digest"]
        t.iter = START+2
        original = t.model.forward
        t.model.forward = lambda *a: pytest.fail("Replay must not run a model forward")
        t.replay_cursor()
        t.model.forward = original
        assert t.cursor_replay == ext.cursor_receipt(m, arm, 0)
        assert t.actual_updates == 2 and t.health_records[0] == m["sources"][arm]["final_rank"]
        with pytest.raises(ValueError, match="fresh loader"):
            t.replay_cursor()
        advance(t, 2, 4)
        assert t.actual_updates == 4
        assert state_digest(t.raw_model.state_dict()) == state_digest(finals[arm]["weights"])
        assert state_digest(t.optimizer.state_dict()) == finals[arm]["optimizer"]
        assert t.grad_scaler.state_dict() == finals[arm]["scaler"]
        assert t.tpa_conflict_projection == ext.ARMS[arm]["projection"]
        assert t._get_tpa().lambda_orth_base == .1 and t._get_tpa().lambda_div_base == .03
        assert t.state_dict()["apr_projection_trial"] == ext.trial_tag(m, arm)
        assert t.state_dict()["apr_projection_extension"] == ext.extension_tag(m, 4, t.cursor_replay)
        t.close_streams()
        assert runner.read_rows(tmp_path/"extended"/arm/"pairing_rank0.jsonl") == runner.read_rows(tmp_path/"continuous"/arm/"pairing_rank0.jsonl")[4:]
        assert [r["update"] for r in t.health_records] == [2, 4]


def test_replay_mismatch_stops_before_new_update(tmp_path, monkeypatch):
    dirs(tmp_path)
    m, build, snapshots, _ = setup(tmp_path, monkeypatch)
    path = tmp_path/"parent/A/pairing_rank0.jsonl"
    rows = runner.read_rows(path)
    rows[0]["mapped"][0]["image_id"] += 1
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    t = build("A", m, snapshots["A"])
    t.iter = START+2
    before = state_digest(t.model.state_dict())
    with pytest.raises(ValueError, match="replay mismatch"):
        t.run_step()
    assert t.actual_updates == 2 and t.iter == START+2 and t.cursor_replay is None
    assert state_digest(t.model.state_dict()) == before
    assert state_digest(t.optimizer.state_dict()) == m["sources"]["A"]["resume"]["optimizer"]
    t.close_streams()


def test_cross_arm_or_wrong_lr_resume_and_r_arm_rejected(tmp_path, monkeypatch):
    dirs(tmp_path)
    m, build, snapshots, _ = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="wrong arm"):
        build("A", m, snapshots["P"])
    with pytest.raises(ValueError, match="Only A/P"):
        ext.make_trainer_class(native_class(), m, "R", 0, PROMPTS)
    state = deepcopy(snapshots["A"][1])
    state["optimizer"]["param_groups"][0]["lr"] = 1e-5
    with pytest.raises(ValueError, match="Incomplete endpoint"):
        ext.resume_identity(state, START+1, [1e-4])


def test_extension_options_keep_lr_and_policy_and_use_own_endpoint():
    m = {"output_dir": "/tmp/extended", "total_updates": 4000,
         "sources": {a: {"checkpoint": {"path": f"/tmp/parent/{a}/model_final.pth"},
                         "resume": {}, "model_digest": a} for a in ext.ARMS}}
    for arm in ext.ARMS:
        opts = worker.training_options(m, arm)
        assert f'train.init_checkpoint="/tmp/parent/{arm}/model_final.pth"' in opts
        assert "train.max_iter=60800" in opts and "train.lr_scheduler_max_iter=85200" in opts
        assert "train.checkpointer.period=60801" in opts and "train.gradient_accumulation_steps=2" in opts
        assert "model.classifier.tpa_train_aggregation=calibrated" in opts
        assert "model.classifier.tpa_prototype_mode_strength=0.0" in opts
        assert f"train.tpa_conflict_projection={str(ext.ARMS[arm]['projection']).lower()}" in opts
        assert not any("loss_apr=" in v or "lambda_div=" in v for v in opts)
    with pytest.raises(ValueError, match="Only A/P"):
        worker.training_options(m, "R")


def test_output_and_budget_constraints(tmp_path):
    parent = tmp_path/"parent"
    parent.mkdir()
    args = SimpleNamespace(parent_dir=str(parent), output_dir=str(parent/"inside"),
                           total_updates=4000, num_gpus=4, cpu_threads=2)
    with pytest.raises(ValueError, match="separate"):
        runner.prepare(args)
    args.output_dir = str(tmp_path/"new")
    for total in (2000, 5000):
        args.total_updates = total
        with pytest.raises(ValueError, match="TOTAL"):
            runner.prepare(args)


def mock_pipeline(root, patch, *, fail_arm=None):
    source = root/"source.json"
    source.write_text("{}")
    identity = file_identity(source)
    output = root/"trial"
    health = {"rare": {"mean_rank": 4.2}, "guard_pass": True}
    for arm in ext.ARMS:
        (output/arm).mkdir(parents=True)
        (output/arm/"last_checkpoint").write_text(f"parent_{arm}")
    m = {"output_dir": str(output), "fingerprint": "test", "total_updates": 4000,
         "additional_updates": 2000, "scope": [], "assets": [],
         **{k: identity for k in ("parent_manifest", "parent_summary", "config", "prompt_bank", "train_annotations", "val_annotations")},
         "sources": {a: {"checkpoint": identity, "transcripts": [identity], "final_rank": health} for a in ext.ARMS},
         "metrics_at_2000": {a: {k: 40.+i for k in runner.METRICS} for i, a in enumerate(ext.ARMS)}}
    events = []
    patch.setattr(runner, "read_manifest", lambda path: m)
    patch.setattr(runner, "verify_arm", lambda m, arm: {"checkpoint": identity, "final_rank": health})
    patch.setattr(runner, "collect_evaluation", lambda path: {"metrics": {k: {"eval_A": 41., "eval_P": 43.}[path.name] for k in runner.METRICS}})
    def run(command, directory):
        if "--arm" in command:
            arm = command[command.index("--arm")+1]
            events.append("train_"+arm)
            if arm == fail_arm:
                raise ValueError("simulated guard failure")
        else:
            events.append(directory.name)
    patch.setattr(runner, "run_evaluation", run)
    return m, events


def test_pipeline_evaluates_only_a_p_and_reports_both_horizons(tmp_path, monkeypatch):
    m, events = mock_pipeline(tmp_path, monkeypatch)
    result = runner.execute(m)
    assert events == ["train_A", "eval_A", "train_P", "eval_P"]
    assert result["complete"] and result["delta_P_minus_A"]["APr"] == 2.
    assert result["delta_P_minus_A_at_2000"]["APr"] == 1.
    assert result["change_from_2000"]["A"]["APr"] == 1.
    assert result["change_from_2000"]["P"]["APr"] == 2.
    assert runner.load_json(Path(m["output_dir"])/"STATUS.json")["state"] == "COMPLETE"
    assert not (Path(m["output_dir"])/"R").exists()


def test_failure_preserves_control_result_and_marks_incomplete(tmp_path, monkeypatch):
    m, events = mock_pipeline(tmp_path, monkeypatch, fail_arm="P")
    monkeypatch.setattr(runner, "prepare", lambda args: m)
    with pytest.raises(ValueError, match="guard failure"):
        runner.main(["--parent-dir", "unused", "--output-dir", m["output_dir"]])
    status = runner.load_json(Path(m["output_dir"])/"STATUS.json")
    summary = runner.load_json(Path(m["output_dir"])/"summary.json")
    assert status["state"] == "FAILED" and status["phase"] == "train_P"
    assert not summary["complete"] and set(summary["evaluations"]) == {"A"}
    assert summary["delta_P_minus_A"] is None and "eval_P" not in events


def test_changed_source_and_incomplete_arm_never_start_gpu_work(tmp_path, monkeypatch):
    m, events = mock_pipeline(tmp_path, monkeypatch)
    source = tmp_path/"source.json"
    source.write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        runner.execute(m)
    assert not events
    source.write_text("{}")
    interrupted = Path(m["output_dir"])/"A/console.log"
    interrupted.write_text("old interrupted run")
    with pytest.raises(ValueError, match="Incomplete extension A"):
        runner.execute(m)
    assert not events and interrupted.read_text() == "old interrupted run"


def test_verifier_certifies_cumulative_seeds_replay_and_checkpoint_provenance(tmp_path, monkeypatch):
    m = {"output_dir": str(tmp_path), "completed_updates": 2000, "total_updates": 2001,
         "start": START+2000, "seed": 42, "parent_fingerprint": "parent",
         "fingerprint": "extension", "sources": {}}
    health = [{"update": u, "guard_pass": True} for u in (2000, 2001)]
    plan = normalization_plan([8, 16], 4, 8)
    checkpoints = {}
    for arm in ext.ARMS:
        directory = tmp_path/arm
        directory.mkdir()
        (directory/"model_final.pth").write_bytes(b"mock-checkpoint")
        (directory/"rank_health.jsonl").write_text("".join(json.dumps(r)+"\n" for r in health))
        initial = {"optimizer": arm, "scheduler": "s", "scaler": "g", "model": arm}
        m["sources"][arm] = {"resume": {"optimizer": arm, "scheduler": "s", "scaler": "g", "lrs": [.0001]},
                              "model_digest": arm, "final_rank": health[0],
                              "transcripts": [{"path": f"parent-{arm}-{rank}"} for rank in range(4)]}
        for rank in range(4):
            rows = []
            for micro in range(2):
                seed = 42+2000*128+rank*8+micro*2
                rows.append({"iteration": START+2000, "micro": micro, "normalization": plan,
                             "data_seed": seed, "forward_seed": seed+1, "lrs": [.0001], "multiplier": 1.,
                             "policy": ext.ARMS[arm], "mapped": [rank, micro], "fedloss": [1, 2],
                             "losses": {"loss_apr": .023, "loss_class": float(ord(arm))},
                             "apr_components": {"loss_prototype_diversity": .2, "loss_balance": .1,
                                                "lambda_orth": .1, "lambda_balance": .03}})
            updates = [{"iteration": START+2000, "update": 2001, "arm": arm, "policy": ext.ARMS[arm],
                        "normalization": plan, "preclip_norms": {"detector": 1., "tpa": 2.},
                        "routing": {"enabled": arm == "A"}}]
            transcript, log = directory/f"pairing_rank{rank}.jsonl", directory/f"updates_rank{rank}.jsonl"
            transcript.write_text("".join(json.dumps(r)+"\n" for r in rows))
            log.write_text("".join(json.dumps(r)+"\n" for r in updates))
            runner.save_json(directory/f"complete_rank{rank}.json", {
                "complete": True, "arm": arm, "rank": rank, "policy": ext.ARMS[arm],
                "start": START+2000, "stop": START+2001, "updates": 1, "total_updates": 2001,
                "initial_state": initial, "health_records": health, "cursor_replay": ext.cursor_receipt(m, arm, rank),
                "manifest_fingerprint": "extension", "transcript": file_identity(transcript), "update_log": file_identity(log)})
        checkpoints[arm] = {"iteration": START+2000, "trainer": {
            "apr_projection_trial": ext.trial_tag(m, arm),
            "apr_projection_extension": ext.extension_tag(m, 2001, ext.cursor_receipt(m, arm, 0))}}
    monkeypatch.setattr(runner, "load_trusted_torch_file", lambda path: checkpoints[Path(path).parent.name])
    def endpoint(checkpoint, iteration):
        assert checkpoint["iteration"] == iteration
    monkeypatch.setattr(runner, "endpoint_state", endpoint)
    monkeypatch.setattr(runner, "resume_identity", lambda *a: None)  # Real AdamW restoration is tested above.
    for arm in ext.ARMS:
        assert runner.verify_arm(m, arm)["final_rank"] == health[-1]
    checkpoints["P"]["trainer"]["apr_projection_extension"]["completed_updates"] = 0
    with pytest.raises(ValueError, match="provenance"):
        runner.verify_arm(m, "P")
    checkpoints["P"]["trainer"]["apr_projection_extension"]["completed_updates"] = 2000
    path = tmp_path/"P/pairing_rank3.jsonl"
    rows = runner.read_rows(path)
    rows[0]["data_seed"] = 42
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    receipt_path = tmp_path/"P/complete_rank3.json"
    receipt = runner.load_json(receipt_path)
    receipt["transcript"] = file_identity(path)
    runner.save_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="timeline"):
        runner.verify_arm(m, "P")


def test_manifest_rejects_changed_policy_budget_code_and_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    source = tmp_path/"source.json"
    source.write_text("{}")
    code = tmp_path/"train.py"
    code.write_text("# verified\n")
    identity = file_identity(source)
    m = {"schema": "apr_projection_extension_v1", "arms": deepcopy(ext.ARMS), "completed_updates": 2000,
         "total_updates": 4000, "additional_updates": 2000, "start": START+2000, "seed": 42,
         "num_gpus": 4, "lr_horizon": HORIZON, "rank_guard": worker.RANK_GUARD, "rank_period": worker.RANK_PERIOD,
         "sources": {a: {} for a in ext.ARMS}, "cpu_threads": 2, "torch_version": str(torch.__version__),
         "code": {"train.py": file_identity(code)["sha256"]},
         **{k: identity for k in ("config", "prompt_bank", "parent_manifest", "parent_summary")}}
    path = tmp_path/"manifest.json"
    def save():
        m["fingerprint"] = runner.fingerprint({k: v for k, v in m.items() if k != "fingerprint"})
        runner.save_json(path, m)
    save()
    assert worker.read_manifest(path) == m
    m["total_updates"] = 5000
    save()
    with pytest.raises(ValueError, match="protocol"):
        worker.read_manifest(path)
    m["total_updates"] = 4000
    m["arms"]["P"]["barrier_weight"] = 0
    save()
    with pytest.raises(ValueError, match="protocol"):
        worker.read_manifest(path)
    m["arms"] = deepcopy(ext.ARMS)
    save()
    code.write_text("# changed\n")
    with pytest.raises(ValueError, match="code changed"):
        worker.read_manifest(path)
    code.write_text("# verified\n")
    source.write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        worker.read_manifest(path)


def _distributed(rank, rendezvous, directory):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=4)
    patch = pytest.MonkeyPatch()
    try:
        root = Path(directory)
        m, build, snapshots, finals = setup(root, patch, rank, distributed=True)
        for arm in ext.ARMS:
            t = build(arm, m, snapshots[arm])
            advance(t, 2, 4)
            assert t.actual_updates == 4 and t.cursor_replay == ext.cursor_receipt(m, arm, rank)
            for key, value in t.raw_model.state_dict().items():
                torch.testing.assert_close(value, finals[arm]["weights"][key], rtol=1e-6, atol=1e-7)
            assert t.grad_scaler.state_dict() == finals[arm]["scaler"]
            hashes = [None]*4
            dist.all_gather_object(hashes, state_digest(t.raw_model.state_dict()))
            assert len(set(hashes)) == 1
            t.close_streams()
            dist.barrier()
        (root/f"rank{rank}.ok").write_text("A/P continuation verified")
    finally:
        patch.undo()
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo unavailable")
@pytest.mark.skipif(not hasattr(getattr(torch, "amp", None), "GradScaler"), reason="CPU AMP unavailable")
def test_four_rank_resume_matches_continuous_paired_training(tmp_path):
    dirs(tmp_path)
    torch.multiprocessing.spawn(_distributed, args=(str(tmp_path/"rendezvous"), str(tmp_path)), nprocs=4, join=True)
    assert all((tmp_path/f"rank{r}.ok").exists() for r in range(4))
