from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tools import gt_normalization_extension_ops as ext
from tools import extend_gt_normalization_trial as runner
from tools import train_gt_normalization_extension as worker
from tools import gt_normalization_trial_ops as base
from tools.compare_rare_pr_reports import file_identity
from tools.decoder_aux_ablation_ops import START, HORIZON, state_digest, validate_resume
from test_decoder_aux_ablation import native_class
from test_gt_normalization_trial import Model, PROMPTS


class StatefulData:
    """Ordering changes both inputs/IDs and must NOT restart at extension."""
    def __iter__(self):
        generator = torch.Generator().manual_seed(741)
        index = 0
        while True:
            private = int(torch.randint(0,100000,(),generator=generator))
            count = 1 if index%2 == 0 else 3
            index += 1
            yield [{"image_id":private+i, "image":torch.rand(2)+index*.01,
                    "instances":SimpleNamespace(gt_classes=torch.zeros(count,dtype=torch.long),
                                                 gt_boxes=SimpleNamespace(tensor=torch.rand(count,4)))} for i in range(4)]


def inputs(tmp_path, patch, rank=0, distributed=False):
    patch.setattr(torch.cuda,"is_available",lambda:True)
    patch.setattr(torch.cuda,"is_current_stream_capturing",lambda:False)
    patch.setattr(torch.cuda.amp,"autocast",lambda **kw:nullcontext())
    torch.manual_seed(23)
    native = native_class()
    model = Model()
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    sum(p.square().sum() for p in model.parameters()).backward()
    optimizer.step()
    optimizer.zero_grad()
    kwargs = dict(amp=True,clip_grad_params={"max_norm":.5,"norm_type":2},separate_tpa_grad_clip=True,
                  tpa_conflict_projection=True,lr_scheduler_max_iter=HORIZON,gradient_accumulation_steps=2)
    initial = native(model,StatefulData(),optimizer,grad_scaler=torch.amp.GradScaler("cpu",init_scale=1024.),**kwargs)
    state,weights = deepcopy(initial.state_dict()),deepcopy(model.state_dict())
    parent = {"output_dir":str(tmp_path/"continuous"),"updates":4,"seed":42,"fingerprint":"parent",
              "resume":validate_resume({"iteration":START-1,"trainer":state}),"model_digest":state_digest(weights)}

    def build(arm, manifest=None, model_state=None, trainer_state=None):
        new = Model()
        new.load_state_dict(weights if model_state is None else model_state)
        opt = torch.optim.AdamW(new.parameters(),lr=1e-4,weight_decay=1e-4)
        if distributed:
            new = nn.parallel.DistributedDataParallel(new,find_unused_parameters=True)
        cls = (base.make_trainer_class(native,parent,arm,rank,PROMPTS) if manifest is None
               else ext.make_trainer_class(native,manifest,arm,rank,PROMPTS))
        trainer = cls(new,StatefulData(),opt,grad_scaler=torch.amp.GradScaler("cpu",init_scale=1.),**kwargs)
        trainer.load_state_dict(deepcopy(state if trainer_state is None else trainer_state))
        return trainer

    return parent,build


def advance(trainer, begin, end):
    for update in range(begin,end):
        trainer.iter = START+update
        trainer.run_step()
        # D2's native after_step LR hook advances this value; LR is constant on
        # the high-LR plateau. The test shell has no hook executor.
        trainer.hooks["LRScheduler"]["last_epoch"] = START+update+1


def dirs(path):
    for group in ("continuous","parent","extended"):
        for arm in base.ARMS:
            (path/group/arm).mkdir(parents=True)


def paired_setup(tmp_path,patch,rank=0,distributed=False):
    parent,build = inputs(tmp_path,patch,rank,distributed)
    snapshots,sources,finals = {},{},{}
    for arm in base.ARMS:
        trainer = build(arm)
        advance(trainer,0,2)
        saved_state = deepcopy(trainer.state_dict())
        weights = deepcopy(trainer.raw_model.state_dict())
        transcript = tmp_path/"parent"/arm/f"pairing_rank{rank}.jsonl"
        transcript.write_text((tmp_path/"continuous"/arm/f"pairing_rank{rank}.jsonl").read_text())
        # Each distributed rank needs metadata for its own source transcript.
        if distributed:
            torch.distributed.barrier()
        transcript_ids = [file_identity(tmp_path/"parent"/arm/f"pairing_rank{r}.jsonl")
                          for r in range(4 if distributed else 1)]
        sources[arm] = {"checkpoint":{"path":str(tmp_path/"parent"/arm/"model_final.pth")},
                        "resume":ext.resume_identity(saved_state,START+1,parent["resume"]["lrs"]),
                        "model_digest":state_digest(weights),"transcripts":transcript_ids}
        snapshots[arm] = (weights,saved_state)
        advance(trainer,2,4)
        finals[arm] = (deepcopy(trainer.raw_model.state_dict()),state_digest(trainer.optimizer.state_dict()),
                       deepcopy(trainer.grad_scaler.state_dict()))
        trainer.close_streams()
        if distributed:
            torch.distributed.barrier()
    m = {"output_dir":str(tmp_path/"extended"),"completed_updates":2,"total_updates":4,"seed":42,
         "additional_updates":2,"start":START+2,"parent_fingerprint":"parent","fingerprint":"extension",
         "sources":sources}
    return m,build,snapshots,finals


def test_restart_plus_data_only_replay_matches_uninterrupted_native_updates(tmp_path,monkeypatch):
    dirs(tmp_path)
    m,build,snapshots,finals = paired_setup(tmp_path,monkeypatch)
    for arm in base.ARMS:
        trainer = build(arm,m,*snapshots[arm])
        assert trainer.actual_updates == 2
        trainer.iter = START+2
        # No forward is allowed during cursor reconstruction.
        original = trainer.model.forward
        trainer.model.forward = lambda *a: pytest.fail("Replay must not run the model")
        trainer.replay_cursor()
        trainer.model.forward = original
        assert trainer.actual_updates == 2
        assert trainer.cursor_replay["state_unchanged"]
        assert trainer.cursor_replay["microbatches"] == 4
        advance(trainer,2,4)
        assert trainer.actual_updates == 4
        for key,value in trainer.raw_model.state_dict().items():
            torch.testing.assert_close(value,finals[arm][0][key],rtol=0,atol=0)
        assert state_digest(trainer.optimizer.state_dict()) == finals[arm][1]
        assert trainer.grad_scaler.state_dict() == finals[arm][2]
        assert trainer.state_dict()["gt_normalization_extension"]["additional_updates"] == 2
        trainer.close_streams()
        rows = runner.read_rows(tmp_path/"extended"/arm/"pairing_rank0.jsonl")
        continuous = runner.read_rows(tmp_path/"continuous"/arm/"pairing_rank0.jsonl")
        assert rows == continuous[4:]
        assert [r["update"] for r in trainer.health_records] == [2,4]


def test_changed_data_replay_aborts_before_optimizer_update(tmp_path,monkeypatch):
    dirs(tmp_path)
    m,build,snapshots,_ = paired_setup(tmp_path,monkeypatch)
    path = tmp_path/"parent/A/pairing_rank0.jsonl"
    rows = runner.read_rows(path)
    rows[0]["mapped"][0]["image_id"] += 1
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    trainer = build("A",m,*snapshots["A"])
    trainer.iter = START+2
    before = state_digest(trainer.model.state_dict())
    with pytest.raises(ValueError,match="replay mismatch"):
        trainer.run_step()
    assert trainer.actual_updates == 2 and trainer.iter == START+2
    assert state_digest(trainer.model.state_dict()) == before
    assert trainer.cursor_replay is None
    trainer.close_streams()


def test_wrong_arm_state_and_wrong_lr_are_rejected(tmp_path,monkeypatch):
    dirs(tmp_path)
    m,build,snapshots,_ = paired_setup(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match="wrong arm"):
        build("A",m,*snapshots["B"])
    state = deepcopy(snapshots["A"][1])
    state["optimizer"]["param_groups"][0]["lr"] = 1e-5
    with pytest.raises(ValueError,match="Incomplete endpoint"):
        ext.resume_identity(state,START+1,[1e-4])
    state = deepcopy(snapshots["A"][1])
    state["hooks"]["LRScheduler"]["last_epoch"] = START
    with pytest.raises(ValueError,match="Incomplete endpoint"):
        ext.resume_identity(state,START+1,[1e-4])


def test_extension_options_use_own_endpoints_original_horizon_and_total_stop():
    m = {"output_dir":"/tmp/extended","total_updates":2000,
         "sources":{a:{"checkpoint":{"path":f"/tmp/parent/{a}/model_final.pth"},
                       "resume":{},"model_digest":a} for a in base.ARMS}}
    for arm in base.ARMS:
        opts = worker.training_options(m,arm)
        assert f'train.init_checkpoint="/tmp/parent/{arm}/model_final.pth"' in opts
        assert "train.max_iter=58800" in opts
        assert "train.checkpointer.period=58801" in opts
        assert "train.lr_scheduler_max_iter=85200" in opts
        assert "train.gradient_accumulation_steps=2" in opts
        assert "model.classifier.tpa_train_aggregation=calibrated" in opts
        assert "model.classifier.tpa_prototype_mode_strength=0.0" in opts


def test_preparation_cannot_overwrite_parent_or_use_unbounded_budget(tmp_path):
    parent = tmp_path/"parent"
    parent.mkdir()
    args = SimpleNamespace(parent_dir=str(parent),output_dir=str(parent/"inside"),
                           total_updates=2000,num_gpus=4,cpu_threads=2)
    with pytest.raises(ValueError,match="separate"):
        runner.prepare(args)
    args.output_dir = str(tmp_path/"new")
    args.total_updates = 10000
    with pytest.raises(ValueError,match="TOTAL"):
        runner.prepare(args)


def test_extension_receipts_check_cursor_seeds_counts_and_checkpoint_provenance(tmp_path,monkeypatch):
    # One new update beyond an existing 500, with all four rank receipts.
    m = {"output_dir":str(tmp_path),"completed_updates":500,"total_updates":501,
         "start":START+500,"seed":42,"parent_fingerprint":"parent","fingerprint":"extended",
         "sources":{}}
    health = [{"update":u,"guard_pass":True} for u in (500,501)]
    plan = base.normalization_plan([8,16],4,8)
    saved = {}
    for arm in base.ARMS:
        folder = tmp_path/arm
        folder.mkdir()
        (folder/"model_final.pth").write_bytes(b"mock-checkpoint")
        (folder/"rank_health.jsonl").write_text("".join(json.dumps(row)+"\n" for row in health))
        m["sources"][arm] = {"resume":{"optimizer":arm,"scheduler":"s","scaler":"g","lrs":[.0001]},
                              "model_digest":arm,"transcripts":[{"path":f"parent-{arm}-{r}"} for r in range(4)]}
        for rank in range(4):
            rows = []
            for micro in range(2):
                seed = 42+500*128+rank*8+micro*2
                rows.append({"iteration":START+500,"micro":micro,"normalization":plan,
                             "data_seed":seed,"forward_seed":seed+1,"lrs":[.0001],
                             "multiplier":plan["detection_loss_multipliers"][micro] if arm == "B" else 1.,
                             "mapped":[rank,micro],"fedloss":[1,2,3],"losses":{"loss_class":float(ord(arm))}})
            updates = [{"iteration":START+500,"update":501,"arm":arm,"normalization":plan,
                        "preclip_norms":{"detector":1.,"tpa":2.}}]
            transcript, log = folder/f"pairing_rank{rank}.jsonl", folder/f"updates_rank{rank}.jsonl"
            transcript.write_text("".join(json.dumps(row)+"\n" for row in rows))
            log.write_text("".join(json.dumps(row)+"\n" for row in updates))
            replay = {"verified":True,"windows":500,"microbatches":1000,"model_forwards":0,
                      "optimizer_updates":0,"state_unchanged":True,
                      "transcript":m["sources"][arm]["transcripts"][rank]}
            runner.save_json(folder/f"complete_rank{rank}.json", {
                "complete":True,"arm":arm,"rank":rank,"start":START+500,"stop":START+501,
                "updates":1,"total_updates":501,
                "initial_state":{"optimizer":arm,"scheduler":"s","scaler":"g","model":arm},
                "cursor_replay":replay,"manifest_fingerprint":"extended","health_records":health,
                "transcript":file_identity(transcript),"update_log":file_identity(log)})
            if rank == 0:
                saved[arm] = {"iteration":START+500,"trainer":{
                    "gt_normalization_trial":{"arm":arm,"normalization":base.ARMS[arm],"start":START,
                                              "updates":501,"manifest_fingerprint":"extended"},
                    "gt_normalization_extension":{"parent_fingerprint":"parent","completed_updates":500,
                        "additional_updates":1,"total_updates":501,"manifest_fingerprint":"extended",
                        "cursor_replay":replay}}}
    monkeypatch.setattr(runner,"load_trusted_torch_file",lambda p: saved[Path(p).parent.name])
    def endpoint(ckpt,iteration):
        assert ckpt["iteration"] == iteration
    monkeypatch.setattr(runner,"endpoint_state",endpoint)
    monkeypatch.setattr(runner,"resume_identity",lambda *a: None)  # Separately tested with real optimizer state.
    for arm in base.ARMS:
        assert runner.verify_arm(m,arm)["final_rank"] == health[-1]
    saved["B"]["trainer"]["gt_normalization_extension"]["completed_updates"] = 0
    with pytest.raises(ValueError,match="provenance"):
        runner.verify_arm(m,"B")
    saved["B"]["trainer"]["gt_normalization_extension"]["completed_updates"] = 500
    path = tmp_path/"B/pairing_rank3.jsonl"
    rows = runner.read_rows(path)
    rows[0]["data_seed"] = 42  # Restarted stream must fail even if A/B losses need not match.
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    receipt_path = tmp_path/"B/complete_rank3.json"
    receipt = runner.load_json(receipt_path)
    receipt["transcript"] = file_identity(path)
    runner.save_json(receipt_path,receipt)
    with pytest.raises(ValueError,match="timeline"):
        runner.verify_arm(m,"B")


def test_incomplete_extension_never_silently_restarts_or_overwrites(tmp_path,monkeypatch):
    parent = tmp_path/"source.json"
    parent.write_text("{}")
    identity = file_identity(parent)
    output = tmp_path/"new"
    (output/"A").mkdir(parents=True)
    (output/"A/last_checkpoint").write_text("parent-A-500")
    (output/"A/console.log").write_text("interrupted")
    m = {"output_dir":str(output),"parent_manifest":identity,"parent_summary":identity,
         "train_annotations":identity,"val_annotations":identity,"assets":[],
         "sources":{a:{"checkpoint":identity,"transcripts":[identity]} for a in base.ARMS}}
    before = {p:str(p.read_bytes()) for p in (parent,output/"A/last_checkpoint",output/"A/console.log")}
    monkeypatch.setattr(runner,"run_evaluation",lambda *a: pytest.fail("No GPU work allowed"))
    with pytest.raises(ValueError,match="Incomplete extension"):
        runner.execute(m)
    assert {p:str(p.read_bytes()) for p in before} == before


def _ddp_extension(rank,rendezvous,directory):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method=Path(rendezvous).as_uri(),rank=rank,world_size=4)
    patch = pytest.MonkeyPatch()
    try:
        root = Path(directory)
        m,build,snapshots,finals = paired_setup(root,patch,rank,distributed=True)
        for arm in base.ARMS:
            trainer = build(arm,m,*snapshots[arm])
            advance(trainer,2,4)
            assert trainer.cursor_replay["verified"] and trainer.actual_updates == 4
            for key,value in trainer.raw_model.state_dict().items():
                torch.testing.assert_close(value,finals[arm][0][key],rtol=1e-6,atol=1e-7)
            assert trainer.grad_scaler.state_dict() == finals[arm][2]
            hashes = [None]*4
            dist.all_gather_object(hashes,state_digest(trainer.raw_model.state_dict()))
            assert len(set(hashes)) == 1
            trainer.close_streams()
            dist.barrier()
        (root/f"rank{rank}.ok").write_text("extension paired")
    finally:
        patch.undo()
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(),reason="Gloo unavailable")
def test_four_rank_extension_restores_stream_and_matches_continuous_training(tmp_path):
    dirs(tmp_path)
    torch.multiprocessing.spawn(_ddp_extension,args=(str(tmp_path/"rendezvous"),str(tmp_path)),nprocs=4,join=True)
    assert all((tmp_path/f"rank{r}.ok").exists() for r in range(4))
