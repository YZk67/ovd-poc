from contextlib import nullcontext
import json
from pathlib import Path

import pytest
import torch

from tools import fresh_gt_normalization_ops as ops
from tools import run_fresh_gt_normalization_trial as runner
from tools import train_fresh_gt_normalization_arm as worker
from tools.decoder_aux_ablation_ops import HORIZON, state_digest
from test_decoder_aux_ablation import native_class
from test_gt_normalization_trial import Data, Model, PROMPTS


def record(iteration=0):
    return {"iteration":iteration,"micro":0,"mapped":[1],"fedloss":[1,2],
            "lrs":[1e-7],"amp_scale":1024.,"rng_after":{"cpu":"same"},
            "normalization":{"counts":[8,16]},"loss_keys":["loss_class"],
            "losses":{"loss_class":1.},"multiplier":1.}


def test_fresh_pair_requires_first_loss_and_all_sampling_fields():
    a = record()
    ops.verify_fresh_pair({**a,"multiplier":1.2},a)
    with pytest.raises(ValueError,match="first-window"):
        ops.verify_fresh_pair({**a,"losses":{"loss_class":2.}},a)
    later = record(1)
    ops.verify_fresh_pair({**later,"losses":{"loss_class":9.}},later)
    with pytest.raises(ValueError,match="Unpaired"):
        ops.verify_fresh_pair({**later,"fedloss":[3]},later)


def test_fresh_rank_guard_accepts_observed_initial_formation_but_not_collapse_or_deadline_failure():
    observed = {"all":{"mean_rank":3.1204,"p10_rank":2.784,"min_rank":2.4409,
                       "mean_cos":.8943,"rank_below_2_count":0},
                "rare":{"mean_rank":3.1377,"p10_rank":2.8132,"min_rank":2.4794,
                        "mean_cos":.8909,"rank_below_2_count":0},"guard_pass":False}
    initial = ops.apply_fresh_rank_guard(observed,0)
    assert initial["guard_pass"] and not initial["strict_guard_pass"]
    stable = ops.apply_fresh_rank_guard(observed,50,initial)
    assert stable["guard_pass"]
    assert not ops.apply_fresh_rank_guard(observed,500,initial)["guard_pass"]
    collapsed = {**observed,"rare":{**observed["rare"],"min_rank":1.5,"rank_below_2_count":1}}
    assert not ops.apply_fresh_rank_guard(collapsed,0)["guard_pass"]


def build(tmp_path,monkeypatch,arm,rank=0,distributed=False):
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(torch.cuda,"is_current_stream_capturing",lambda:False)
    monkeypatch.setattr(torch.cuda.amp,"autocast",lambda **kw:nullcontext())
    torch.manual_seed(23)
    native = native_class()
    model = Model()
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    kwargs = dict(amp=True,clip_grad_params={"max_norm":.5,"norm_type":2},
                  separate_tpa_grad_clip=True,tpa_conflict_projection=True,
                  lr_scheduler_max_iter=HORIZON,gradient_accumulation_steps=2)
    manifest = {"output_dir":str(tmp_path),"updates":2,"seed":42,"fingerprint":"fresh"}
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,find_unused_parameters=True)
    cls = ops.make_trainer_class(native,manifest,arm,rank,PROMPTS,[{"name":"toy"}])
    trainer = cls(model,Data(),optimizer,grad_scaler=torch.amp.GradScaler("cpu",init_scale=1024.),**kwargs)
    trainer.iter = 0
    trainer.hooks = {"LRScheduler":{"last_epoch":0,"base_lrs":[1e-4]}}
    return trainer


def test_fresh_actual_native_steps_pair_initial_state_and_keep_tpa_noncollapsed(tmp_path,monkeypatch):
    for arm in ("A","B"):
        (tmp_path/arm).mkdir()
        trainer = build(tmp_path,monkeypatch,arm)
        initial = state_digest(trainer.raw_model.state_dict())
        initial_weight = trainer.raw_model.transformer.decoder.layers[0].weight.detach().clone()
        for iteration in range(2):
            trainer.iter = iteration
            trainer.run_step()
        assert trainer.actual_updates == 2
        assert trainer.initial_state["model"] == initial
        assert not torch.equal(trainer.raw_model.transformer.decoder.layers[0].weight,initial_weight)
        assert [row["update"] for row in trainer.health_records] == [0,2]
        assert all(row["guard_pass"] for row in trainer.health_records)
        assert trainer.state_dict()["fresh_gt_normalization_trial"]["start"] == 0
        trainer.close_streams()
    assert runner.read_rows(tmp_path/"A/pairing_rank0.jsonl")[0]["iteration"] == 0
    assert runner.read_rows(tmp_path/"B/pairing_rank0.jsonl")[0]["iteration"] == 0
    assert json.loads((tmp_path/"A/initial_rank0.json").read_text()) == json.loads(
        (tmp_path/"B/initial_rank0.json").read_text())


def test_training_options_are_true_fresh_start_with_original_lr_timeline():
    manifest = {"output_dir":"/tmp/fresh","updates":2000,
                "initial_checkpoint":{"path":"/tmp/clip.pth"}}
    a,b = [worker.training_options(manifest,arm) for arm in ("A","B")]
    assert [x for x in a if "output_dir=" not in x] == [x for x in b if "output_dir=" not in x]
    for option in ("train.init_checkpoint_scope=backbone_only","train.max_iter=2000",
                   "train.lr_scheduler_max_iter=85200","train.gradient_accumulation_steps=2",
                   "dataloader.train.total_batch_size=16","train.seed=42",
                   "model.classifier.tpa_prototype_mode_strength=0.0"):
        assert option in a
    assert all("resume" not in option for option in a)


def test_prepare_rejects_existing_output_before_config_or_gpu_work(tmp_path):
    args = type("Args",(),dict(output_dir=str(tmp_path),config_file="must-not-open",updates=2000,
                               num_gpus=4,seed=42,cpu_threads=2))()
    with pytest.raises(ValueError,match="NEW output"):
        runner.prepare(args)


def test_python39_syntax_and_declared_code_files_exist():
    import ast
    for relative in (*runner.CODE_FILES,"tools/train_fresh_gt_normalization_arm.py"):
        path = runner.ROOT/relative
        assert path.is_file()
        ast.parse(path.read_text(),filename=str(path),feature_version=(3,9))


def _distributed_fresh(rank,rendezvous,directory):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method=Path(rendezvous).as_uri(),rank=rank,world_size=4)
    patch = pytest.MonkeyPatch()
    try:
        root = Path(directory)
        states = {}
        for arm in ("A","B"):
            trainer = build(root,patch,arm,rank,distributed=True)
            for iteration in range(2):
                trainer.iter = iteration
                trainer.run_step()
            states[arm] = trainer.initial_state
            assert trainer.actual_updates == 2
            assert all(row["guard_pass"] for row in trainer.health_records)
            hashes = [None]*4
            dist.all_gather_object(hashes,state_digest(trainer.raw_model.state_dict()))
            assert len(set(hashes)) == 1
            trainer.close_streams()
            dist.barrier()
        assert states["A"] == states["B"]
        (root/f"rank{rank}.ok").write_text("fresh paired")
    finally:
        patch.undo()
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(),reason="Gloo unavailable")
def test_four_rank_fresh_start_pairs_initial_state_inputs_and_updates(tmp_path):
    for arm in ("A","B"):
        (tmp_path/arm).mkdir()
    torch.multiprocessing.spawn(_distributed_fresh,
        args=(str(tmp_path/"rendezvous"),str(tmp_path)),nprocs=4,join=True)
    assert all((tmp_path/f"rank{rank}.ok").exists() for rank in range(4))
