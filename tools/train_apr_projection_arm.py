#!/usr/bin/env python3
"""Internal native four-GPU worker for the APR/projection overnight experiment."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch

from tools.apr_projection_trial_ops import ARMS, make_trainer_class
from tools.compare_rare_pr_reports import file_identity,load_json,save_json
from tools.decoder_aux_ablation_ops import START,HORIZON
from tools.diagnose_rare_fp_regions import fingerprint
from tools.gt_normalization_trial_ops import RANK_GUARD,RANK_PERIOD
from tools.train_gt_normalization_arm import training_options as normalization_options


def training_options(m,arm):
    # Common settings only; none of the GT-normalization intervention is used.
    common = normalization_options(m,"A")
    result = [option.replace(str(Path(m["output_dir"])/"A"),str(Path(m["output_dir"])/arm))
              if option.startswith(("train.output_dir=","dataloader.evaluator.output_dir=")) else option
              for option in common]
    result += [f"train.tpa_conflict_projection={str(ARMS[arm]['projection']).lower()}"]
    # Absolute D2 iteration period: one intermediate checkpoint at update500,
    # then model_final at update2000; no intermediate LVIS evaluation.
    if m["updates"] > 500:
        result = [f"train.checkpointer.period={START+500}" if opt.startswith("train.checkpointer.period=")
                  else opt for opt in result]
    return result


def read_manifest(path):
    m = load_json(path)
    if fingerprint({k:v for k,v in m.items() if k != "fingerprint"}) != m["fingerprint"]:
        raise ValueError("APR/projection manifest fingerprint mismatch")
    if (m["schema"] != "apr_projection_trial_v1" or m["arms"] != ARMS or m["start"] != START
            or not 1 <= m["updates"] <= 2000 or m["num_gpus"] != 4 or m["seed"] != 42
            or m["lr_horizon"] != HORIZON or m["rank_guard"] != RANK_GUARD or m["rank_period"] != RANK_PERIOD
            or m["cpu_threads"] < 1 or str(torch.__version__) != m["torch_version"]):
        raise ValueError("Unexpected trial protocol/environment")
    for key in ("config","prompt_bank","reference_manifest"):
        if file_identity(m[key]["path"]) != m[key]:
            raise ValueError("Trial input changed: "+key)
    for name,digest in m["code"].items():
        target = (ROOT/name).resolve()
        if ROOT not in target.parents or file_identity(target)["sha256"] != digest:
            raise ValueError("Training/experiment code changed: "+name)
    return m


def worker(path,arm):
    from detectron2.config import LazyConfig
    from detectron2.engine import default_setup
    from detectron2.utils import comm
    from tools import train_net as native
    from tools.audit_accumulation_objective import parameter_groups

    m = read_manifest(path)
    torch.set_num_threads(m["cpu_threads"])
    rank = comm.get_rank()
    if comm.get_world_size() != 4 or file_identity(m["checkpoint"]["path"]) != m["checkpoint"]:
        raise ValueError("Require four GPUs and the verified original full 8ep checkpoint")
    output = Path(m["output_dir"])/arm
    if (output/"last_checkpoint").read_text().strip() != m["checkpoint"]["path"]:
        raise ValueError("Weights-only/partial-arm resume is forbidden")
    cfg = LazyConfig.apply_overrides(LazyConfig.load(m["config"]["path"]),training_options(m,arm))
    if (cfg.dataloader.train.dataset.names != "lvis_v1_train_norare" or not cfg.model.use_fed_loss
            or cfg.model.cluster_fed_loss or cfg.model.get("teacher_rpsa",False)
            or Path(cfg.model.query_path).resolve() != Path(m["prompt_bank"]["path"]).resolve()
            or Path(cfg.model.eval_query_path).resolve() != Path(m["prompt_bank"]["path"]).resolve()
            or float(cfg.model.criterion.weight_dict.loss_apr) != 1.):
        raise ValueError("Dataset/FedLoss/prompt/teacher/APR protocol differs")
    args = SimpleNamespace(config_file=m["config"]["path"],resume=True,eval_only=False,
                           num_gpus=4,num_machines=1,machine_rank=0,opts=[])
    default_setup(cfg,args)
    torch.backends.cudnn.benchmark = False
    prompts = torch.from_numpy(np.load(m["prompt_bank"]["path"],allow_pickle=False)).float()
    if tuple(prompts.shape) != (1203,8,768):
        raise ValueError("Unexpected prompt bank")
    cls = make_trainer_class(native.Trainer,m,arm,rank,prompts)
    instances = []
    def construct(*a,**kw):
        trainer = cls(*a,**kw)
        instances.append(trainer)
        if parameter_groups(trainer.raw_model)[0] != m["inventory"]:
            raise ValueError("Trainable parameter scope changed")
        print(f"[policy {arm}] {trainer.policy}; native micro GT normalization",flush=True)
        return trainer
    original = native.Trainer
    native.Trainer = construct
    try:
        native.do_train(args,cfg)
        if len(instances) != 1:
            raise ValueError("Expected one trainer")
        t = instances[0]
        if t.actual_updates != m["updates"] or t.iter != START+m["updates"]:
            raise ValueError("Incomplete update budget")
        if t.reference_stream and t.reference_stream.readline():
            raise ValueError("Unused control pairing records")
        for stream in (t.pair_stream,t.update_stream,t.health_stream):
            if stream is not None:
                stream.flush()
        save_json(output/f"complete_rank{rank}.json",{
            "complete":True,"arm":arm,"rank":rank,"policy":t.policy,"start":START,"stop":t.iter,
            "updates":t.actual_updates,"initial_state":t.initial_state,"health_records":t.health_records,
            "manifest_fingerprint":m["fingerprint"],"transcript":file_identity(output/f"pairing_rank{rank}.jsonl"),
            "update_log":file_identity(output/f"updates_rank{rank}.jsonl")})
        comm.synchronize()
    finally:
        native.Trainer = original
        for trainer in instances:
            trainer.close_streams()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True)
    p.add_argument("--arm",required=True,choices=tuple(ARMS))
    args = p.parse_args()
    read_manifest(args.manifest)
    if torch.cuda.device_count() != 4:
        raise ValueError("Expose exactly four GPUs")
    from detectron2.engine import launch
    launch(worker,num_gpus_per_machine=4,dist_url="auto",args=(args.manifest,args.arm))


if __name__ == "__main__":
    main()
