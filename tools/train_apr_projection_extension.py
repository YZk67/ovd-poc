#!/usr/bin/env python3
"""Internal four-GPU worker: extend only A/P from 2000 to 4000 total updates."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools.apr_projection_extension_ops import ARMS, arm_view, make_trainer_class
from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.decoder_aux_ablation_ops import START, HORIZON
from tools.diagnose_rare_fp_regions import fingerprint
from tools.gt_normalization_trial_ops import RANK_GUARD, RANK_PERIOD
from tools.train_apr_projection_arm import training_options as parent_options


def training_options(m, arm):
    if arm not in ARMS:
        raise ValueError("Only A/P may be extended")
    return [f"train.checkpointer.period={START+m['total_updates']+1}"
            if opt.startswith("train.checkpointer.period=") else opt
            for opt in parent_options(arm_view(m, arm), arm)]


def read_manifest(path):
    m = load_json(path)
    if fingerprint({k: v for k, v in m.items() if k != "fingerprint"}) != m["fingerprint"]:
        raise ValueError("APR extension manifest fingerprint mismatch")
    if (m["schema"] != "apr_projection_extension_v1" or m["arms"] != ARMS
            or m["completed_updates"] != 2000 or not 2000 < m["total_updates"] <= 4000
            or m["additional_updates"] != m["total_updates"]-2000 or m["start"] != START+2000
            or m["num_gpus"] != 4 or m["seed"] != 42 or m["lr_horizon"] != HORIZON
            or m["rank_guard"] != RANK_GUARD or m["rank_period"] != RANK_PERIOD
            or set(m["sources"]) != set(ARMS) or m["cpu_threads"] < 1
            or str(torch.__version__) != m["torch_version"]):
        raise ValueError("Unexpected APR extension protocol/environment")
    for key in ("config", "prompt_bank", "parent_manifest", "parent_summary"):
        if file_identity(m[key]["path"]) != m[key]:
            raise ValueError("Extension input changed: "+key)
    for name, digest in m["code"].items():
        target = (ROOT/name).resolve()
        if ROOT not in target.parents or file_identity(target)["sha256"] != digest:
            raise ValueError("Original/extension code changed: "+name)
    return m


def worker(path, arm):
    from detectron2.config import LazyConfig
    from detectron2.engine import default_setup
    from detectron2.utils import comm
    from tools import train_net as native
    from tools.audit_accumulation_objective import parameter_groups

    m = read_manifest(path)
    torch.set_num_threads(m["cpu_threads"])
    rank = comm.get_rank()
    if comm.get_world_size() != 4:
        raise ValueError("Require four GPUs")
    source = m["sources"][arm]
    for identity in (source["checkpoint"], *source["transcripts"]):
        if file_identity(identity["path"]) != identity:
            raise ValueError("Parent endpoint/transcript changed: "+identity["path"])
    output = Path(m["output_dir"])/arm
    if (output/"last_checkpoint").read_text().strip() != source["checkpoint"]["path"]:
        raise ValueError("Must resume this arm's own complete 2000-update checkpoint")
    cfg = LazyConfig.apply_overrides(LazyConfig.load(m["config"]["path"]), training_options(m, arm))
    if (cfg.dataloader.train.dataset.names != "lvis_v1_train_norare" or not cfg.model.use_fed_loss
            or cfg.model.cluster_fed_loss or cfg.model.get("teacher_rpsa", False)
            or Path(cfg.model.query_path).resolve() != Path(m["prompt_bank"]["path"]).resolve()
            or Path(cfg.model.eval_query_path).resolve() != Path(m["prompt_bank"]["path"]).resolve()
            or float(cfg.model.criterion.weight_dict.loss_apr) != 1.):
        raise ValueError("Dataset/FedLoss/prompt/teacher/APR protocol differs")
    args = SimpleNamespace(config_file=m["config"]["path"], resume=True, eval_only=False,
                           num_gpus=4, num_machines=1, machine_rank=0, opts=[])
    default_setup(cfg, args)
    torch.backends.cudnn.benchmark = False
    prompts = torch.from_numpy(np.load(m["prompt_bank"]["path"], allow_pickle=False)).float()
    if tuple(prompts.shape) != (1203, 8, 768):
        raise ValueError("Unexpected prompt bank")
    cls = make_trainer_class(native.Trainer, m, arm, rank, prompts)
    instances = []

    def construct(*a, **kw):
        trainer = cls(*a, **kw)
        instances.append(trainer)
        if parameter_groups(trainer.raw_model)[0] != m["inventory"]:
            raise ValueError("Trainable parameter scope changed")
        print(f"[policy {arm}] {trainer.policy}; resume OWN endpoint, native micro GT normalization", flush=True)
        return trainer

    original = native.Trainer
    native.Trainer = construct
    try:
        native.do_train(args, cfg)
        if len(instances) != 1:
            raise ValueError("Expected one trainer")
        t = instances[0]
        if t.actual_updates != m["total_updates"] or t.iter != START+m["total_updates"]:
            raise ValueError("Incomplete extension update budget")
        if t.reference_stream and t.reference_stream.readline():
            raise ValueError("Unused A extension pairing records")
        for stream in (t.pair_stream, t.update_stream, t.health_stream):
            if stream is not None:
                stream.flush()
        save_json(output/f"complete_rank{rank}.json", {
            "complete": True, "arm": arm, "rank": rank, "policy": t.policy,
            "start": m["start"], "stop": t.iter, "updates": m["additional_updates"],
            "total_updates": t.actual_updates, "initial_state": t.initial_state,
            "cursor_replay": t.cursor_replay, "health_records": t.health_records,
            "manifest_fingerprint": m["fingerprint"],
            "transcript": file_identity(output/f"pairing_rank{rank}.jsonl"),
            "update_log": file_identity(output/f"updates_rank{rank}.jsonl")})
        comm.synchronize()
    finally:
        native.Trainer = original
        for trainer in instances:
            trainer.close_streams()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--arm", required=True, choices=tuple(ARMS))
    args = p.parse_args()
    read_manifest(args.manifest)
    if torch.cuda.device_count() != 4:
        raise ValueError("Expose exactly four GPUs")
    from detectron2.engine import launch
    launch(worker, num_gpus_per_machine=4, dist_url="auto", args=(args.manifest, args.arm))


if __name__ == "__main__":
    main()
