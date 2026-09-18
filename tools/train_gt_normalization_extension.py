#!/usr/bin/env python3
"""Internal four-rank worker for extending the verified 500-update A/B trial."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.decoder_aux_ablation_ops import START, HORIZON
from tools.diagnose_rare_fp_regions import fingerprint
from tools.gt_normalization_extension_ops import arm_view, make_trainer_class
from tools.gt_normalization_trial_ops import ARMS, RANK_GUARD, RANK_PERIOD
from tools.train_gt_normalization_arm import training_options as parent_options


def training_options(manifest, arm):
    return parent_options(arm_view(manifest, arm), arm)


def read_manifest(path):
    m = load_json(path)
    if fingerprint({k:v for k,v in m.items() if k != "fingerprint"}) != m["fingerprint"]:
        raise ValueError("Extension manifest fingerprint mismatch")
    if (m["schema"] != "gt_normalization_extension_v1" or m["completed_updates"] != 500
            or not 500 < m["total_updates"] <= 2000 or m["additional_updates"] != m["total_updates"]-500
            or m["num_gpus"] != 4 or m["seed"] != 42 or m["arms"] != ARMS
            or m["start"] != START+500 or m["lr_horizon"] != HORIZON
            or m["rank_guard"] != RANK_GUARD or m["rank_period"] != RANK_PERIOD):
        raise ValueError("Unexpected extension budget/protocol")
    if str(torch.__version__) != m["torch_version"]:
        raise ValueError("Use the original lami/PyTorch environment")
    for key in ("config", "prompt_bank", "parent_manifest", "parent_summary"):
        if file_identity(m[key]["path"]) != m[key]:
            raise ValueError("Extension input changed: " + key)
    for name, digest in m["code"].items():
        target = (ROOT/name).resolve()
        if ROOT not in target.parents or file_identity(target)["sha256"] != digest:
            raise ValueError("Original/extension code changed: " + name)
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
        raise ValueError("Expected four GPUs")
    source = m["sources"][arm]
    if file_identity(source["checkpoint"]["path"]) != source["checkpoint"]:
        raise ValueError("Parent arm checkpoint changed")
    for identity in source["transcripts"]:
        if file_identity(identity["path"]) != identity:
            raise ValueError("Parent data transcript changed")
    output = Path(m["output_dir"])/arm
    if (output/"last_checkpoint").read_text().strip() != source["checkpoint"]["path"]:
        raise ValueError("Must resume this arm's own completed 500-update checkpoint")
    cfg = LazyConfig.apply_overrides(LazyConfig.load(m["config"]["path"]), training_options(m, arm))
    if (cfg.dataloader.train.dataset.names != "lvis_v1_train_norare"
            or not cfg.model.use_fed_loss or cfg.model.cluster_fed_loss or cfg.model.get("teacher_rpsa",False)
            or Path(cfg.model.query_path).resolve() != Path(m["prompt_bank"]["path"]).resolve()
            or Path(cfg.model.eval_query_path).resolve() != Path(m["prompt_bank"]["path"]).resolve()):
        raise ValueError("Dataset/FedLoss/teacher/prompt protocol differs")
    args = SimpleNamespace(config_file=m["config"]["path"], resume=True, eval_only=False,
                           num_gpus=4, num_machines=1, machine_rank=0, opts=[])
    default_setup(cfg, args)
    torch.backends.cudnn.benchmark = False
    prompts = torch.from_numpy(np.load(m["prompt_bank"]["path"], allow_pickle=False)).float()
    if tuple(prompts.shape) != (1203,8,768):
        raise ValueError("Unexpected prompt bank")
    cls = make_trainer_class(native.Trainer, m, arm, rank, prompts)
    instances = []

    def construct(*a, **kw):
        trainer = cls(*a, **kw)
        instances.append(trainer)
        if parameter_groups(trainer.raw_model)[0] != m["inventory"]:
            raise ValueError("Trainable scope changed from original A/B")
        return trainer

    original = native.Trainer
    native.Trainer = construct
    try:
        native.do_train(args, cfg)
        if len(instances) != 1:
            raise ValueError("Expected one trainer")
        trainer = instances[0]
        if trainer.actual_updates != m["total_updates"] or trainer.iter != START+m["total_updates"]:
            raise ValueError("Incomplete extension update budget")
        if trainer.reference_stream and trainer.reference_stream.readline():
            raise ValueError("A extension transcript has unused records")
        for stream in (trainer.pair_stream, trainer.update_stream, trainer.health_stream):
            if stream is not None:
                stream.flush()
        save_json(output/f"complete_rank{rank}.json", {
            "complete":True, "arm":arm, "rank":rank, "start":m["start"], "stop":trainer.iter,
            "updates":m["additional_updates"], "total_updates":trainer.actual_updates,
            "initial_state":trainer.initial_state, "cursor_replay":trainer.cursor_replay,
            "manifest_fingerprint":m["fingerprint"], "health_records":trainer.health_records,
            "transcript":file_identity(output/f"pairing_rank{rank}.jsonl"),
            "update_log":file_identity(output/f"updates_rank{rank}.jsonl")})
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
    launch(worker, num_gpus_per_machine=4, dist_url="auto", args=(args.manifest,args.arm))


if __name__ == "__main__":
    main()
