#!/usr/bin/env python3
"""Internal worker: native Trainer, paired data, only GT denominator differs."""
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
from tools.gt_normalization_trial_ops import ARMS, RANK_GUARD, RANK_PERIOD, make_trainer_class
from tools.train_tpa_formula_arm import training_options as formula_options


def training_options(manifest, arm):
    # Reuse only the common resume/inference options, NOT the frozen-TPA trainer.
    options = formula_options(manifest, arm)
    result = []
    for option in options:
        if option.startswith("model.classifier.tpa_train_aggregation="):
            option = "model.classifier.tpa_train_aggregation=calibrated"
        elif option.startswith("train.checkpointer.period="):
            # Keep final.pth, avoid saving a second identical periodic checkpoint.
            option = f"train.checkpointer.period={START+manifest['updates']+1}"
        result.append(option)
    return result


def read_manifest(path):
    m = load_json(path)
    if fingerprint({k: v for k, v in m.items() if k != "fingerprint"}) != m["fingerprint"]:
        raise ValueError("Normalization trial manifest fingerprint mismatch")
    if (m.get("schema") != "gt_normalization_ab_v1" or m["arms"] != ARMS
            or not 1 <= m["updates"] <= 500 or m["num_gpus"] != 4 or m["seed"] != 42
            or m["start"] != START or m["lr_horizon"] != HORIZON
            or m["rank_guard"] != RANK_GUARD or m["rank_period"] != RANK_PERIOD):
        raise ValueError("Unexpected paired normalization protocol")
    if str(torch.__version__) != m["torch_version"]:
        raise ValueError("Use the same lami/PyTorch environment as preparation")
    for name in ("config", "prompt_bank"):
        if file_identity(m[name]["path"]) != m[name]:
            raise ValueError(f"Trial input changed: {name}")
    for relative, digest in m["code"].items():
        target = (ROOT / relative).resolve()
        if ROOT not in target.parents or file_identity(target)["sha256"] != digest:
            raise ValueError(f"Trial code changed: {relative}")
    return m


def worker(manifest_path, arm):
    from detectron2.config import LazyConfig
    from detectron2.engine import default_setup
    from detectron2.utils import comm
    from tools import train_net as native
    from tools.audit_accumulation_objective import parameter_groups

    m = read_manifest(manifest_path)
    torch.set_num_threads(m["cpu_threads"])
    rank = comm.get_rank()
    if comm.get_world_size() != 4:
        raise ValueError("Require four GPUs, physical batch16, accumulation2")
    output = Path(m["output_dir"]) / arm
    if (output / "last_checkpoint").read_text().strip() != m["checkpoint"]["path"]:
        raise ValueError("Must resume the original full 8ep checkpoint")
    cfg = LazyConfig.apply_overrides(LazyConfig.load(m["config"]["path"]), training_options(m, arm))
    if (cfg.dataloader.train.dataset.names != "lvis_v1_train_norare"
            or not cfg.model.use_fed_loss or cfg.model.cluster_fed_loss
            or cfg.model.get("teacher_rpsa", False)
            or Path(cfg.model.query_path).resolve() != Path(m["prompt_bank"]["path"]).resolve()
            or Path(cfg.model.eval_query_path).resolve() != Path(m["prompt_bank"]["path"]).resolve()):
        raise ValueError("Native dataset/FedLoss/prompt-bank protocol changed")
    args = SimpleNamespace(config_file=m["config"]["path"], resume=True, eval_only=False,
                           num_gpus=4, num_machines=1, machine_rank=0, opts=[])
    default_setup(cfg, args)
    torch.backends.cudnn.benchmark = False
    prompts = torch.from_numpy(np.load(m["prompt_bank"]["path"], allow_pickle=False)).float()
    if tuple(prompts.shape) != (1203, 8, 768):
        raise ValueError("Expected full 1203 x 8 x 768 prompt bank")
    cls = make_trainer_class(native.Trainer, m, arm, rank, prompts)
    instances = []

    def construct(*a, **kw):
        trainer = cls(*a, **kw)
        instances.append(trainer)
        inventory, _, _ = parameter_groups(trainer.raw_model)
        if inventory != m["inventory"]:
            raise ValueError("Trainable parameter scope differs from optimizer-aware audit")
        return trainer

    original = native.Trainer
    native.Trainer = construct
    try:
        native.do_train(args, cfg)
        if len(instances) != 1:
            raise ValueError("Expected exactly one trainer")
        trainer = instances[0]
        if trainer.actual_updates != m["updates"] or trainer.iter != START+m["updates"]:
            raise ValueError("Training did not complete exact paired budget")
        if trainer.reference_stream and trainer.reference_stream.readline():
            raise ValueError("A transcript has unused records")
        for stream in (trainer.pair_stream, trainer.update_stream, trainer.health_stream):
            if stream is not None:
                stream.flush()
        receipt = {"complete": True, "arm": arm, "rank": rank, "updates": trainer.actual_updates,
                   "start": START, "stop": trainer.iter, "initial_state": trainer.initial_state,
                   "manifest_fingerprint": m["fingerprint"],
                   "transcript": file_identity(output / f"pairing_rank{rank}.jsonl"),
                   "update_log": file_identity(output / f"updates_rank{rank}.jsonl"),
                   "health_records": trainer.health_records}
        save_json(output / f"complete_rank{rank}.json", receipt)
        comm.synchronize()
    finally:
        native.Trainer = original
        for trainer in instances:
            trainer.close_streams()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--arm", required=True, choices=tuple(ARMS))
    args = parser.parse_args()
    read_manifest(args.manifest)
    if torch.cuda.device_count() != 4:
        raise ValueError("Expose exactly four GPUs")
    from detectron2.engine import launch
    launch(worker, num_gpus_per_machine=4, dist_url="auto", args=(args.manifest, args.arm))


if __name__ == "__main__":
    main()
