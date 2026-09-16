#!/usr/bin/env python3
"""Internal four-GPU worker for run_decoder_aux_ablation.py; not a general trainer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.decoder_aux_ablation_ops import HORIZON, START, make_trainer_class
from tools.diagnose_rare_fp_regions import fingerprint


def read_manifest(path):
    m = load_json(path)
    if fingerprint({k: v for k, v in m.items() if k != "fingerprint"}) != m["fingerprint"]:
        raise ValueError("Paired manifest fingerprint mismatch")
    if not 1 <= m["updates"] <= 500 or m["num_gpus"] != 4 or m["seed"] != 42:
        raise ValueError("Unexpected trial budget/protocol")
    if str(torch.__version__) != m["torch_version"]:
        raise ValueError("Use the same PyTorch environment as the audit/preflight")
    for relative, digest in m["code"].items():
        target = (ROOT / relative).resolve()
        if ROOT not in target.parents or file_identity(target)["sha256"] != digest:
            raise ValueError(f"Paired trial code changed: {relative}")
    return m


def training_options(m, arm):
    output = str(Path(m["output_dir"]) / arm)
    return [
        f"train.output_dir={json.dumps(output)}",
        f"dataloader.evaluator.output_dir={json.dumps(output)}",
        f"train.init_checkpoint={json.dumps(m['checkpoint']['path'])}",
        "train.init_checkpoint_scope=full",
        f"train.max_iter={START + m['updates']}", f"train.lr_scheduler_max_iter={HORIZON}",
        "train.gradient_accumulation_steps=2", "dataloader.train.total_batch_size=16",
        "dataloader.train.num_workers=0", "train.seed=42", "train.amp.enabled=True",
        "train.eval_period=0", "train.eval_after_train=False",
        f"train.checkpointer.period={START + m['updates']}", "train.log_period=50",
        "train.device=cuda", "model.device=cuda",
        "model.alpha=0.0", "model.beta=0.3", "model.novel_scale=3.0",
        "model.classifier.tpa_num_prototypes=5", "model.classifier.tpa_tau=0.004375",
        "model.classifier.tpa_cls_tau=0.07", "model.classifier.tpa_slot_prior_strength=0.2",
        "model.classifier.tpa_prototype_mode_strength=0.0",
        "model.classifier.tpa_eval_legacy_logsumexp=False",
        "model.classifier.tpa_eval_logit_bias=0.0", "model.tpa_eval_mode_scale=1.0",
        "model.soft_category_topk=3", "model.inference_query_class_topk=0",
        "model.select_box_nums_for_evaluation=300", "model.score_ensemble=True",
    ]


def worker(manifest_path, arm):
    from detectron2.config import LazyConfig
    from detectron2.engine import default_setup
    from detectron2.utils import comm
    from tools import train_net as native

    m = read_manifest(manifest_path)
    torch.set_num_threads(m["cpu_threads"])
    rank = comm.get_rank()
    if comm.get_world_size() != 4:
        raise ValueError("Expected four ranks, physical batch16, accumulation2")
    output = Path(m["output_dir"]) / arm
    if (output / "last_checkpoint").read_text().strip() != m["checkpoint"]["path"]:
        raise ValueError("Arm must start from the original full 8ep checkpoint, not a partial trial")
    cfg = LazyConfig.load(m["config"]["path"])
    cfg = LazyConfig.apply_overrides(cfg, training_options(m, arm))
    args = SimpleNamespace(config_file=m["config"]["path"], resume=True, eval_only=False,
                           num_gpus=4, num_machines=1, machine_rank=0, opts=[])
    default_setup(cfg, args)
    torch.backends.cudnn.benchmark = False
    cls = make_trainer_class(native.Trainer, m, arm, rank)
    instances = []
    def construct(*a, **kw):
        trainer = cls(*a, **kw)
        instances.append(trainer)
        return trainer
    original = native.Trainer
    native.Trainer = construct
    try:
        native.do_train(args, cfg)
        if len(instances) != 1:
            raise ValueError("Expected one native trainer")
        trainer = instances[0]
        if trainer.actual_updates != m["updates"] or trainer.iter != START + m["updates"]:
            raise ValueError("Training did not complete the exact optimizer-update budget")
        if trainer.reference_stream and trainer.reference_stream.readline():
            raise ValueError("Unconsumed A pairing records")
        trainer.pair_stream.flush()
        receipt = {"complete": True, "arm": arm, "rank": rank, "updates": trainer.actual_updates,
                   "start": START, "stop": trainer.iter, "initial_state": trainer.initial_state,
                   "manifest_fingerprint": m["fingerprint"],
                   "transcript": file_identity(output / f"pairing_rank{rank}.jsonl"),
                   "gradient_log": file_identity(output / f"updates_rank{rank}.jsonl")}
        save_json(output / f"complete_rank{rank}.json", receipt)
        comm.synchronize()
    finally:
        native.Trainer = original
        for trainer in instances:
            trainer.pair_stream.close()
            trainer.update_stream.close()
            if trainer.reference_stream:
                trainer.reference_stream.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--arm", required=True, choices=("A", "B"))
    args = p.parse_args()
    m = read_manifest(args.manifest)
    if torch.cuda.device_count() != 4:
        raise ValueError("Set CUDA_VISIBLE_DEVICES to exactly four GPUs")
    from detectron2.engine import launch
    launch(worker, num_gpus_per_machine=4, dist_url="auto", args=(args.manifest, args.arm))


if __name__ == "__main__":
    main()
