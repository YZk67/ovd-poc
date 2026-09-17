#!/usr/bin/env python3
"""Internal four-GPU worker for run_tpa_formula_screen.py."""
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
from tools.decoder_aux_ablation_ops import HORIZON, START
from tools.diagnose_rare_fp_regions import fingerprint
from tools.tpa_formula_screen_ops import ARMS, make_trainer_class


def read_manifest(path):
    manifest = load_json(path)
    payload = {key: value for key, value in manifest.items() if key != "fingerprint"}
    if fingerprint(payload) != manifest["fingerprint"]:
        raise ValueError("Formula-screen manifest fingerprint mismatch")
    if not 1 <= manifest["updates"] <= 500 or manifest["num_gpus"] != 4 or manifest["seed"] != 42:
        raise ValueError("Unexpected formula-screen budget/protocol")
    if str(torch.__version__) != manifest["torch_version"]:
        raise ValueError("Use the same PyTorch environment as preparation")
    if file_identity(manifest["config"]["path"]) != manifest["config"]:
        raise ValueError("Config changed since preparation")
    for relative, digest in manifest["code"].items():
        target = (ROOT / relative).resolve()
        if ROOT not in target.parents or file_identity(target)["sha256"] != digest:
            raise ValueError(f"Formula-screen code changed: {relative}")
    return manifest


def training_options(manifest, arm):
    output = str(Path(manifest["output_dir"]) / arm)
    return [
        f"train.output_dir={json.dumps(output)}",
        f"dataloader.evaluator.output_dir={json.dumps(output)}",
        f"train.init_checkpoint={json.dumps(manifest['checkpoint']['path'])}",
        "train.init_checkpoint_scope=full",
        f"train.max_iter={START + manifest['updates']}",
        f"train.lr_scheduler_max_iter={HORIZON}",
        "train.gradient_accumulation_steps=2",
        "dataloader.train.total_batch_size=16",
        "dataloader.train.num_workers=0",
        "train.seed=42",
        "train.amp.enabled=True",
        "train.eval_period=0",
        "train.eval_after_train=False",
        f"train.checkpointer.period={START + manifest['updates']}",
        "train.log_period=50",
        "train.device=cuda",
        "model.device=cuda",
        "model.alpha=0.0",
        "model.beta=0.3",
        "model.novel_scale=3.0",
        "model.classifier.tpa_num_prototypes=5",
        "model.classifier.tpa_tau=0.004375",
        "model.classifier.tpa_cls_tau=0.07",
        "model.classifier.tpa_slot_prior_strength=0.2",
        "model.classifier.tpa_prototype_mode_strength=0.0",
        f"model.classifier.tpa_train_aggregation={json.dumps(ARMS[arm])}",
        # Evaluation remains calibrated for every arm.
        "model.classifier.tpa_eval_legacy_logsumexp=False",
        "model.classifier.tpa_eval_logit_bias=0.0",
        "model.tpa_eval_mode_scale=1.0",
        "model.soft_category_topk=3",
        "model.inference_query_class_topk=0",
        "model.select_box_nums_for_evaluation=300",
        "model.score_ensemble=True",
    ]


def worker(manifest_path, arm):
    from detectron2.config import LazyConfig
    from detectron2.engine import default_setup
    from detectron2.utils import comm
    from tools import train_net as native

    manifest = read_manifest(manifest_path)
    torch.set_num_threads(manifest["cpu_threads"])
    rank = comm.get_rank()
    if comm.get_world_size() != 4:
        raise ValueError("Expected four ranks, physical batch16, accumulation2")
    output = Path(manifest["output_dir"]) / arm
    if (output / "last_checkpoint").read_text().strip() != manifest["checkpoint"]["path"]:
        raise ValueError("Arm must start from the original full 8ep checkpoint")
    cfg = LazyConfig.load(manifest["config"]["path"])
    cfg = LazyConfig.apply_overrides(cfg, training_options(manifest, arm))
    args = SimpleNamespace(
        config_file=manifest["config"]["path"],
        resume=True,
        eval_only=False,
        num_gpus=4,
        num_machines=1,
        machine_rank=0,
        opts=[],
    )
    default_setup(cfg, args)
    torch.backends.cudnn.benchmark = False
    trainer_class = make_trainer_class(native.Trainer, manifest, arm, rank)
    instances = []

    def construct(*args, **kwargs):
        trainer = trainer_class(*args, **kwargs)
        instances.append(trainer)
        return trainer

    original = native.Trainer
    native.Trainer = construct
    try:
        native.do_train(args, cfg)
        if len(instances) != 1:
            raise ValueError("Expected one native trainer")
        trainer = instances[0]
        if trainer.actual_updates != manifest["updates"] or trainer.iter != START + manifest["updates"]:
            raise ValueError("Training did not complete the exact update budget")
        if trainer.reference_stream and trainer.reference_stream.readline():
            raise ValueError("Unconsumed A pairing records")
        trainer._assert_geometry_frozen()
        trainer.pair_stream.flush()
        trainer.update_stream.flush()
        receipt = {
            "complete": True,
            "arm": arm,
            "aggregation": ARMS[arm],
            "rank": rank,
            "updates": trainer.actual_updates,
            "start": START,
            "stop": trainer.iter,
            "initial_state": trainer.initial_state,
            "tpa_geometry_digest": trainer.initial_geometry_digest,
            "manifest_fingerprint": manifest["fingerprint"],
            "transcript": file_identity(output / f"pairing_rank{rank}.jsonl"),
            "update_log": file_identity(output / f"updates_rank{rank}.jsonl"),
        }
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--arm", required=True, choices=tuple(ARMS))
    args = parser.parse_args()
    manifest = read_manifest(args.manifest)
    if torch.cuda.device_count() != 4:
        raise ValueError("Set CUDA_VISIBLE_DEVICES to exactly four GPUs")
    from detectron2.engine import launch
    launch(worker, num_gpus_per_machine=4, dist_url="auto", args=(args.manifest, args.arm))


if __name__ == "__main__":
    main()
