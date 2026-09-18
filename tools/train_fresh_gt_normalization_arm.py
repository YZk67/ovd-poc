#!/usr/bin/env python3
"""Internal four-rank worker for a fresh-start GT-normalization arm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch

from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.decoder_aux_ablation_ops import HORIZON
from tools.diagnose_rare_fp_regions import fingerprint
from tools.fresh_gt_normalization_ops import FRESH_RANK_GUARD, make_trainer_class
from tools.gt_normalization_trial_ops import ARMS, RANK_PERIOD


def training_options(manifest, arm):
    output = str(Path(manifest["output_dir"])/arm)
    return [
        f"train.output_dir={json.dumps(output)}", f"dataloader.evaluator.output_dir={json.dumps(output)}",
        f"train.init_checkpoint={json.dumps(manifest['initial_checkpoint']['path'])}",
        "train.init_checkpoint_scope=backbone_only", f"train.max_iter={manifest['updates']}",
        f"train.lr_scheduler_max_iter={HORIZON}", "train.gradient_accumulation_steps=2",
        "dataloader.train.total_batch_size=16", "dataloader.train.num_workers=0",
        "train.seed=42", "train.amp.enabled=True", "train.eval_period=0",
        "train.eval_after_train=False", f"train.checkpointer.period={manifest['updates']+1}",
        "train.log_period=50", "train.device=cuda", "model.device=cuda",
        "model.alpha=0.0", "model.beta=0.3", "model.novel_scale=3.0",
        "model.classifier.tpa_num_prototypes=5", "model.classifier.tpa_tau=0.004375",
        "model.classifier.tpa_cls_tau=0.07", "model.classifier.tpa_slot_prior_strength=0.2",
        "model.classifier.tpa_prototype_mode_strength=0.0",
        "model.classifier.tpa_train_aggregation=calibrated",
        "model.classifier.tpa_eval_legacy_logsumexp=False",
        "model.classifier.tpa_eval_logit_bias=0.0", "model.tpa_eval_mode_scale=1.0",
        "model.soft_category_topk=3", "model.inference_query_class_topk=0",
        "model.select_box_nums_for_evaluation=300", "model.score_ensemble=True",
    ]


def read_manifest(path):
    manifest = load_json(path)
    if fingerprint({k:v for k,v in manifest.items() if k != "fingerprint"}) != manifest["fingerprint"]:
        raise ValueError("Fresh trial manifest fingerprint mismatch")
    if (manifest.get("schema") != "fresh_gt_normalization_ab_v1" or manifest["updates"] != 2000
            or manifest["num_gpus"] != 4 or manifest["seed"] != 42 or manifest["arms"] != ARMS
            or manifest["lr_horizon"] != HORIZON or manifest["rank_guard"] != FRESH_RANK_GUARD
            or manifest["rank_period"] != RANK_PERIOD):
        raise ValueError("Unexpected fresh paired protocol")
    if str(torch.__version__) != manifest["torch_version"]:
        raise ValueError("Use the prepared lami/PyTorch environment")
    for key in ("config","prompt_bank","initial_checkpoint","train_annotations","val_annotations"):
        if file_identity(manifest[key]["path"]) != manifest[key]:
            raise ValueError("Fresh trial input changed: "+key)
    for identity in manifest["assets"]:
        if file_identity(identity["path"]) != identity:
            raise ValueError("Fresh trial asset changed: "+identity["path"])
    for relative,digest in manifest["code"].items():
        target = (ROOT/relative).resolve()
        if ROOT not in target.parents or file_identity(target)["sha256"] != digest:
            raise ValueError("Fresh trial code changed: "+relative)
    return manifest


def worker(manifest, arm):
    from detectron2.config import LazyConfig
    from detectron2.engine import default_setup
    from detectron2.utils import comm
    from tools import train_net as native
    from tools.audit_accumulation_objective import parameter_groups

    torch.set_num_threads(manifest["cpu_threads"])
    rank = comm.get_rank()
    if comm.get_world_size() != 4:
        raise ValueError("Require four GPUs, physical batch16, accumulation2")
    output = Path(manifest["output_dir"])/arm
    # The parent runner opens console.log before spawning this worker.
    if {path.name for path in output.iterdir()} not in (set(),{"console.log"}):
        raise ValueError("Fresh arm directory contains prior state; no resume or overwrite")
    cfg = LazyConfig.apply_overrides(LazyConfig.load(manifest["config"]["path"]),
                                     training_options(manifest,arm))
    if (cfg.dataloader.train.dataset.names != "lvis_v1_train_norare"
            or not cfg.dataloader.train.mapper.is_train or not cfg.model.use_fed_loss
            or cfg.model.cluster_fed_loss or cfg.model.get("teacher_rpsa",False)
            or cfg.train.init_checkpoint_scope != "backbone_only"
            or Path(cfg.train.init_checkpoint).resolve() != Path(manifest["initial_checkpoint"]["path"]).resolve()
            or Path(cfg.model.query_path).resolve() != Path(manifest["prompt_bank"]["path"]).resolve()
            or Path(cfg.model.eval_query_path).resolve() != Path(manifest["prompt_bank"]["path"]).resolve()):
        raise ValueError("Fresh CLIP-only/dataset/FedLoss/prompt protocol changed")
    args = SimpleNamespace(config_file=manifest["config"]["path"],resume=False,eval_only=False,
                           num_gpus=4,num_machines=1,machine_rank=0,opts=[])
    default_setup(cfg,args)
    torch.backends.cudnn.benchmark = False
    prompts = torch.from_numpy(np.load(manifest["prompt_bank"]["path"],allow_pickle=False)).float()
    if tuple(prompts.shape) != (1203,8,768):
        raise ValueError("Unexpected prompt bank")
    instances = []
    original = native.Trainer

    def construct(*a, **kw):
        # Inventory is determined before the backbone-only load, but trainable
        # scope is structural and the actual weights are captured at step zero.
        model = a[0] if a else kw["model"]
        raw = model.module if hasattr(model,"module") else model
        inventory,_,_ = parameter_groups(raw)
        cls = make_trainer_class(original,manifest,arm,rank,prompts,inventory)
        trainer = cls(*a,**kw)
        instances.append(trainer)
        return trainer

    native.Trainer = construct
    try:
        native.do_train(args,cfg)
        if len(instances) != 1:
            raise ValueError("Expected one fresh trainer")
        trainer = instances[0]
        if trainer.actual_updates != manifest["updates"] or trainer.iter != manifest["updates"]:
            raise ValueError("Fresh trial did not complete exact budget")
        if trainer.reference_stream and trainer.reference_stream.readline():
            raise ValueError("Fresh A transcript has unused records")
        for stream in (trainer.pair_stream,trainer.update_stream,trainer.health_stream):
            if stream is not None:
                stream.flush()
        save_json(output/f"complete_rank{rank}.json",{
            "complete":True,"arm":arm,"rank":rank,"start":0,"stop":trainer.iter,
            "updates":trainer.actual_updates,"initial_state":trainer.initial_state,
            "manifest_fingerprint":manifest["fingerprint"],"health_records":trainer.health_records,
            "initial_receipt":file_identity(output/f"initial_rank{rank}.json"),
            "transcript":file_identity(output/f"pairing_rank{rank}.jsonl"),
            "update_log":file_identity(output/f"updates_rank{rank}.jsonl")})
        comm.synchronize()
    finally:
        native.Trainer = original
        for trainer in instances:
            trainer.close_streams()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--arm",required=True,choices=tuple(ARMS))
    args = parser.parse_args()
    manifest = read_manifest(args.manifest)
    if torch.cuda.device_count() != 4:
        raise ValueError("Expose exactly four GPUs")
    from detectron2.engine import launch
    launch(worker,num_gpus_per_machine=4,dist_url="auto",args=(manifest,args.arm))


if __name__ == "__main__":
    main()
