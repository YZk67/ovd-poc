#!/usr/bin/env python3
"""Fresh CLIP-only A/B for 2000 updates, followed by two full LVIS evaluations."""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch

from lami_dino.checkpoint_init import load_trusted_torch_file
from tools.accumulation_objective_ops import normalization_plan
from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.decoder_aux_ablation_ops import HORIZON
from tools.diagnose_rare_fp_regions import fingerprint
from tools.evaluate_decoder_rollback import endpoint_state, evaluation_command, run_evaluation
from tools.fresh_gt_normalization_ops import verify_fresh_pair
from tools.gt_normalization_trial_ops import ARMS, RANK_GUARD, RANK_PERIOD
from tools.run_gt_normalization_trial import read_rows
from tools.run_tpa_formula_screen import collect_evaluation
from tools.summarize_eq2_counterfactual import METRICS
from tools.train_fresh_gt_normalization_arm import read_manifest

CODE_FILES = (
    "tools/run_fresh_gt_normalization_trial.py", "tools/train_fresh_gt_normalization_arm.py",
    "tools/fresh_gt_normalization_ops.py", "tools/train_net.py",
    "tools/gt_normalization_trial_ops.py", "tools/accumulation_objective_ops.py",
    "tools/decoder_loss_audit_ops.py", "tools/tpa_formula_screen_ops.py",
    "tools/tpa_geometry_audit_ops.py", "lami_dino/prototype_ops.py",
)


def repo_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT/path).resolve()


def prepare(args):
    output = Path(args.output_dir).expanduser().resolve()
    config = Path(args.config_file).expanduser().resolve()
    if output.exists():
        raise ValueError("Use a NEW output directory; do not pre-create or tee inside it")
    if args.updates != 2000 or args.num_gpus != 4 or args.seed != 42 or args.cpu_threads < 1:
        raise ValueError("Locked fresh screen: 2000 updates, four GPUs, seed42, positive CPU threads")
    from detectron2.config import LazyConfig
    from detectron2.data import MetadataCatalog

    cfg = LazyConfig.load(str(config))
    if (cfg.train.init_checkpoint_scope != "backbone_only" or cfg.train.max_iter != HORIZON
            or cfg.train.lr_scheduler_max_iter != HORIZON or cfg.train.gradient_accumulation_steps != 2
            or cfg.dataloader.train.dataset.names != "lvis_v1_train_norare"
            or not cfg.dataloader.train.mapper.is_train or not cfg.model.use_fed_loss
            or cfg.model.cluster_fed_loss or cfg.model.get("teacher_rpsa",False)
            or not cfg.train.tpa_conflict_projection or not cfg.train.separate_tpa_grad_clip
            or not cfg.train.clip_grad.enabled
            or dict(cfg.train.clip_grad.params) != {"max_norm":.5,"norm_type":2}):
        raise ValueError("Config is not the formal fresh CLIP-only 12ep/no-rare training protocol")
    initial = file_identity(repo_path(cfg.train.init_checkpoint))
    prompt_path = repo_path(cfg.model.query_path)
    if prompt_path != repo_path(cfg.model.eval_query_path):
        raise ValueError("Train/eval prompt bank differs")
    prompt = file_identity(prompt_path)
    asset_paths = {repo_path(cfg.model[key]) for key in (
        "vlm_query_path","clip_head_path","seen_classes","all_classes","cat_freq_path") if cfg.model.get(key)}
    asset_paths.update(repo_path(cfg.model.classifier[key]) for key in (
        "zs_weight_path","eval_zs_weight_path","text_embed_path","eval_text_embed_path")
        if cfg.model.classifier.get(key) and repo_path(cfg.model.classifier[key]) != prompt_path)
    assets = [file_identity(path) for path in sorted(asset_paths)]
    train_ann = file_identity(MetadataCatalog.get("lvis_v1_train_norare").json_file)
    val_ann = file_identity(MetadataCatalog.get("lvis_v1_val").json_file)
    code = {name:file_identity(ROOT/name)["sha256"] for name in CODE_FILES}
    manifest = {
        "schema":"fresh_gt_normalization_ab_v1","output_dir":str(output),
        "config":file_identity(config),"initial_checkpoint":initial,"prompt_bank":prompt,
        "assets":assets,"train_annotations":train_ann,"val_annotations":val_ann,
        "updates":2000,"num_gpus":4,"seed":42,"cpu_threads":args.cpu_threads,
        "torch_version":str(torch.__version__),"arms":ARMS,"lr_horizon":HORIZON,
        "rank_guard":RANK_GUARD,"rank_period":RANK_PERIOD,"code":code,
        "scope":[
            "Both arms start from seed42 random detector/TPA plus the same backbone-only CLIP checkpoint.",
            "A uses native per-micro GT normalization; B pools GT across two physical microbatches.",
            "Only final/aux/encoder/DN detection class/L1/GIoU multipliers differ; APR/RPSA do not.",
            "Physical batch16, accumulation2, effective batch32; original 85200-step LR schedule starts at zero.",
            "No-radius K5, slot prior, APR routing, separate clipping and calibrated inference stay fixed.",
            "Initial model/optimizer/scheduler/scaler and every data/RNG/FedLoss record are paired.",
            "2000 updates are about 0.28 epoch: an early screen, not final 4ep/12ep evidence.",
        ],
    }
    manifest["fingerprint"] = fingerprint(manifest)
    required = 2*initial["bytes"] + 4*1024**3
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < required:
        raise ValueError(f"Need at least {required/1024**3:.1f} GiB free")
    output.mkdir(parents=True,exist_ok=False)
    for arm in ARMS:
        (output/arm).mkdir()
    save_json(output/"manifest.json",manifest)
    print("[prepared] fresh CLIP-only A/B; 2000 updates/arm; LR timeline 0..1999/85200",flush=True)
    return manifest


def verify_arm(manifest, arm):
    directory = Path(manifest["output_dir"])/arm
    health = read_rows(directory/"rank_health.jsonl")
    expected_health = sorted({0,manifest["updates"],*range(RANK_PERIOD,manifest["updates"]+1,RANK_PERIOD)})
    if [row["update"] for row in health] != expected_health or not all(row["guard_pass"] for row in health):
        raise ValueError("Incomplete/failed fresh prototype rank checks")
    receipts = []
    for rank in range(4):
        receipt = load_json(directory/f"complete_rank{rank}.json")
        initial = load_json(directory/f"initial_rank{rank}.json")
        if (not receipt.get("complete") or receipt["arm"] != arm or receipt["rank"] != rank
                or receipt["start"] != 0 or receipt["stop"] != manifest["updates"]
                or receipt["updates"] != manifest["updates"] or receipt["initial_state"] != initial
                or receipt["manifest_fingerprint"] != manifest["fingerprint"]
                or receipt["health_records"] != health
                or file_identity(directory/f"initial_rank{rank}.json") != receipt["initial_receipt"]):
            raise ValueError("Invalid fresh-start completion receipt")
        for field,name in (("transcript",f"pairing_rank{rank}.jsonl"),
                           ("update_log",f"updates_rank{rank}.jsonl")):
            if file_identity(directory/name) != receipt[field]:
                raise ValueError("Fresh transcript changed")
        rows = read_rows(directory/f"pairing_rank{rank}.jsonl")
        updates = read_rows(directory/f"updates_rank{rank}.jsonl")
        if len(rows) != 2*manifest["updates"] or len(updates) != manifest["updates"]:
            raise ValueError("Incomplete fresh microbatch/update count")
        for index,row in enumerate(rows):
            plan = normalization_plan(row["normalization"]["global_gt_counts"],4,8)
            seed = manifest["seed"]+(index//2)*128+rank*8+(index%2)*2
            multiplier = plan["detection_loss_multipliers"][index%2] if arm == "B" else 1.
            if (row["iteration"] != index//2 or row["micro"] != index%2
                    or row["data_seed"] != seed or row["forward_seed"] != seed+1
                    or row["normalization"] != plan or row["multiplier"] != multiplier
                    or not row["lrs"] or any(not math.isfinite(value) or value <= 0 for value in row["lrs"])):
                raise ValueError("Fresh timeline/seed/normalization/LR differs")
        for index,row in enumerate(updates):
            if (row["iteration"] != index or row["update"] != index+1 or row["arm"] != arm
                    or row["normalization"] != rows[2*index]["normalization"]
                    or row["normalization"] != rows[2*index+1]["normalization"]
                    or any(not math.isfinite(value) for value in row["preclip_norms"].values())):
                raise ValueError("Invalid fresh native update record")
        if arm == "B":
            reference = read_rows(directory.parent/"A"/f"pairing_rank{rank}.jsonl")
            if len(reference) != len(rows):
                raise ValueError("Fresh A/B transcript lengths differ")
            for current,prior in zip(rows,reference):
                verify_fresh_pair(current,prior)
        receipts.append(receipt)
    # DDP and both arms must begin with exactly the same post-CLIP-load state.
    states = [receipt["initial_state"] for receipt in receipts]
    if any(state != states[0] for state in states[1:]):
        raise ValueError("Fresh DDP ranks did not share one initialization")
    if arm == "B":
        for rank,state in enumerate(states):
            if state != load_json(directory.parent/"A"/f"initial_rank{rank}.json"):
                raise ValueError("Fresh A/B initial state differs")
    path = directory/"model_final.pth"
    checkpoint = load_trusted_torch_file(path)
    endpoint_state(checkpoint,manifest["updates"]-1)
    trainer = checkpoint.get("trainer",{})
    expected = {"arm":arm,"normalization":ARMS[arm],"start":0,
                "updates":manifest["updates"],"manifest_fingerprint":manifest["fingerprint"],
                "initial_state":states[0]}
    if (trainer.get("fresh_gt_normalization_trial") != expected
            or trainer.get("iteration") != manifest["updates"]-1
            or trainer.get("lr_scheduler_max_iter") != HORIZON
            or trainer.get("gradient_accumulation_steps") != 2
            or trainer.get("hooks",{}).get("LRScheduler",{}).get("last_epoch") != manifest["updates"]
            or not trainer.get("optimizer",{}).get("state") or not trainer.get("grad_scaler")):
        raise ValueError("Final checkpoint lacks complete fresh optimizer/provenance state")
    del checkpoint
    gc.collect()
    return {"checkpoint":file_identity(path),"initial_state":states[0],
            "receipts":receipts,"health":health,"final_rank":health[-1]}


def execute(manifest):
    protected = [manifest[key] for key in ("config","initial_checkpoint","prompt_bank",
                                           "train_annotations","val_annotations")]+manifest["assets"]
    for identity in protected:
        if file_identity(identity["path"]) != identity:
            raise ValueError("Prepared fresh input changed: "+identity["path"])
    output = Path(manifest["output_dir"])
    stages,evaluations = {},{}
    for arm in ARMS:
        directory = output/arm
        if not (directory/"complete_rank0.json").exists():
            if any(directory.iterdir()):
                raise ValueError(f"Incomplete fresh {arm}; retain it and use a new output directory")
            command = [sys.executable,"-u",str(ROOT/"tools/train_fresh_gt_normalization_arm.py"),
                       "--manifest",str(output/"manifest.json"),"--arm",arm]
            print(f"[fresh train {arm}] {ARMS[arm]}: 2000 optimizer updates",flush=True)
            run_evaluation(command,directory)
        stages[arm] = verify_arm(manifest,arm)
    if stages["A"]["initial_state"] != stages["B"]["initial_state"]:
        raise ValueError("Fresh A/B initial states differ")
    for arm in ARMS:
        directory = output/f"eval_{arm}"
        checkpoint = stages[arm]["checkpoint"]
        receipt_path = directory/"verified_result.json"
        if receipt_path.exists():
            prior = load_json(receipt_path)
            evaluations[arm] = collect_evaluation(directory)
            if (prior["checkpoint"] != checkpoint or prior["manifest_fingerprint"] != manifest["fingerprint"]
                    or prior["result"] != evaluations[arm]):
                raise ValueError("Stale fresh evaluation")
            continue
        if directory.exists():
            raise ValueError("Incomplete fresh evaluation; no overwrite")
        directory.mkdir()
        command = evaluation_command(manifest["config"]["path"],checkpoint["path"],directory,4)
        save_json(directory/"command.json",command)
        print(f"[evaluate {arm}] fresh 2000-update full LVIS",flush=True)
        run_evaluation(command,directory)
        evaluations[arm] = collect_evaluation(directory)
        save_json(receipt_path,{"checkpoint":checkpoint,"manifest_fingerprint":manifest["fingerprint"],
                                "result":evaluations[arm]})
    for identity in protected:
        if file_identity(identity["path"]) != identity:
            raise ValueError("Fresh source changed during trial")
    summary = {"complete":True,"manifest_fingerprint":manifest["fingerprint"],
               "updates_per_arm":manifest["updates"],"training":stages,"evaluations":evaluations,
               "delta_B_minus_A":{key:evaluations["B"]["metrics"][key]-evaluations["A"]["metrics"][key]
                                  for key in METRICS},"scope":manifest["scope"],
               "decision":"FRESH_2000_UPDATE_SCREEN_NOT_4EP_OR_12EP_EVIDENCE"}
    save_json(output/"summary.json",summary)
    print("\n=== Fresh-start paired GT normalization trial ===")
    for arm in ARMS:
        print(arm,evaluations[arm]["metrics"],"rank",stages[arm]["final_rank"])
    print("B - A",summary["delta_B_minus_A"])
    print(f"[save] {output/'summary.json'}",flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file",default="lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py")
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--updates",type=int,default=2000)
    parser.add_argument("--num-gpus",type=int,default=4)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--cpu-threads",type=int,default=2)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--execute-prepared",action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.cpu_threads)
    if args.execute_prepared:
        manifest = read_manifest(Path(args.output_dir)/"manifest.json")
        if (Path(args.output_dir).resolve() != Path(manifest["output_dir"])
                or file_identity(args.config_file) != manifest["config"]
                or any(getattr(args,key) != manifest[key] for key in
                       ("updates","num_gpus","seed","cpu_threads"))):
            raise ValueError("Prepared fresh settings/config differ")
    else:
        manifest = prepare(args)
    if args.prepare_only:
        print("[prepare-only] No training/evaluation launched",flush=True)
        return manifest
    return execute(manifest)


if __name__ == "__main__":
    main()
