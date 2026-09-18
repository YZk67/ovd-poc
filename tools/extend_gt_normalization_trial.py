#!/usr/bin/env python3
"""Extend completed A/B from 500 to 2000 updates, without repeating model updates.

Resume each arm's OWN full checkpoint; replay/verify only its data cursor.
Retain all original results, evaluate each new endpoint once, no parameter sweep.
"""
from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from lami_dino.checkpoint_init import load_trusted_torch_file
from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.decoder_aux_ablation_ops import START, HORIZON, state_digest
from tools.diagnose_rare_fp_regions import fingerprint
from tools.evaluate_decoder_rollback import endpoint_state, evaluation_command, run_evaluation
from tools.gt_normalization_extension_ops import resume_identity
from tools.gt_normalization_trial_ops import ARMS, RANK_PERIOD, normalization_plan, verify_pair
from tools.run_gt_normalization_trial import read_rows, verify_arm as verify_parent_arm
from tools.run_tpa_formula_screen import collect_evaluation
from tools.summarize_eq2_counterfactual import METRICS
from tools.train_gt_normalization_arm import read_manifest as read_parent
from tools.train_gt_normalization_extension import read_manifest

CODE_FILES = ("tools/extend_gt_normalization_trial.py", "tools/train_gt_normalization_extension.py",
              "tools/gt_normalization_extension_ops.py")


def prepare(args):
    parent_dir, output = Path(args.parent_dir).expanduser().resolve(), Path(args.output_dir).expanduser().resolve()
    if output.exists() or output in parent_dir.parents or parent_dir in output.parents:
        raise ValueError("Use a NEW output directory separate from the original 500-update trial")
    if not 500 < args.total_updates <= 2000 or args.num_gpus != 4 or args.cpu_threads < 1:
        raise ValueError("Require four GPUs, 501..2000 TOTAL updates, positive CPU threads")
    parent = read_parent(parent_dir/"manifest.json")  # All original code remains unchanged.
    summary = load_json(parent_dir/"summary.json")
    if (Path(parent["output_dir"]) != parent_dir or parent["updates"] != 500
            or not summary.get("complete") or summary["updates_per_arm"] != 500
            or summary["manifest_fingerprint"] != parent["fingerprint"]):
        raise ValueError("Require the original completed 500-update A/B summary/manifest")
    if args.cpu_threads != parent["cpu_threads"]:
        raise ValueError("Keep the parent's CPU thread setting")
    if Path(parent["checkpoint"]["path"]).resolve().parent in output.parents:
        raise ValueError("Do not write inside the original detector training directory")
    for identity in (parent["train_annotations"], parent["val_annotations"], *parent["assets"]):
        if file_identity(identity["path"]) != identity:
            raise ValueError("Parent training/evaluation asset changed: " + identity["path"])
    sources, metrics = {}, {}
    for arm in ARMS:
        print(f"[verify parent {arm}] checkpoint, all-rank pairing and evaluation receipts", flush=True)
        stage = verify_parent_arm(parent, arm)
        evaluation = collect_evaluation(parent_dir/f"eval_{arm}")
        if stage != summary["training"][arm] or evaluation != summary["evaluations"][arm]:
            raise ValueError("Parent checkpoint/receipts/evaluation do not match its completed summary")
        ckpt = load_trusted_torch_file(stage["checkpoint"]["path"])
        state = endpoint_state(ckpt, START+500-1)
        resume = resume_identity(ckpt["trainer"], START+500-1, parent["resume"]["lrs"])
        sources[arm] = {"checkpoint":stage["checkpoint"], "resume":resume, "model_digest":state_digest(state),
                        "transcripts":[r["transcript"] for r in stage["receipts"]],
                        "final_rank":stage["final_rank"]}
        metrics[arm] = evaluation["metrics"]
        del state, ckpt
        gc.collect()
    code = dict(parent["code"])
    code.update({p:file_identity(ROOT/p)["sha256"] for p in CODE_FILES})
    m = {k:parent[k] for k in ("config","prompt_bank","assets","train_annotations","val_annotations",
                             "inventory","seed","num_gpus","cpu_threads","torch_version","arms",
                             "lr_horizon","rank_guard","rank_period")}
    m.update(schema="gt_normalization_extension_v1", output_dir=str(output),
             parent_manifest=file_identity(parent_dir/"manifest.json"), parent_summary=file_identity(parent_dir/"summary.json"),
             parent_fingerprint=parent["fingerprint"], completed_updates=500, total_updates=args.total_updates,
             additional_updates=args.total_updates-500, start=START+500, sources=sources, metrics_at_500=metrics,
             code=code, scope=parent["scope"]+[
                 "Resume A500->A2000 and B500->B2000, with each arm's own full optimizer/scheduler/scaler.",
                 "Replay the first 500 DATA windows with original seeds and hash checks; NO model forwards/backwards/updates.",
                 "Then continue the original seed/data timeline, retain LR horizon85200, rank guards and all loss policies.",
                 "No claim of bitwise CUDA equality with an uninterrupted process; no automatic full 12ep run."])
    m["fingerprint"] = fingerprint(m)
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    required = sum(s["checkpoint"]["bytes"] for s in sources.values()) + 4*1024**3
    if shutil.disk_usage(ancestor).free < required:
        raise ValueError(f"Need {required/1024**3:.1f} GiB free, in addition to retained parent outputs")
    output.mkdir(parents=True, exist_ok=False)
    for arm in ARMS:
        (output/arm).mkdir()
        with (output/arm/"last_checkpoint").open("x") as handle:
            handle.write(sources[arm]["checkpoint"]["path"])
    save_json(output/"manifest.json",m)
    print(f"[prepared] {m['additional_updates']} NEW updates/arm; cumulative={args.total_updates}; "
          f"iter {m['start']}..{START+args.total_updates-1}; LR horizon={HORIZON}",flush=True)
    return m


def verify_arm(m, arm):
    directory = Path(m["output_dir"])/arm
    completed, total, start = m["completed_updates"], m["total_updates"], m["start"]
    source = m["sources"][arm]
    expected_initial = {k:source["resume"][k] for k in ("optimizer","scheduler","scaler")}
    expected_initial["model"] = source["model_digest"]
    health = read_rows(directory/"rank_health.jsonl")
    checkpoints = sorted({completed,total,*range(completed+RANK_PERIOD,total+1,RANK_PERIOD)})
    if [r["update"] for r in health] != checkpoints or not all(r["guard_pass"] for r in health):
        raise ValueError("Incomplete/failed extension rank checks")
    receipts = []
    for rank in range(4):
        r = load_json(directory/f"complete_rank{rank}.json")
        replay = {"verified":True, "windows":completed, "microbatches":completed*2,
                  "model_forwards":0, "optimizer_updates":0, "state_unchanged":True,
                  "transcript":source["transcripts"][rank]}
        if (not r.get("complete") or r["arm"] != arm or r["rank"] != rank or r["start"] != start
                or r["stop"] != START+total or r["updates"] != total-completed or r["total_updates"] != total
                or r["initial_state"] != expected_initial or r["cursor_replay"] != replay
                or r["manifest_fingerprint"] != m["fingerprint"] or r["health_records"] != health):
            raise ValueError("Unverified extension resume/cursor/update receipt")
        for key,name in (("transcript",f"pairing_rank{rank}.jsonl"),("update_log",f"updates_rank{rank}.jsonl")):
            if file_identity(directory/name) != r[key]:
                raise ValueError("Extension transcript changed")
        rows, updates = read_rows(directory/f"pairing_rank{rank}.jsonl"), read_rows(directory/f"updates_rank{rank}.jsonl")
        if len(rows) != 2*(total-completed) or len(updates) != total-completed:
            raise ValueError("Incomplete extension batch/update count")
        for i,row in enumerate(rows):
            plan = normalization_plan(row["normalization"]["global_gt_counts"],4,8)
            seed = m["seed"]+(completed+i//2)*128+rank*8+(i%2)*2
            factor = plan["detection_loss_multipliers"][i%2] if arm == "B" else 1.
            if (row["iteration"] != start+i//2 or row["micro"] != i%2 or row["data_seed"] != seed
                    or row["forward_seed"] != seed+1 or row["normalization"] != plan
                    or row["multiplier"] != factor or row["lrs"] != source["resume"]["lrs"]):
                raise ValueError("Extension timeline/seed/normalization/LR differs")
        for i,row in enumerate(updates):
            if (row["iteration"] != start+i or row["update"] != completed+i+1 or row["arm"] != arm
                    or row["normalization"] != rows[2*i]["normalization"]
                    or row["normalization"] != rows[2*i+1]["normalization"]
                    or any(not math.isfinite(v) for v in row["preclip_norms"].values())):
                raise ValueError("Invalid native extension update record")
        if arm == "B":
            prior = read_rows(directory.parent/"A"/f"pairing_rank{rank}.jsonl")
            if len(prior) != len(rows):
                raise ValueError("A/B extension transcript lengths differ")
            for b,a in zip(rows,prior):
                verify_pair(b,a)
        receipts.append(r)
    path = directory/"model_final.pth"
    ckpt = load_trusted_torch_file(path)
    endpoint_state(ckpt,START+total-1)
    resume_identity(ckpt["trainer"],START+total-1,source["resume"]["lrs"])
    tag = {"arm":arm,"normalization":ARMS[arm],"start":START,"updates":total,"manifest_fingerprint":m["fingerprint"]}
    extension = {"parent_fingerprint":m["parent_fingerprint"],"completed_updates":completed,
                 "additional_updates":total-completed,"total_updates":total,"manifest_fingerprint":m["fingerprint"],
                 "cursor_replay":receipts[0]["cursor_replay"]}
    if ckpt["trainer"].get("gt_normalization_trial") != tag or ckpt["trainer"].get("gt_normalization_extension") != extension:
        raise ValueError("Final checkpoint lacks verified extension provenance")
    del ckpt
    gc.collect()
    return {"checkpoint":file_identity(path),"receipts":receipts,"health":health,"final_rank":health[-1]}


def execute(m):
    protected = [m["parent_manifest"],m["parent_summary"],m["train_annotations"],m["val_annotations"],*m["assets"]]
    for source in m["sources"].values():
        protected.extend([source["checkpoint"],*source["transcripts"]])
    for identity in protected:
        if file_identity(identity["path"]) != identity:
            raise ValueError("Verified extension input changed: " + identity["path"])
    output = Path(m["output_dir"])
    stages, evaluations = {}, {}
    for arm in ARMS:
        directory = output/arm
        if not (directory/"complete_rank0.json").exists():
            if {p.name for p in directory.iterdir()} != {"last_checkpoint"}:
                raise ValueError(f"Incomplete extension {arm}; no unsafe mid-arm resume or overwrite")
            command = [sys.executable,"-u",str(ROOT/"tools/train_gt_normalization_extension.py"),
                       "--manifest",str(output/"manifest.json"),"--arm",arm]
            print(f"[extend {arm}] replay DATA cursor then {m['additional_updates']} NEW optimizer updates",flush=True)
            run_evaluation(command,directory)
        stages[arm] = verify_arm(m,arm)
    for arm in ARMS:
        directory, checkpoint = output/f"eval_{arm}", stages[arm]["checkpoint"]
        receipt = directory/"verified_result.json"
        if receipt.exists():
            prior = load_json(receipt)
            evaluations[arm] = collect_evaluation(directory)
            if (prior["checkpoint"] != checkpoint or prior["manifest_fingerprint"] != m["fingerprint"]
                    or prior["result"] != evaluations[arm]):
                raise ValueError("Stale extended evaluation")
            continue
        if directory.exists():
            raise ValueError("Incomplete extended evaluation; inspect logs, no overwrite")
        directory.mkdir()
        command = evaluation_command(m["config"]["path"],checkpoint["path"],directory,4)
        save_json(directory/"command.json",command)
        print(f"[evaluate {arm}] cumulative update {m['total_updates']}, full native LVIS",flush=True)
        run_evaluation(command,directory)
        evaluations[arm] = collect_evaluation(directory)
        save_json(receipt,{"checkpoint":checkpoint,"manifest_fingerprint":m["fingerprint"],"result":evaluations[arm]})
    for identity in (m["parent_manifest"],m["parent_summary"],m["val_annotations"],
                     *(s["checkpoint"] for s in m["sources"].values())):
        if file_identity(identity["path"]) != identity:
            raise ValueError("Protected parent/evaluation input changed during extension")
    summary = {"complete":True,"manifest_fingerprint":m["fingerprint"],"total_updates_per_arm":m["total_updates"],
               "additional_updates_per_arm":m["additional_updates"],"training":stages,"evaluations":evaluations,
               "metrics_at_500":m["metrics_at_500"],"scope":m["scope"],
               "delta_B_minus_A":{k:evaluations["B"]["metrics"][k]-evaluations["A"]["metrics"][k] for k in METRICS},
               "change_from_500":{a:{k:evaluations[a]["metrics"][k]-m["metrics_at_500"][a][k] for k in METRICS} for a in ARMS}}
    save_json(output/"summary.json",summary)
    print("\n=== Extended paired GT normalization trial ===")
    for arm in ARMS:
        print(arm,evaluations[arm]["metrics"],"rank",stages[arm]["final_rank"])
    print("B - A",summary["delta_B_minus_A"])
    print("Change from 500",summary["change_from_500"])
    print(f"[save] {output/'summary.json'}",flush=True)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-dir",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--total-updates",type=int,default=2000)
    p.add_argument("--num-gpus",type=int,default=4)
    p.add_argument("--cpu-threads",type=int,default=2)
    p.add_argument("--prepare-only",action="store_true")
    p.add_argument("--execute-prepared",action="store_true")
    args = p.parse_args(argv)
    if args.cpu_threads < 1:
        raise ValueError("Positive CPU threads required")
    torch.set_num_threads(args.cpu_threads)
    if args.execute_prepared:
        m = read_manifest(Path(args.output_dir)/"manifest.json")
        if (Path(args.output_dir).resolve() != Path(m["output_dir"])
                or file_identity(Path(args.parent_dir)/"manifest.json") != m["parent_manifest"]
                or any(getattr(args,k) != m[k] for k in ("total_updates","num_gpus","cpu_threads"))):
            raise ValueError("Prepared extension settings/parent differ")
    else:
        m = prepare(args)
    if args.prepare_only:
        print("[prepare-only] No training/evaluation launched",flush=True)
        return m
    return execute(m)


if __name__ == "__main__":
    main()
