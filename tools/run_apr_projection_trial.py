#!/usr/bin/env python3
"""Run both APR/projection rounds unattended: A, P, R; three LVIS evaluations.

A: native barrier + balance + projection. P: only projection disabled.
R: projection disabled and ONLY the barrier coefficient zero; balance retained.
All arms restart from the same complete no-radius 8ep state, not from each other.
"""
from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch

from lami_dino.checkpoint_init import load_trusted_torch_file
from tools.apr_projection_trial_ops import ARMS,check_apr_record,verify_pair
from tools.compare_rare_pr_reports import file_identity,load_json,save_json
from tools.decoder_aux_ablation_ops import START,HORIZON,state_digest,validate_resume
from tools.diagnose_rare_fp_regions import fingerprint
from tools.evaluate_decoder_rollback import endpoint_state,evaluation_command,run_evaluation
from tools.gt_normalization_extension_ops import resume_identity
from tools.gt_normalization_trial_ops import RANK_PERIOD,normalization_plan
from tools.run_gt_normalization_trial import read_rows
from tools.run_tpa_formula_screen import collect_evaluation
from tools.summarize_eq2_counterfactual import METRICS
from tools.train_apr_projection_arm import read_manifest
from tools.train_gt_normalization_arm import read_manifest as read_reference

CODE_FILES = ("tools/run_apr_projection_trial.py","tools/train_apr_projection_arm.py",
              "tools/apr_projection_trial_ops.py","tools/gt_normalization_extension_ops.py")


def protected_inputs(m):
    return [m[k] for k in ("checkpoint","config","prompt_bank","reference_manifest",
                           "train_annotations","val_annotations")]+m["assets"]


def check_inputs(m):
    for identity in protected_inputs(m):
        print("[identity] "+identity["path"],flush=True)
        if file_identity(identity["path"]) != identity:
            raise ValueError("Verified input changed: "+identity["path"])


def prepare(args):
    output = Path(args.output_dir).expanduser().resolve()
    reference = Path(args.reference_trial).expanduser().resolve()
    if output.exists() or reference in output.parents or output in reference.parents:
        raise ValueError("Use a NEW output directory separate from the reference trial; do not pre-create it")
    if not 1 <= args.updates <= 2000 or args.num_gpus != 4 or args.cpu_threads < 1:
        raise ValueError("Require 1..2000 optimizer updates/arm, four GPUs and positive CPU threads")
    print(f"[prepare] reading verified 8ep initialization from {reference/'manifest.json'}",flush=True)
    parent = read_reference(reference/"manifest.json")
    if Path(parent["output_dir"]) != reference or args.cpu_threads != parent["cpu_threads"]:
        raise ValueError("Use the original reference directory and CPU thread setting")
    if Path(parent["checkpoint"]["path"]).resolve().parent in output.parents:
        raise ValueError("Output cannot be inside the source detector training directory")
    for identity in (parent["checkpoint"],parent["train_annotations"],parent["val_annotations"],*parent["assets"]):
        print("[identity] "+identity["path"],flush=True)
        if file_identity(identity["path"]) != identity:
            raise ValueError("Reference input changed: "+identity["path"])
    print("[prepare] verifying original no-radius 8ep model, AdamW, scheduler and AMP",flush=True)
    checkpoint = load_trusted_torch_file(parent["checkpoint"]["path"])
    state = endpoint_state(checkpoint,START-1)
    if validate_resume(checkpoint) != parent["resume"] or state_digest(state) != parent["model_digest"]:
        raise ValueError("Reference initialization state differs")
    del state,checkpoint
    gc.collect()
    code = dict(parent["code"])
    code.update({p:file_identity(ROOT/p)["sha256"] for p in CODE_FILES})
    m = {k:parent[k] for k in ("checkpoint","config","prompt_bank","assets","train_annotations","val_annotations",
                              "resume","model_digest","inventory","seed","num_gpus","cpu_threads","torch_version",
                              "start","lr_horizon","rank_guard","rank_period")}
    m.update(schema="apr_projection_trial_v1",output_dir=str(output),updates=args.updates,arms=ARMS,code=code,
             reference_manifest=file_identity(reference/"manifest.json"),scope=[
                 "All arms restore the SAME full no-radius 8ep model/optimizer/scheduler/AMP; never chain arms.",
                 "Round1 P-A isolates disabling conflict projection while keeping the complete APR loss.",
                 "Round2 R-P isolates disabling ONLY directional barrier, conditional on projection already disabled.",
                 "Usage balance=.03 and global APR weight1 remain active; R's runtime lambda_orth_base=0 is recorded explicitly.",
                 "Native micro GT normalization, no teacher, no radius, same LR horizon85200, AMP, native clipping/AdamW.",
                 "New paired training stream; not historical dataloader replay; no promise of bitwise deterministic CUDA.",
                 "Full-bank all/rare rank guards initially/every50/endpoint; failed guard stops the pipeline, never auto-retunes.",
                 "Each 2000-update endpoint is evaluated once on full LVIS. Intermediate losses are not performance evidence.",
                 "Single-seed continuation, NOT full 2x2 interaction, from-scratch attribution, or old-checkpoint provenance.",
             ])
    m["fingerprint"] = fingerprint(m)
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    # 3 full final checkpoints + 3 intermediate (500-update) checkpoints,
    # prediction JSONs and logs. No source checkpoint is copied/deleted.
    required = 6*m["checkpoint"]["bytes"]+6*1024**3
    if shutil.disk_usage(ancestor).free < required:
        raise ValueError(f"Need at least {required/1024**3:.1f} GiB free for retained results")
    output.mkdir(parents=True,exist_ok=False)
    for arm in ARMS:
        (output/arm).mkdir()
        with (output/arm/"last_checkpoint").open("x") as handle:
            handle.write(m["checkpoint"]["path"])
    save_json(output/"manifest.json",m)
    print(f"[prepared] three arms x {m['updates']} updates + three full LVIS evaluations; "
          f"start={START}, stop={START+m['updates']}, LR horizon={HORIZON}",flush=True)
    return m


def verify_arm(m,arm):
    directory = Path(m["output_dir"])/arm
    health = read_rows(directory/"rank_health.jsonl")
    expected_health = sorted({0,m["updates"],*range(RANK_PERIOD,m["updates"]+1,RANK_PERIOD)})
    if [r["update"] for r in health] != expected_health or not all(r["guard_pass"] for r in health):
        raise ValueError("Incomplete or failed prototype health checks")
    initial = {k:m["resume"][k] for k in ("optimizer","scheduler","scaler")}
    initial["model"] = m["model_digest"]
    receipts = []
    for rank in range(4):
        receipt = load_json(directory/f"complete_rank{rank}.json")
        if (not receipt.get("complete") or receipt["arm"] != arm or receipt["rank"] != rank
                or receipt["policy"] != ARMS[arm] or receipt["start"] != START
                or receipt["stop"] != START+m["updates"] or receipt["updates"] != m["updates"]
                or receipt["initial_state"] != initial or receipt["health_records"] != health
                or receipt["manifest_fingerprint"] != m["fingerprint"]):
            raise ValueError("Invalid APR/projection completion receipt")
        for key,name in (("transcript",f"pairing_rank{rank}.jsonl"),("update_log",f"updates_rank{rank}.jsonl")):
            if file_identity(directory/name) != receipt[key]:
                raise ValueError("Trial transcript changed")
        rows = read_rows(directory/f"pairing_rank{rank}.jsonl")
        updates = read_rows(directory/f"updates_rank{rank}.jsonl")
        if len(rows) != 2*m["updates"] or len(updates) != m["updates"]:
            raise ValueError("Wrong microbatch/update count")
        for i,row in enumerate(rows):
            plan = normalization_plan(row["normalization"]["global_gt_counts"],4,8)
            seed = m["seed"]+(i//2)*128+rank*8+(i%2)*2
            if (row["iteration"] != START+i//2 or row["micro"] != i%2 or row["data_seed"] != seed
                    or row["forward_seed"] != seed+1 or row["multiplier"] != 1.
                    or row["normalization"] != plan or row["policy"] != ARMS[arm]
                    or row["lrs"] != m["resume"]["lrs"]):
                raise ValueError("Changed data timeline/GT normalization/LR/intervention")
            check_apr_record(row["losses"],row["apr_components"],ARMS[arm])
        for i,row in enumerate(updates):
            if (row["iteration"] != START+i or row["update"] != i+1 or row["arm"] != arm
                    or row["policy"] != ARMS[arm] or row["normalization"] != rows[2*i]["normalization"]
                    or any(not math.isfinite(v) for v in row["preclip_norms"].values())
                    or (arm != "A" and row["routing"] != {"enabled":False})):
                raise ValueError("Invalid native optimizer update record")
        if arm != "A":
            reference = read_rows(directory.parent/"A"/f"pairing_rank{rank}.jsonl")
            if len(reference) != len(rows):
                raise ValueError("Control transcript length differs")
            for current,control in zip(rows,reference):
                verify_pair(current,control)
        receipts.append(receipt)
    path = directory/"model_final.pth"
    checkpoint = load_trusted_torch_file(path)
    endpoint_state(checkpoint,START+m["updates"]-1)
    resume_identity(checkpoint["trainer"],START+m["updates"]-1,m["resume"]["lrs"])
    expected_tag = {"arm":arm,"policy":ARMS[arm],"start":START,"updates":m["updates"],
                    "manifest_fingerprint":m["fingerprint"]}
    if checkpoint["trainer"].get("apr_projection_trial") != expected_tag:
        raise ValueError("Checkpoint does not certify the declared APR/projection intervention")
    del checkpoint
    gc.collect()
    return {"checkpoint":file_identity(path),"receipts":receipts,"health":health,"final_rank":health[-1]}


def status(m,state,phase,**extra):
    save_json(Path(m["output_dir"])/"STATUS.json",{
        "state":state,"phase":phase,"updated_unix":time.time(),"manifest_fingerprint":m["fingerprint"],**extra})


def write_summary(m,stages,evaluations,*,complete=False):
    changes = {}
    for name,later,earlier in (("round1_P_minus_A","P","A"),("round2_R_minus_P","R","P"),("R_minus_A","R","A")):
        if later in evaluations and earlier in evaluations:
            changes[name] = {k:evaluations[later]["metrics"][k]-evaluations[earlier]["metrics"][k] for k in METRICS}
    result = {"complete":complete,"manifest_fingerprint":m["fingerprint"],"updates_per_arm":m["updates"],
              "arms":ARMS,"training":stages,"evaluations":evaluations,"comparisons":changes,"scope":m["scope"],
              "decision":"NO_AUTOMATIC_ADOPTION_OR_FURTHER_TRAINING"}
    save_json(Path(m["output_dir"])/"summary.json",result)
    lines = ["APR / conflict-projection paired trial",f"complete={complete}; updates/arm={m['updates']}",
             "A=full APR + projection; P=full APR, no projection; R=balance only, no projection",
             "arm        AP       APr       APc       APf   rare-rank   guard"]
    for arm,evaluation in evaluations.items():
        metrics,health = evaluation["metrics"],stages[arm]["final_rank"]
        lines.append(f"{arm:3} "+" ".join(f"{metrics[k]:9.4f}" for k in ("AP","APr","APc","APf"))
                     +f" {health['rare']['mean_rank']:11.4f}   {health['guard_pass']}")
    for name,delta in changes.items():
        lines.append(f"{name}: delta_AP={delta['AP']:+.4f} delta_APr={delta['APr']:+.4f}")
    lines.append("Round2 is conditional on projection disabled; not a full interaction or historical-causation test.")
    Path(m["output_dir"],"results.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n"+"\n".join(lines),flush=True)
    return result


def execute(m):
    check_inputs(m)
    output = Path(m["output_dir"])
    stages,evaluations = {},{}
    for arm in ARMS:
        read_manifest(output/"manifest.json")
        directory = output/arm
        status(m,"RUNNING",f"train_{arm}")
        print(f"[stage train_{arm}] {m['updates']} updates; policy={ARMS[arm]}",flush=True)
        if not (directory/"complete_rank0.json").exists():
            if {p.name for p in directory.iterdir()} != {"last_checkpoint"}:
                raise ValueError(f"Incomplete arm {arm}; no unsafe partial resume or overwrite")
            command = [sys.executable,"-u",str(ROOT/"tools/train_apr_projection_arm.py"),
                       "--manifest",str(output/"manifest.json"),"--arm",arm]
            run_evaluation(command,directory)
        stages[arm] = verify_arm(m,arm)
        write_summary(m,stages,evaluations)
        # Finish round1 BEFORE starting round2. Its results survive an R guard
        # failure, while the overall status remains FAILED/incomplete.
        status(m,"RUNNING",f"evaluate_{arm}")
        print(f"[stage evaluate_{arm}] full LVIS, unchanged inference protocol",flush=True)
        directory = output/f"eval_{arm}"
        checkpoint = stages[arm]["checkpoint"]
        receipt = directory/"verified_result.json"
        if receipt.exists():
            prior = load_json(receipt)
            evaluations[arm] = collect_evaluation(directory)
            if (prior["checkpoint"] != checkpoint or prior["manifest_fingerprint"] != m["fingerprint"]
                    or prior["result"] != evaluations[arm]):
                raise ValueError("Stale evaluation receipt")
        else:
            if directory.exists():
                raise ValueError("Incomplete evaluation; no silent overwrite")
            directory.mkdir()
            command = evaluation_command(m["config"]["path"],checkpoint["path"],directory,4)
            save_json(directory/"command.json",command)
            run_evaluation(command,directory)
            evaluations[arm] = collect_evaluation(directory)
            save_json(receipt,{"checkpoint":checkpoint,"manifest_fingerprint":m["fingerprint"],"result":evaluations[arm]})
        write_summary(m,stages,evaluations)
    check_inputs(m)
    result = write_summary(m,stages,evaluations,complete=True)
    status(m,"COMPLETE","all_done",summary=str(output/"summary.json"))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-trial",required=True,help="Original 500-update GT-normalization trial; only its verified 8ep initialization/protocol is reused")
    p.add_argument("--output-dir",required=True)
    p.add_argument("--updates",type=int,default=2000)
    p.add_argument("--num-gpus",type=int,default=4)
    p.add_argument("--cpu-threads",type=int,default=2)
    p.add_argument("--prepare-only",action="store_true")
    p.add_argument("--execute-prepared",action="store_true")
    args = p.parse_args(argv)
    if args.cpu_threads < 1:
        raise ValueError("Positive CPU threads required")
    torch.set_num_threads(args.cpu_threads)
    m = None
    try:
        if args.execute_prepared:
            candidate = read_manifest(Path(args.output_dir)/"manifest.json")
            if (Path(args.output_dir).resolve() != Path(candidate["output_dir"])
                    or file_identity(Path(args.reference_trial)/"manifest.json") != candidate["reference_manifest"]
                    or any(getattr(args,k) != candidate[k] for k in ("updates","num_gpus","cpu_threads"))):
                raise ValueError("Prepared settings/reference differ")
            m = candidate
        else:
            m = prepare(args)
        if args.prepare_only:
            status(m,"PREPARED","no_gpu_work")
            return m
        return execute(m)
    except (Exception,KeyboardInterrupt) as exc:
        if m is not None:
            previous = load_json(Path(m["output_dir"])/"STATUS.json") if (Path(m["output_dir"])/"STATUS.json").exists() else {}
            status(m,"FAILED",previous.get("phase","prepare"),error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
