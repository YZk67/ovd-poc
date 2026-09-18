#!/usr/bin/env python3
"""Extend A/P from 2000 to 4000 TOTAL updates, then evaluate both on full LVIS.

Resume each arm's own complete state, reconstruct its data cursor without model
updates, retain all parent results. No R run, new hyperparameters or auto-adoption.
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
from tools.apr_projection_extension_ops import ARMS, cursor_receipt, extension_tag, trial_tag
from tools.apr_projection_trial_ops import check_apr_record, verify_pair
from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.decoder_aux_ablation_ops import START, HORIZON, state_digest
from tools.diagnose_rare_fp_regions import fingerprint
from tools.evaluate_decoder_rollback import endpoint_state, evaluation_command, run_evaluation
from tools.gt_normalization_extension_ops import resume_identity
from tools.gt_normalization_trial_ops import RANK_PERIOD, normalization_plan
from tools.run_apr_projection_trial import status, verify_arm as verify_parent_arm
from tools.run_gt_normalization_trial import read_rows
from tools.run_tpa_formula_screen import collect_evaluation
from tools.summarize_eq2_counterfactual import METRICS
from tools.train_apr_projection_arm import read_manifest as read_parent
from tools.train_apr_projection_extension import read_manifest

CODE_FILES = ("tools/extend_apr_projection_trial.py", "tools/train_apr_projection_extension.py",
              "tools/apr_projection_extension_ops.py")


def protected_inputs(m):
    identities = [m[k] for k in ("parent_manifest", "parent_summary", "config", "prompt_bank",
                                "train_annotations", "val_annotations")]+m["assets"]
    for source in m["sources"].values():
        identities.extend([source["checkpoint"], *source["transcripts"]])
    return identities


def check_inputs(m):
    for identity in protected_inputs(m):
        print("[identity] "+identity["path"], flush=True)
        if file_identity(identity["path"]) != identity:
            raise ValueError("Verified extension input changed: "+identity["path"])


def prepare(args):
    parent_dir = Path(args.parent_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() or parent_dir in output.parents or output in parent_dir.parents:
        raise ValueError("Use a NEW output directory separate from the original APR trial")
    if not 2000 < args.total_updates <= 4000 or args.num_gpus != 4 or args.cpu_threads < 1:
        raise ValueError("Require four GPUs, 2001..4000 TOTAL updates, positive CPU threads")
    print(f"[prepare] verifying A/P endpoints from {parent_dir}", flush=True)
    parent = read_parent(parent_dir/"manifest.json")
    summary = load_json(parent_dir/"summary.json")
    if (Path(parent["output_dir"]) != parent_dir or parent["updates"] != 2000
            or not summary.get("complete") or summary["updates_per_arm"] != 2000
            or summary["manifest_fingerprint"] != parent["fingerprint"] or summary["arms"] != parent["arms"]):
        raise ValueError("Require the original completed 2000-update APR trial")
    if args.cpu_threads != parent["cpu_threads"]:
        raise ValueError("Keep the parent's CPU thread setting")
    if Path(parent["checkpoint"]["path"]).resolve().parent in output.parents:
        raise ValueError("Output cannot be inside the original detector training directory")
    for identity in (parent["train_annotations"], parent["val_annotations"], *parent["assets"]):
        print("[identity] "+identity["path"], flush=True)
        if file_identity(identity["path"]) != identity:
            raise ValueError("Parent training/evaluation asset changed: "+identity["path"])
    sources, metrics = {}, {}
    # R is deliberately not opened, resumed or evaluated.
    for arm in ARMS:
        print(f"[verify parent {arm}] full state, all-rank pairing, rank checks and official evaluation", flush=True)
        stage = verify_parent_arm(parent, arm)
        evaluation = collect_evaluation(parent_dir/f"eval_{arm}")
        if stage != summary["training"][arm] or evaluation != summary["evaluations"][arm]:
            raise ValueError("Parent endpoint/receipts/evaluation differs from the completed summary")
        receipt = load_json(parent_dir/f"eval_{arm}"/"verified_result.json")
        if (receipt["checkpoint"] != stage["checkpoint"] or receipt["result"] != evaluation
                or receipt["manifest_fingerprint"] != parent["fingerprint"]):
            raise ValueError("Parent evaluation receipt differs")
        ckpt = load_trusted_torch_file(stage["checkpoint"]["path"])
        state = endpoint_state(ckpt, START+2000-1)
        resume = resume_identity(ckpt["trainer"], START+2000-1, parent["resume"]["lrs"])
        sources[arm] = {"checkpoint": stage["checkpoint"], "resume": resume,
                        "model_digest": state_digest(state),
                        "transcripts": [r["transcript"] for r in stage["receipts"]],
                        "final_rank": stage["final_rank"]}
        metrics[arm] = evaluation["metrics"]
        del ckpt, state
        gc.collect()
    code = dict(parent["code"])
    code.update({p: file_identity(ROOT/p)["sha256"] for p in CODE_FILES})
    m = {k: parent[k] for k in ("config", "prompt_bank", "assets", "train_annotations", "val_annotations",
                              "inventory", "seed", "num_gpus", "cpu_threads", "torch_version",
                              "lr_horizon", "rank_guard", "rank_period")}
    m.update(schema="apr_projection_extension_v1", output_dir=str(output), arms=ARMS,
             parent_manifest=file_identity(parent_dir/"manifest.json"),
             parent_summary=file_identity(parent_dir/"summary.json"), parent_fingerprint=parent["fingerprint"],
             completed_updates=2000, total_updates=args.total_updates, additional_updates=args.total_updates-2000,
             start=START+2000, sources=sources, metrics_at_2000=metrics, code=code, scope=[
                 "Only A/P: full APR in both; native projection ON in A, OFF in P. Never resume/evaluate R.",
                 "Each arm restores ITS OWN complete 2000-update model, AdamW moments, LR scheduler and AMP scaler.",
                 "Reconstruct first2000 DATA windows with recorded seeds/hashes; NO model forward/backward/update.",
                 "New updates continue the original cumulative seed/sampler timeline; no restart at the first image.",
                 "Same K5/no-radius/slot-prior/APR/RPSA/FedLoss/native micro GT normalization and clipping.",
                 "LR horizon remains85200; stopping at total4000 is not a new short LR schedule or warmup.",
                 "Full-bank all/rare geometry guarded initially/every50/endpoint; guard failure aborts, no retuning.",
                 "Evaluate each endpoint once with unchanged full LVIS protocol; preserve all2000-update evidence.",
                 "Single-seed continuation, not a full-history cause or guarantee of bitwise CUDA determinism.",
                 "No automatic adoption, R extension, further LR/regularizer changes or full12ep training.",
             ])
    m["fingerprint"] = fingerprint(m)
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    required = sum(s["checkpoint"]["bytes"] for s in sources.values())+4*1024**3
    if shutil.disk_usage(ancestor).free < required:
        raise ValueError(f"Need at least {required/1024**3:.1f} GiB free for NEW checkpoints/predictions/logs")
    output.mkdir(parents=True, exist_ok=False)
    for arm in ARMS:
        (output/arm).mkdir()
        with (output/arm/"last_checkpoint").open("x") as handle:
            handle.write(sources[arm]["checkpoint"]["path"])
    save_json(output/"manifest.json", m)
    print(f"[prepared] A/P each {m['additional_updates']} NEW updates; cumulative={m['total_updates']}; "
          f"iteration {m['start']}..{START+m['total_updates']-1}; LR horizon={HORIZON}", flush=True)
    return m


def verify_arm(m, arm):
    directory = Path(m["output_dir"])/arm
    completed, total, start = m["completed_updates"], m["total_updates"], m["start"]
    source = m["sources"][arm]
    initial = {k: source["resume"][k] for k in ("optimizer", "scheduler", "scaler")}
    initial["model"] = source["model_digest"]
    health = read_rows(directory/"rank_health.jsonl")
    expected = sorted({completed, total, *range(completed+RANK_PERIOD, total+1, RANK_PERIOD)})
    if ([r["update"] for r in health] != expected or not all(r["guard_pass"] for r in health)
            or health[0] != source["final_rank"]):
        raise ValueError("Incomplete/failed extension rank checks or changed starting geometry")
    receipts = []
    for rank in range(4):
        r = load_json(directory/f"complete_rank{rank}.json")
        if (not r.get("complete") or r["arm"] != arm or r["rank"] != rank or r["policy"] != ARMS[arm]
                or r["start"] != start or r["stop"] != START+total or r["updates"] != total-completed
                or r["total_updates"] != total or r["initial_state"] != initial
                or r["cursor_replay"] != cursor_receipt(m, arm, rank)
                or r["manifest_fingerprint"] != m["fingerprint"] or r["health_records"] != health):
            raise ValueError("Unverified extension resume/cursor/update receipt")
        for key, name in (("transcript", f"pairing_rank{rank}.jsonl"), ("update_log", f"updates_rank{rank}.jsonl")):
            if file_identity(directory/name) != r[key]:
                raise ValueError("Extension transcript changed")
        rows = read_rows(directory/f"pairing_rank{rank}.jsonl")
        updates = read_rows(directory/f"updates_rank{rank}.jsonl")
        if len(rows) != 2*(total-completed) or len(updates) != total-completed:
            raise ValueError("Wrong extension batch/update count")
        for i, row in enumerate(rows):
            plan = normalization_plan(row["normalization"]["global_gt_counts"], 4, 8)
            seed = m["seed"]+(completed+i//2)*128+rank*8+(i%2)*2
            if (row["iteration"] != start+i//2 or row["micro"] != i%2 or row["data_seed"] != seed
                    or row["forward_seed"] != seed+1 or row["normalization"] != plan or row["multiplier"] != 1.
                    or row["lrs"] != source["resume"]["lrs"] or row["policy"] != ARMS[arm]):
                raise ValueError("Extension timeline/seed/normalization/LR/policy differs")
            check_apr_record(row["losses"], row["apr_components"], ARMS[arm])
        for i, row in enumerate(updates):
            if (row["iteration"] != start+i or row["update"] != completed+i+1 or row["arm"] != arm
                    or row["policy"] != ARMS[arm] or row["normalization"] != rows[2*i]["normalization"]
                    or row["normalization"] != rows[2*i+1]["normalization"]
                    or any(not math.isfinite(v) for v in row["preclip_norms"].values())
                    or (arm == "P" and row["routing"] != {"enabled": False})):
                raise ValueError("Invalid native extension update record")
        if arm == "P":
            prior = read_rows(directory.parent/"A"/f"pairing_rank{rank}.jsonl")
            if len(prior) != len(rows):
                raise ValueError("A/P extension pairing lengths differ")
            for p, a in zip(rows, prior):
                verify_pair(p, a)
        receipts.append(r)
    path = directory/"model_final.pth"
    ckpt = load_trusted_torch_file(path)
    endpoint_state(ckpt, START+total-1)
    resume_identity(ckpt["trainer"], START+total-1, source["resume"]["lrs"])
    if (ckpt["trainer"].get("apr_projection_trial") != trial_tag(m, arm)
            or ckpt["trainer"].get("apr_projection_extension") != extension_tag(m, total, receipts[0]["cursor_replay"])):
        raise ValueError("Final checkpoint lacks verified extension provenance")
    del ckpt
    gc.collect()
    return {"checkpoint": file_identity(path), "receipts": receipts, "health": health, "final_rank": health[-1]}


def write_summary(m, stages, evaluations, *, complete=False):
    previous = m["metrics_at_2000"]
    delta_old = {k: previous["P"][k]-previous["A"][k] for k in METRICS}
    delta = ({k: evaluations["P"]["metrics"][k]-evaluations["A"]["metrics"][k] for k in METRICS}
             if set(evaluations) == set(ARMS) else None)
    changes = {arm: {k: value["metrics"][k]-previous[arm][k] for k in METRICS}
               for arm, value in evaluations.items()}
    result = {"complete": complete, "manifest_fingerprint": m["fingerprint"], "arms": ARMS,
              "total_updates_per_arm": m["total_updates"], "additional_updates_per_arm": m["additional_updates"],
              "training": stages, "evaluations": evaluations, "metrics_at_2000": previous,
              "delta_P_minus_A_at_2000": delta_old, "delta_P_minus_A": delta, "change_from_2000": changes,
              "scope": m["scope"], "decision": "NO_AUTOMATIC_ADOPTION_OR_FURTHER_TRAINING"}
    save_json(Path(m["output_dir"])/"summary.json", result)
    lines = ["Extended paired APR / conflict-projection trial (A/P only)",
             f"complete={complete}; total/arm={m['total_updates']}; NEW updates/arm={m['additional_updates']}",
             "A=full APR + projection; P=full APR, no projection; R NOT RUN",
             "stage  arm       AP       APr   rare-rank   guard"]
    for arm in ARMS:
        h = m["sources"][arm]["final_rank"]
        lines.append(f"2000   {arm}   {previous[arm]['AP']:9.4f} {previous[arm]['APr']:9.4f} "
                     f"{h['rare']['mean_rank']:11.4f}   {h['guard_pass']}")
    for arm, e in evaluations.items():
        h = stages[arm]["final_rank"]
        lines.append(f"{m['total_updates']:<6} {arm}   {e['metrics']['AP']:9.4f} {e['metrics']['APr']:9.4f} "
                     f"{h['rare']['mean_rank']:11.4f}   {h['guard_pass']}")
        lines.append(f"  {arm} change_from_2000: AP={changes[arm]['AP']:+.4f} APr={changes[arm]['APr']:+.4f}")
    lines.append(f"P-A at2000: AP={delta_old['AP']:+.4f} APr={delta_old['APr']:+.4f}")
    if delta is not None:
        lines.append(f"P-A at{m['total_updates']}: AP={delta['AP']:+.4f} APr={delta['APr']:+.4f}")
    lines.append("Single-seed continuation; no automatic adoption or further training.")
    Path(m["output_dir"], "results.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n"+"\n".join(lines), flush=True)
    return result


def execute(m):
    check_inputs(m)
    output = Path(m["output_dir"])
    stages, evaluations = {}, {}
    write_summary(m, stages, evaluations)
    for arm in ARMS:
        read_manifest(output/"manifest.json")
        directory = output/arm
        status(m, "RUNNING", f"train_{arm}")
        print(f"[extend {arm}] DATA cursor replay, then {m['additional_updates']} NEW optimizer updates", flush=True)
        if not (directory/"complete_rank0.json").exists():
            if {p.name for p in directory.iterdir()} != {"last_checkpoint"}:
                raise ValueError(f"Incomplete extension {arm}; no unsafe partial resume or overwrite")
            command = [sys.executable, "-u", str(ROOT/"tools/train_apr_projection_extension.py"),
                       "--manifest", str(output/"manifest.json"), "--arm", arm]
            run_evaluation(command, directory)
        stages[arm] = verify_arm(m, arm)
        write_summary(m, stages, evaluations)
        status(m, "RUNNING", f"evaluate_{arm}")
        print(f"[evaluate {arm}] cumulative update{m['total_updates']}, full LVIS", flush=True)
        directory = output/f"eval_{arm}"
        checkpoint = stages[arm]["checkpoint"]
        receipt = directory/"verified_result.json"
        if receipt.exists():
            prior = load_json(receipt)
            evaluations[arm] = collect_evaluation(directory)
            if (prior["checkpoint"] != checkpoint or prior["manifest_fingerprint"] != m["fingerprint"]
                    or prior["result"] != evaluations[arm]):
                raise ValueError("Stale extension evaluation receipt")
        else:
            if directory.exists():
                raise ValueError("Incomplete extension evaluation; no silent overwrite")
            directory.mkdir()
            command = evaluation_command(m["config"]["path"], checkpoint["path"], directory, 4)
            save_json(directory/"command.json", command)
            run_evaluation(command, directory)
            evaluations[arm] = collect_evaluation(directory)
            save_json(receipt, {"checkpoint": checkpoint, "manifest_fingerprint": m["fingerprint"], "result": evaluations[arm]})
        write_summary(m, stages, evaluations)
    check_inputs(m)
    result = write_summary(m, stages, evaluations, complete=True)
    status(m, "COMPLETE", "all_done", summary=str(output/"summary.json"))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--total-updates", type=int, default=4000)
    p.add_argument("--num-gpus", type=int, default=4)
    p.add_argument("--cpu-threads", type=int, default=2)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--execute-prepared", action="store_true")
    args = p.parse_args(argv)
    if args.cpu_threads < 1:
        raise ValueError("Positive CPU threads required")
    torch.set_num_threads(args.cpu_threads)
    m = None
    try:
        if args.execute_prepared:
            candidate = read_manifest(Path(args.output_dir)/"manifest.json")
            if (Path(args.output_dir).resolve() != Path(candidate["output_dir"])
                    or file_identity(Path(args.parent_dir)/"manifest.json") != candidate["parent_manifest"]
                    or any(getattr(args, k) != candidate[k] for k in ("total_updates", "num_gpus", "cpu_threads"))):
                raise ValueError("Prepared extension settings/parent differ")
            m = candidate
        else:
            m = prepare(args)
        if args.prepare_only:
            status(m, "PREPARED", "no_gpu_work")
            return m
        return execute(m)
    except (Exception, KeyboardInterrupt) as exc:
        if m is not None:
            path = Path(m["output_dir"])/"STATUS.json"
            prior = load_json(path) if path.exists() else {}
            status(m, "FAILED", prior.get("phase", "prepare"), error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
