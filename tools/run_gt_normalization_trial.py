#!/usr/bin/env python3
"""Bounded native-micro vs pooled-GT continuation, then two full LVIS evaluations.

Same complete no-radius 8ep checkpoint; 500 updates/arm; TPA stays trainable.
Native per-micro FedLoss, APR/RPSA, AdamW, LR and inference remain unchanged.
No automatic long run, sweep, or deletion of prior results.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from lami_dino.checkpoint_init import load_trusted_torch_file
from tools.audit_accumulation_updates import check_sources, validate_reference
from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.decoder_aux_ablation_ops import START, HORIZON, state_digest, validate_resume
from tools.diagnose_rare_fp_regions import fingerprint
from tools.evaluate_decoder_rollback import endpoint_state, evaluation_command, run_evaluation
from tools.gt_normalization_trial_ops import ARMS, RANK_GUARD, RANK_PERIOD, normalization_plan, verify_pair
from tools.run_tpa_formula_screen import collect_evaluation
from tools.summarize_eq2_counterfactual import METRICS
from tools.train_gt_normalization_arm import read_manifest

CODE_FILES = (
    "tools/gt_normalization_trial_ops.py", "tools/train_gt_normalization_arm.py",
    "tools/run_gt_normalization_trial.py", "tools/train_tpa_formula_arm.py",
    "tools/tpa_formula_screen_ops.py", "tools/run_tpa_formula_screen.py",
    "tools/diagnose_rare_fp_regions.py", "tools/summarize_eq2_counterfactual.py",
)
SCOPE = [
    "A native per-micro GT denominator; B pools GT across the two physical microbatches.",
    "Only detection class/L1/GIoU final/aux/encoder/DN loss multipliers change; APR/RPSA do not.",
    "Both arms restore the same complete no-radius 8ep model, optimizer, scheduler and scaler.",
    "TPA stays trainable; unchanged slot prior/APR/conflict routing/separate clipping. No rank-maximizing intervention.",
    "Each arm computes its OWN native clipping coefficients after routing, as in the update audit.",
    "New paired training stream (not historical replay): images/augmentations/FedLoss/forward RNG/LR/AMP verified.",
    "Matching/features naturally diverge after updates; CUDA custom kernels need not be bitwise deterministic.",
    "Physical batch16 x accumulation2; not an exact physical eight-GPU batch32 reproduction.",
    "Full-bank all/rare rank checks initially, every 50 updates and at the endpoint. Guard is not AP evidence.",
    "One short seed is a screening result, not proof of old-model provenance or 8-to-12ep causality.",
]


def validate_update_report(report):
    if (not report.get("complete") or report.get("live_optimizer_steps") != 0
            or not all(report.get(k) is True for k in ("model_state_unchanged", "live_optimizer_state_unchanged",
                                                       "source_checkpoint_unchanged"))
            or report.get("decision") != "CONDITIONAL_ONE_STEP_DIFFERENCES_NOT_APR_ATTRIBUTION"):
        raise ValueError("Require the completed optimizer-aware zero-live-update audit")
    signature = hashlib.sha256(json.dumps(report["inputs"], sort_keys=True).encode()).hexdigest()
    if signature != report["fingerprint"] or report["protocol"]["iteration"] != START:
        raise ValueError("Update audit fingerprint/endpoint mismatch")
    if not report["windows"] or any(not w.get("paired_native_forward_verified") for w in report["windows"]):
        raise ValueError("Missing paired-forward audit verification")


def prepare(args):
    if args.num_gpus != 4 or args.seed != 42 or not 1 <= args.updates <= 500 or args.cpu_threads < 1:
        raise ValueError("Locked trial: four GPUs, seed42, 1..500 updates, positive CPU threads")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise ValueError("Use a NEW output directory; do not pre-create or tee inside it. "
                         "Use --execute-prepared only for intact completed stages.")
    report = load_json(args.update_audit)
    validate_update_report(report)
    parent_id = report["inputs"]["objective_audit"]
    if file_identity(parent_id["path"]) != parent_id:
        raise ValueError("Parent objective report changed")
    parent = load_json(parent_id["path"])
    validate_reference(parent)
    if (parent["fingerprint"] != report["inputs"]["reference_fingerprint"]
            or report["checkpoint"] != parent["inputs"]["checkpoint"]
            or report["config"] != parent["inputs"]["config"]):
        raise ValueError("Audit/checkpoint/config identities disagree")
    if str(torch.__version__) != report["runtime"]["torch"]:
        raise ValueError("Use the same lami/PyTorch environment as the audit")
    protected = [Path(report["checkpoint"]["path"]).resolve().parent,
                 Path(args.update_audit).resolve().parent, Path(parent_id["path"]).resolve().parent]
    if any(output == p or output in p.parents or p in output.parents for p in protected):
        raise ValueError("Output must be separate from source checkpoint/audit directories")
    check_sources(parent)
    for relative, digest in report["inputs"]["code"].items():
        target = (ROOT / relative).resolve()
        if ROOT not in target.parents or file_identity(target)["sha256"] != digest:
            raise ValueError("Optimizer-aware audit code changed: " + relative)
    checkpoint = load_trusted_torch_file(report["checkpoint"]["path"])
    state = endpoint_state(checkpoint, START-1)
    resume = validate_resume(checkpoint)
    if resume != report["resume_identity"]:
        raise ValueError("Restored optimizer/scheduler/scaler differs from update audit")
    prompt = [x for x in parent["assets"] if Path(x["path"]).name == "lvis_claude_prompts_convnextl.npy"]
    if len(prompt) != 1:
        raise ValueError("Expected one audited full prompt bank")
    code = dict(parent["inputs"]["code"])
    code.update(report["inputs"]["code"])
    for name in CODE_FILES:
        code[name] = file_identity(ROOT / name)["sha256"]
    m = {"schema": "gt_normalization_ab_v1", "output_dir": str(output),
         "update_audit": file_identity(args.update_audit), "objective_audit": parent_id,
         "checkpoint": report["checkpoint"], "config": report["config"], "prompt_bank": prompt[0],
         "assets": parent["assets"], "train_annotations": parent["train_annotations"],
         "val_annotations": file_identity(Path(parent["train_annotations"]["path"]).parent / "lvis_v1_val.json"),
         "resume": resume, "model_digest": state_digest(state), "inventory": report["inventory"],
         "seed": args.seed, "updates": args.updates, "num_gpus": 4, "cpu_threads": args.cpu_threads,
         "torch_version": str(torch.__version__), "code": code, "arms": ARMS,
         "start": START, "lr_horizon": HORIZON, "rank_guard": RANK_GUARD, "rank_period": RANK_PERIOD,
         "scope": SCOPE}
    m["fingerprint"] = fingerprint(m)
    del state, checkpoint
    gc.collect()
    parent_dir = output.parent
    while not parent_dir.exists():
        parent_dir = parent_dir.parent
    required = 2*Path(m["checkpoint"]["path"]).stat().st_size + 4*1024**3
    if shutil.disk_usage(parent_dir).free < required:
        raise ValueError(f"Need {required/1024**3:.1f} GiB free for checkpoints/predictions/logs")
    output.mkdir(parents=True, exist_ok=False)
    for arm in ARMS:
        directory = output / arm
        directory.mkdir()
        with (directory / "last_checkpoint").open("x") as handle:
            handle.write(m["checkpoint"]["path"])
    save_json(output / "manifest.json", m)
    print(f"[prepared] {args.updates} updates/arm; start={START}; stop={START+args.updates}; "
          f"LR horizon={HORIZON}; TPA TRAINABLE; two full LVIS evaluations", flush=True)
    return m


def read_rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle]


def verify_arm(m, arm):
    directory = Path(m["output_dir"]) / arm
    expected_initial = {k: m["resume"][k] for k in ("optimizer", "scheduler", "scaler")}
    expected_initial["model"] = m["model_digest"]
    health = read_rows(directory / "rank_health.jsonl")
    expected_checks = sorted({0, m["updates"], *range(RANK_PERIOD, m["updates"]+1, RANK_PERIOD)})
    if [r["update"] for r in health] != expected_checks or not all(r["guard_pass"] for r in health):
        raise ValueError("Incomplete/failed prototype health guard")
    receipts = []
    for rank in range(4):
        receipt = load_json(directory / f"complete_rank{rank}.json")
        if (not receipt.get("complete") or receipt["arm"] != arm or receipt["rank"] != rank
                or receipt["start"] != START or receipt["stop"] != START+m["updates"]
                or receipt["updates"] != m["updates"] or receipt["initial_state"] != expected_initial
                or receipt["manifest_fingerprint"] != m["fingerprint"] or receipt["health_records"] != health):
            raise ValueError("Invalid training receipt")
        for field, filename in (("transcript", f"pairing_rank{rank}.jsonl"), ("update_log", f"updates_rank{rank}.jsonl")):
            if file_identity(directory / filename) != receipt[field]:
                raise ValueError("Training transcript changed")
        rows = read_rows(directory / f"pairing_rank{rank}.jsonl")
        updates = read_rows(directory / f"updates_rank{rank}.jsonl")
        if len(rows) != 2*m["updates"] or len(updates) != m["updates"]:
            raise ValueError("Wrong microbatch/optimizer update count")
        for i, row in enumerate(rows):
            plan = normalization_plan(row["normalization"]["global_gt_counts"], 4, 8)
            coefficient = plan["detection_loss_multipliers"][i%2] if arm == "B" else 1.
            if (row["iteration"] != START+i//2 or row["micro"] != i%2
                    or row["lrs"] != m["resume"]["lrs"] or row["normalization"] != plan
                    or row["multiplier"] != coefficient):
                raise ValueError("Incorrect paired window/multiplier/LR")
        for i, row in enumerate(updates):
            if (row["iteration"] != START+i or row["update"] != i+1 or row["arm"] != arm
                    or row["normalization"] != rows[2*i]["normalization"]
                    or row["normalization"] != rows[2*i+1]["normalization"]
                    or any(not math.isfinite(v) for v in row["preclip_norms"].values())):
                raise ValueError("Invalid native update/routing/clipping record")
        if arm == "B":
            reference = read_rows(directory.parent / "A" / f"pairing_rank{rank}.jsonl")
            if len(reference) != len(rows):
                raise ValueError("Pair transcript lengths differ")
            for row, prior in zip(rows, reference):
                verify_pair(row, prior)
        receipts.append(receipt)
    path = directory / "model_final.pth"
    ckpt = load_trusted_torch_file(path)
    endpoint_state(ckpt, START+m["updates"]-1)
    trainer = ckpt["trainer"]
    expected = {"arm": arm, "normalization": ARMS[arm], "start": START, "updates": m["updates"],
                "manifest_fingerprint": m["fingerprint"]}
    if (trainer.get("gt_normalization_trial") != expected
            or trainer.get("iteration") != START+m["updates"]-1
            or trainer.get("lr_scheduler_max_iter") != HORIZON
            or trainer.get("gradient_accumulation_steps") != 2
            or trainer.get("hooks", {}).get("LRScheduler", {}).get("last_epoch") != START+m["updates"]
            or not trainer.get("optimizer", {}).get("state") or not trainer.get("grad_scaler")):
        raise ValueError("Final checkpoint does not certify full native paired continuation")
    del ckpt
    gc.collect()
    return {"checkpoint": file_identity(path), "receipts": receipts, "health": health,
            "final_rank": health[-1]}


def execute(m):
    # Fail before any training if assets/annotations/checkpoint changed since preparation.
    for identity in (m["checkpoint"], m["update_audit"], m["objective_audit"], m["train_annotations"],
                     m["val_annotations"], *m["assets"]):
        if file_identity(identity["path"]) != identity:
            raise ValueError("Prepared input changed: " + identity["path"])
    output = Path(m["output_dir"])
    stages, evaluations = {}, {}
    for arm in ARMS:
        directory = output / arm
        if not (directory / "complete_rank0.json").exists():
            if {p.name for p in directory.iterdir()} != {"last_checkpoint"}:
                raise ValueError(f"Incomplete {arm}; retain evidence and use a fresh output directory")
            command = [sys.executable, "-u", str(ROOT / "tools/train_gt_normalization_arm.py"),
                       "--manifest", str(output / "manifest.json"), "--arm", arm]
            print(f"[train {arm}] {ARMS[arm]}: {m['updates']} optimizer updates", flush=True)
            run_evaluation(command, directory)
        stages[arm] = verify_arm(m, arm)
    for arm in ARMS:
        directory = output / f"eval_{arm}"
        checkpoint = stages[arm]["checkpoint"]
        receipt_path = directory / "verified_result.json"
        if receipt_path.exists():
            prior = load_json(receipt_path)
            evaluations[arm] = collect_evaluation(directory)
            if (prior["checkpoint"] != checkpoint or prior["manifest_fingerprint"] != m["fingerprint"]
                    or prior["result"] != evaluations[arm]):
                raise ValueError("Stale evaluation receipt")
            continue
        if directory.exists():
            raise ValueError(f"Incomplete evaluation {arm}; inspect it, no silent overwrite")
        directory.mkdir()
        command = evaluation_command(m["config"]["path"], checkpoint["path"], directory, 4)
        save_json(directory / "command.json", command)
        print(f"[evaluate {arm}] full LVIS, unchanged calibrated inference", flush=True)
        run_evaluation(command, directory)
        evaluations[arm] = collect_evaluation(directory)
        save_json(receipt_path, {"checkpoint": checkpoint, "manifest_fingerprint": m["fingerprint"],
                                 "result": evaluations[arm]})
    for identity in (m["checkpoint"], m["val_annotations"]):
        if file_identity(identity["path"]) != identity:
            raise ValueError("Source checkpoint/validation annotations changed during trial")
    summary = {"complete": True, "manifest_fingerprint": m["fingerprint"], "updates_per_arm": m["updates"],
               "training": stages, "evaluations": evaluations, "scope": m["scope"],
               "delta_B_minus_A": {k: evaluations["B"]["metrics"][k]-evaluations["A"]["metrics"][k]
                                   for k in METRICS},
               "decision": "SHORT_PAIRED_SCREEN_NOT_LONG_TERM_OR_HISTORICAL_ATTRIBUTION"}
    save_json(output / "summary.json", summary)
    print("\n=== Paired GT normalization trial ===")
    for arm in ARMS:
        print(arm, evaluations[arm]["metrics"], "rank", stages[arm]["final_rank"])
    print("B - A", summary["delta_B_minus_A"])
    print(f"[save] {output / 'summary.json'}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--execute-prepared", action="store_true")
    args = parser.parse_args(argv)
    if args.cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    if args.execute_prepared:
        m = read_manifest(Path(args.output_dir) / "manifest.json")
        if (Path(args.output_dir).resolve() != Path(m["output_dir"])
                or file_identity(args.update_audit) != m["update_audit"]
                or any(getattr(args, k) != m[k] for k in ("updates", "num_gpus", "seed", "cpu_threads"))):
            raise ValueError("Prepared trial settings/input differ")
    else:
        m = prepare(args)
    if args.prepare_only:
        print("[prepare-only] No training/evaluation launched", flush=True)
        return m
    return execute(m)


if __name__ == "__main__":
    main()
