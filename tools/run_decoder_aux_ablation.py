#!/usr/bin/env python3
"""One paired 8ep continuation: native A vs aux-class->decoder gradient-blocked B.

Each arm performs 500 optimizer updates, then full native LVIS inference. No
loss-weight sweep, checkpoint swapping, validation-GT training or LR restart.
Trusted local reports/checkpoints only; fail closed on incomplete resume/pairing.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from lami_dino.checkpoint_init import load_trusted_torch_file
from tools.audit_decoder_classification_sources import validate_parent
from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.decoder_aux_ablation_ops import ARMS, HORIZON, START, state_digest, validate_resume, verify_pair
from tools.diagnose_rare_fp_regions import fingerprint
from tools.evaluate_decoder_rollback import endpoint_state, evaluation_command, run_evaluation, verified_identity
from tools.summarize_eq2_counterfactual import METRICS, read_last_result
from tools.train_decoder_aux_arm import read_manifest

NEW_CODE = ("tools/decoder_aux_ablation_ops.py", "tools/train_decoder_aux_arm.py",
            "tools/run_decoder_aux_ablation.py")


def validate_split(path):
    report = load_json(path)
    if (not report.get("complete") or not report.get("parent_replay", {}).get("verified")
            or report.get("training_updates") != 0 or report.get("optimizer_created") is not False):
        raise ValueError("Need the completed, replay-verified 8ep classification split report")
    inputs = report["inputs"]
    if fingerprint(inputs) != report["fingerprint"] or inputs.get("iteration") != START-1:
        raise ValueError("Classification audit fingerprint/endpoint mismatch")
    for relative, digest in inputs["code"].items():
        target = (ROOT / relative).resolve()
        if ROOT not in target.parents or file_identity(target)["sha256"] != digest:
            raise ValueError(f"Code changed since classification audit: {relative}")
    verified_identity(report["capture"], "split capture")
    verified_identity(inputs["parent_audit"], "parent loss-source audit")
    parent, query, _ = validate_parent(inputs["parent_audit"]["path"])
    if inputs["sources"] != parent["inputs"]["sources"]:
        raise ValueError("Split and parent source identities differ")
    return report, query


def prepare(args):
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise ValueError("Use a NEW output directory (do not pre-create or tee into it). "
                         "For an intact prepared/completed stage use --execute-prepared.")
    if args.seed != 42 or not 1 <= args.updates <= 500 or args.num_gpus != 4 or args.cpu_threads < 1:
        raise ValueError("Locked trial: seed42, four GPUs, 1..500 updates, positive CPU threads")
    report, query = validate_split(args.classification_audit)
    sources = report["inputs"]["sources"]
    source = sources["old_checkpoint"]
    protected = [Path(v["path"]).resolve() for v in sources.values()]
    protected += [Path(args.classification_audit).resolve().parent, Path(source["path"]).resolve().parent]
    if any(output == p or output in p.parents or p in output.parents for p in protected):
        raise ValueError("Trial output must be separate from original training/audit directories")
    checkpoint = load_trusted_torch_file(source["path"])
    state = endpoint_state(checkpoint, START-1)
    resume = validate_resume(checkpoint)
    code = dict(report["inputs"]["code"])
    code.update({relative: file_identity(ROOT / relative)["sha256"] for relative in NEW_CODE})
    m = {"output_dir": str(output), "classification_audit": file_identity(args.classification_audit),
         "checkpoint": source, "config": sources["config_file"], "resume": resume,
         "model_digest": state_digest(state), "decoder_keys": sorted(query["inventory"]["decoder_core"]["keys"]),
         "seed": args.seed, "updates": args.updates, "num_gpus": 4, "cpu_threads": args.cpu_threads,
         "torch_version": str(torch.__version__), "code": code,
         "start": START, "lr_horizon": HORIZON, "effective_batch": 32,
         "scope": [
             "Only direct auxiliary class_0..4 gradient to decoder_core is subtracted in B.",
             "Original optimizer moments, weight decay, AMP scaler, APR routing, LR horizon retained.",
             "Both arms clip FULL gradients first; B subtracts the equally clipped auxiliary component.",
             "No second clipping: B core gradient can exceed the original clipping bound after subtraction.",
             "Both arms use new paired inputs/RNG, workers=0; not exact historical sampler/RNG replay.",
             "CUDA custom kernels may be nondeterministic; input/RNG pairing is checked, not bitwise final weights.",
             "Full LVIS outcome of one seed/short window is evidence for this intervention, not a global causal proof.",
         ]}
    m["fingerprint"] = fingerprint(m)
    del state, checkpoint
    gc.collect()
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    # Two full optimizer checkpoints, two prediction JSONs, plus safety margin.
    required = 2 * Path(source["path"]).stat().st_size + 3 * 1024**3
    if shutil.disk_usage(parent).free < required:
        raise ValueError(f"Need at least {required / 1024**3:.1f} GiB free for this paired trial")
    output.mkdir(parents=True, exist_ok=False)
    for arm in ARMS:
        directory = output / arm
        directory.mkdir()
        with (directory / "last_checkpoint").open("x") as handle:
            handle.write(source["path"])
    save_json(output / "manifest.json", m)
    print(f"[prepared] full 8ep resume at {START}; {m['updates']} updates/arm; "
          f"stop={START+m['updates']}; LR horizon={HORIZON}; effective batch32", flush=True)
    return m


def worker_command(m, arm):
    return [sys.executable, "-u", str(ROOT / "tools/train_decoder_aux_arm.py"),
            "--manifest", str(Path(m["output_dir"]) / "manifest.json"), "--arm", arm]


def verify_arm(m, arm):
    directory = Path(m["output_dir"]) / arm
    expected_state = {k: m["resume"][k] for k in ("optimizer", "scheduler", "scaler")}
    expected_state["model"] = m["model_digest"]
    receipts = []
    for rank in range(4):
        r = load_json(directory / f"complete_rank{rank}.json")
        if (not r.get("complete") or r["arm"] != arm or r["rank"] != rank
                or r["updates"] != m["updates"] or r["start"] != START or r["stop"] != START+m["updates"]
                or r["manifest_fingerprint"] != m["fingerprint"] or r["initial_state"] != expected_state):
            raise ValueError("Incomplete/unmatched native training receipt")
        transcript = directory / f"pairing_rank{rank}.jsonl"
        if Path(r["transcript"]["path"]).resolve() != transcript.resolve():
            raise ValueError("Unexpected transcript path")
        verified_identity(r["transcript"], f"{arm} rank{rank} pairing transcript")
        gradient_path = directory / f"updates_rank{rank}.jsonl"
        if Path(r["gradient_log"]["path"]).resolve() != gradient_path.resolve():
            raise ValueError("Unexpected gradient log path")
        verified_identity(r["gradient_log"], f"{arm} rank{rank} gradient log")
        with gradient_path.open() as handle:
            updates = [json.loads(line) for line in handle]
        if len(updates) != m["updates"] or any(
                u["iteration"] != START+i or u["update"] != i+1 or u["remove"] != (arm == "B")
                or not 0 < u["clip_coefficient"] <= 1
                or any(not math.isfinite(u[k]) for k in ("full_detector_norm", "aux_norm", "core_post_norm"))
                for i, u in enumerate(updates)):
            raise ValueError("Incomplete/nonfinite gradient intervention log")
        with transcript.open() as handle:
            rows = [json.loads(line) for line in handle]
        if len(rows) != 2*m["updates"]:
            raise ValueError("Incomplete paired microbatch count")
        for i, row in enumerate(rows):
            if (row["iteration"] != START+i//2 or row["micro"] != i%2
                    or row["lrs"] != m["resume"]["lrs"]):
                raise ValueError("Pairing transcript iteration/microbatch/LR mismatch")
        if arm == "B":
            with (directory.parent / "A" / transcript.name).open() as handle:
                reference = [json.loads(line) for line in handle]
            if len(reference) != len(rows):
                raise ValueError("A/B transcript lengths differ")
            for row, prior in zip(rows, reference):
                verify_pair(row, prior)
        receipts.append(r)
    path = directory / "model_final.pth"
    saved = load_trusted_torch_file(path)
    endpoint_state(saved, START+m["updates"]-1)
    trainer = saved["trainer"]
    trial = trainer.get("decoder_aux_trial", {})
    if (trainer.get("iteration") != START+m["updates"]-1
            or trainer.get("lr_scheduler_max_iter") != HORIZON
            or trainer.get("gradient_accumulation_steps") != 2
            or trainer.get("hooks", {}).get("LRScheduler", {}).get("last_epoch") != START+m["updates"]
            or not trainer.get("optimizer", {}).get("state") or not trainer.get("grad_scaler")
            or trial != {"arm": arm, "start": START, "updates": m["updates"],
                         "manifest_fingerprint": m["fingerprint"], "clipping": "full-gradient-reference"}):
        raise ValueError("Final checkpoint does not certify the completed paired update budget")
    del saved
    gc.collect()
    return {"checkpoint": file_identity(path), "receipts": receipts}


def collect_evaluation(directory):
    console = (directory / "console.log").read_text(errors="replace")
    if "Skipping evaluation" in console or "Traceback (most recent call last)" in console:
        raise RuntimeError("Native evaluation failed/skipped; no successful summary")
    metrics = dict(zip(METRICS, read_last_result(directory / "log.txt")))
    if len(metrics) != 9 or any(not math.isfinite(v) or not 0 <= v <= 100 for v in metrics.values()):
        raise ValueError("Missing/invalid official LVIS metrics")
    predictions = directory / "lvis_instances_results.json"
    if not predictions.is_file() or predictions.stat().st_size <= 2:
        raise ValueError("Missing nonempty full-validation predictions")
    return {"metrics": metrics, "predictions": file_identity(predictions)}


def execute(m):
    output = Path(m["output_dir"])
    verified_identity(m["checkpoint"], "original full 8ep checkpoint")
    stages = {}
    for arm in ARMS:
        directory = output / arm
        if not (directory / "complete_rank0.json").exists():
            # Never silently continue half a trial with lost RNG/transcript state.
            if {p.name for p in directory.iterdir()} != {"last_checkpoint"}:
                raise ValueError(f"Incomplete {arm} run; retain for diagnosis and use a fresh trial directory")
            print(f"[train {arm}] {m['updates']} optimizer updates, original full resume", flush=True)
            run_evaluation(worker_command(m, arm), directory)
        stages[arm] = verify_arm(m, arm)
    evaluations = {}
    for arm in ARMS:
        directory = output / f"eval_{arm}"
        checkpoint = stages[arm]["checkpoint"]
        receipt = directory / "verified_result.json"
        if receipt.exists():
            prior = load_json(receipt)
            if prior["checkpoint"] != checkpoint or prior["manifest_fingerprint"] != m["fingerprint"]:
                raise ValueError("Stale evaluation receipt")
            evaluations[arm] = collect_evaluation(directory)
            if evaluations[arm] != prior["result"]:
                raise ValueError("Saved evaluation outputs changed")
            continue
        if directory.exists():
            raise ValueError(f"Incomplete evaluation {arm}; inspect its log before explicitly rerunning it")
        directory.mkdir()
        command = evaluation_command(m["config"]["path"], checkpoint["path"], directory, 4)
        save_json(directory / "command.json", command)
        print(f"[evaluate {arm}] full LVIS, native boxes/ROI/top-300", flush=True)
        run_evaluation(command, directory)
        evaluations[arm] = collect_evaluation(directory)
        save_json(receipt, {"checkpoint": checkpoint, "manifest_fingerprint": m["fingerprint"],
                            "result": evaluations[arm]})
    verified_identity(m["checkpoint"], "original checkpoint still unchanged")
    result = {"complete": True, "manifest_fingerprint": m["fingerprint"], "updates_per_arm": m["updates"],
              "training": stages, "evaluations": evaluations,
              "delta_B_minus_A": {k: evaluations["B"]["metrics"][k]-evaluations["A"]["metrics"][k]
                                  for k in METRICS}, "scope": m["scope"]}
    save_json(output / "summary.json", result)
    print("\n=== Paired auxiliary-classification decoder intervention ===")
    for arm in ARMS:
        print(arm, evaluations[arm]["metrics"])
    print("B - A", result["delta_B_minus_A"])
    print(f"[save] {output / 'summary.json'}", flush=True)
    return result


def run(args):
    if args.cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    if args.execute_prepared:
        m = read_manifest(Path(args.output_dir) / "manifest.json")
        if Path(m["output_dir"]).resolve() != Path(args.output_dir).resolve():
            raise ValueError("Trial directory must not be moved")
        if (m["updates"], m["seed"], m["num_gpus"], m["cpu_threads"]) != (args.updates, args.seed, args.num_gpus, args.cpu_threads):
            raise ValueError("Prepared trial options differ")
        if file_identity(args.classification_audit) != m["classification_audit"]:
            raise ValueError("Classification audit changed")
    else:
        m = prepare(args)
    if args.prepare_only:
        print("[prepare-only] No GPU inference or training was launched", flush=True)
        return m
    return execute(m)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--classification-audit", required=True)
    p.add_argument("--output-dir", required=True, help="NEW output directory; do not pre-create")
    p.add_argument("--updates", type=int, default=500, help="1..500; smaller budgets are smoke tests only")
    p.add_argument("--num-gpus", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--execute-prepared", action="store_true", help="Reuse verified completed stages, never partial training")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
