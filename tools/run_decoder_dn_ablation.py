#!/usr/bin/env python3
"""Reuse verified native A; train/evaluate one DN-class->decoder blocked B.

The reference A is the immutable 500-update native arm from the completed
auxiliary-gradient experiment.  This halves the new GPU work while preserving
the exact paired data/RNG transcript and formal A endpoint.
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

from lami_dino.checkpoint_init import load_trusted_torch_file
from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.decoder_aux_ablation_ops import HORIZON, START, state_digest, verify_pair
from tools.diagnose_rare_fp_regions import fingerprint
from tools.evaluate_decoder_rollback import endpoint_state, evaluation_command, run_evaluation, verified_identity
from tools.run_decoder_aux_ablation import collect_evaluation, verify_arm as verify_reference_arm
from tools.summarize_eq2_counterfactual import METRICS
from tools.train_decoder_aux_arm import read_manifest as read_reference_manifest
from tools.train_decoder_dn_arm import read_manifest

UPDATES = 500
NEW_CODE = (
    "tools/decoder_dn_ablation_ops.py",
    "tools/train_decoder_dn_arm.py",
    "tools/run_decoder_dn_ablation.py",
)


def validate_reference(directory):
    directory = Path(directory).expanduser().resolve()
    manifest = read_reference_manifest(directory / "manifest.json")
    summary = load_json(directory / "summary.json")
    if (manifest["updates"] != UPDATES or manifest["start"] != START
            or manifest["lr_horizon"] != HORIZON or manifest["effective_batch"] != 32
            or not summary.get("complete") or summary.get("updates_per_arm") != UPDATES
            or summary.get("manifest_fingerprint") != manifest["fingerprint"]):
        raise ValueError("Need the completed formal 500-update auxiliary A/B reference trial")
    stage = verify_reference_arm(manifest, "A")
    evaluation = collect_evaluation(directory / "eval_A")
    if stage != summary["training"]["A"] or evaluation != summary["evaluations"]["A"]:
        raise ValueError("Reference A outputs disagree with its signed summary")
    return directory, manifest, summary, stage, evaluation


def prepare(args):
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise ValueError("Use a NEW output directory; do not pre-create or tee into it")
    if args.num_gpus != 4 or args.seed != 42 or args.cpu_threads < 1:
        raise ValueError("Locked trial: seed42, four GPUs, positive CPU threads")
    reference, prior, _prior_summary, stage_a, evaluation_a = validate_reference(
        args.reference_trial)
    source = prior["checkpoint"]
    verified_identity(source, "original full 8ep checkpoint")
    verified_identity(prior["config"], "locked training config")
    protected = [reference, Path(source["path"]).resolve().parent]
    if any(output == path or output in path.parents or path in output.parents for path in protected):
        raise ValueError("DN output must be separate from the reference and source training directories")
    checkpoint = load_trusted_torch_file(source["path"])
    model = endpoint_state(checkpoint, START-1)
    if state_digest(model) != prior["model_digest"]:
        raise ValueError("Reference manifest model digest no longer matches the 8ep checkpoint")
    del model, checkpoint
    gc.collect()
    transcripts = {
        str(receipt["rank"]): receipt["transcript"]
        for receipt in stage_a["receipts"]
    }
    if set(transcripts) != {"0", "1", "2", "3"}:
        raise ValueError("Reference A lacks four pairing transcripts")
    code = dict(prior["code"])
    code.update({relative: file_identity(ROOT / relative)["sha256"] for relative in NEW_CODE})
    manifest = {
        "output_dir": str(output),
        "intervention": "dn_classification_to_decoder_core",
        "checkpoint": source, "config": prior["config"], "resume": prior["resume"],
        "model_digest": prior["model_digest"], "decoder_keys": prior["decoder_keys"],
        "seed": 42, "updates": UPDATES, "num_gpus": 4,
        "cpu_threads": args.cpu_threads, "torch_version": prior["torch_version"],
        "start": START, "lr_horizon": HORIZON, "effective_batch": 32,
        "code": code,
        "reference_A": {
            "directory": str(reference),
            "manifest": file_identity(reference / "manifest.json"),
            "summary": file_identity(reference / "summary.json"),
            "checkpoint": stage_a["checkpoint"],
            "evaluation": evaluation_a,
            "transcripts": transcripts,
            "manifest_fingerprint": prior["fingerprint"],
        },
        "scope": [
            "A is the immutable verified native 500-update arm reused from the auxiliary trial.",
            "B removes only direct loss_class_dn and loss_class_dn_0..4 gradient to decoder_core.",
            "Final/auxiliary classification, encoder classification, L1/GIoU, APR and RPSA remain native.",
            "B replays A data, augmentation, FedLoss, forward RNG, LR and initial AMP scale exactly.",
            "Original optimizer moments, weight decay, AMP scaler and 85200-step LR horizon are retained.",
            "Full gradients set the native clipping coefficient before the DN component is subtracted.",
            "No second clipping is applied; inherited AdamW moments are not erased.",
            "One paired seed/window is causal evidence for this intervention, not all training stages.",
        ],
    }
    manifest["fingerprint"] = fingerprint(manifest)
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    required = Path(source["path"]).stat().st_size + 2 * 1024**3
    if shutil.disk_usage(parent).free < required:
        raise ValueError(f"Need at least {required / 1024**3:.1f} GiB free")
    output.mkdir(parents=True, exist_ok=False)
    (output / "B").mkdir()
    with (output / "B" / "last_checkpoint").open("x") as handle:
        handle.write(source["path"])
    save_json(output / "manifest.json", manifest)
    print(
        f"[prepared] reuse reference A={reference}; train DN-blocked B only; "
        f"updates={UPDATES}, stop={START+UPDATES}, LR horizon={HORIZON}", flush=True,
    )
    return manifest


def worker_command(manifest):
    return [
        sys.executable, "-u", str(ROOT / "tools/train_decoder_dn_arm.py"),
        "--manifest", str(Path(manifest["output_dir"]) / "manifest.json"),
    ]


def verify_b(manifest):
    directory = Path(manifest["output_dir"]) / "B"
    expected = {key: manifest["resume"][key] for key in ("optimizer", "scheduler", "scaler")}
    expected["model"] = manifest["model_digest"]
    receipts = []
    for rank in range(4):
        receipt = load_json(directory / f"complete_rank{rank}.json")
        if (not receipt.get("complete") or receipt.get("arm") != "B"
                or receipt.get("intervention") != manifest["intervention"]
                or receipt.get("rank") != rank or receipt.get("updates") != UPDATES
                or receipt.get("start") != START or receipt.get("stop") != START+UPDATES
                or receipt.get("manifest_fingerprint") != manifest["fingerprint"]
                or receipt.get("initial_state") != expected):
            raise ValueError("Incomplete or unmatched DN-B training receipt")
        verified_identity(receipt["transcript"], f"DN B rank{rank} pairing transcript")
        verified_identity(receipt["gradient_log"], f"DN B rank{rank} gradient log")
        with Path(receipt["gradient_log"]["path"]).open() as handle:
            updates = [json.loads(line) for line in handle]
        if len(updates) != UPDATES or any(
            row["iteration"] != START+index or row["update"] != index+1
            or row.get("remove") is not True or not 0 < row["clip_coefficient"] <= 1
            or any(not math.isfinite(row[key]) for key in (
                "full_detector_norm", "dn_norm", "core_post_norm"))
            for index, row in enumerate(updates)
        ):
            raise ValueError("Incomplete or nonfinite DN gradient intervention log")
        with Path(receipt["transcript"]["path"]).open() as handle:
            rows = [json.loads(line) for line in handle]
        with Path(manifest["reference_A"]["transcripts"][str(rank)]["path"]).open() as handle:
            reference = [json.loads(line) for line in handle]
        if len(rows) != 2*UPDATES or len(reference) != len(rows):
            raise ValueError("Incomplete DN B/reference-A microbatch transcript")
        for row, prior in zip(rows, reference):
            verify_pair(row, prior)
        receipts.append(receipt)
    checkpoint_path = directory / "model_final.pth"
    checkpoint = load_trusted_torch_file(checkpoint_path)
    endpoint_state(checkpoint, START+UPDATES-1)
    trainer = checkpoint["trainer"]
    certificate = trainer.get("decoder_dn_trial", {})
    expected_certificate = {
        "arm": "B", "start": START, "updates": UPDATES,
        "manifest_fingerprint": manifest["fingerprint"],
        "source": "loss_class_dn_and_0_through_4",
        "clipping": "full-gradient-reference",
    }
    if (trainer.get("iteration") != START+UPDATES-1
            or trainer.get("lr_scheduler_max_iter") != HORIZON
            or trainer.get("gradient_accumulation_steps") != 2
            or trainer.get("hooks", {}).get("LRScheduler", {}).get("last_epoch") != START+UPDATES
            or not trainer.get("optimizer", {}).get("state") or not trainer.get("grad_scaler")
            or certificate != expected_certificate):
        raise ValueError("Final B checkpoint does not certify the DN intervention")
    del checkpoint
    gc.collect()
    return {"checkpoint": file_identity(checkpoint_path), "receipts": receipts}


def verify_reference_inputs(manifest):
    verified_identity(manifest["checkpoint"], "original full 8ep checkpoint")
    verified_identity(manifest["config"], "locked training config")
    reference = manifest["reference_A"]
    for key in ("manifest", "summary", "checkpoint"):
        verified_identity(reference[key], f"reference A {key}")
    verified_identity(reference["evaluation"]["predictions"], "reference A predictions")
    for rank, identity in reference["transcripts"].items():
        verified_identity(identity, f"reference A rank{rank} transcript")


def execute(manifest):
    output = Path(manifest["output_dir"])
    verify_reference_inputs(manifest)
    directory = output / "B"
    if not (directory / "complete_rank0.json").exists():
        if {path.name for path in directory.iterdir()} != {"last_checkpoint"}:
            raise ValueError("Incomplete DN B run; retain it for diagnosis and use a fresh directory")
        print("[train B] 500 updates; remove only DN-classification gradient to decoder_core", flush=True)
        run_evaluation(worker_command(manifest), directory)
    stage_b = verify_b(manifest)
    eval_dir = output / "eval_B"
    receipt = eval_dir / "verified_result.json"
    if receipt.exists():
        prior = load_json(receipt)
        if (prior["checkpoint"] != stage_b["checkpoint"]
                or prior["manifest_fingerprint"] != manifest["fingerprint"]):
            raise ValueError("Stale DN B evaluation receipt")
        evaluation_b = collect_evaluation(eval_dir)
        if evaluation_b != prior["result"]:
            raise ValueError("Saved DN B evaluation outputs changed")
    else:
        if eval_dir.exists():
            raise ValueError("Incomplete DN B evaluation; inspect its log before rerunning")
        eval_dir.mkdir()
        command = evaluation_command(
            manifest["config"]["path"], stage_b["checkpoint"]["path"], eval_dir, 4)
        save_json(eval_dir / "command.json", command)
        print("[evaluate B] full LVIS, native boxes/ROI/top-300", flush=True)
        run_evaluation(command, eval_dir)
        evaluation_b = collect_evaluation(eval_dir)
        save_json(receipt, {
            "checkpoint": stage_b["checkpoint"],
            "manifest_fingerprint": manifest["fingerprint"], "result": evaluation_b,
        })
    verify_reference_inputs(manifest)
    evaluation_a = manifest["reference_A"]["evaluation"]
    result = {
        "complete": True, "intervention": manifest["intervention"],
        "manifest_fingerprint": manifest["fingerprint"], "updates_per_arm": UPDATES,
        "training": {
            "A": {"reused": True, "checkpoint": manifest["reference_A"]["checkpoint"],
                  "source_trial": manifest["reference_A"]["directory"]},
            "B": stage_b,
        },
        "evaluations": {"A": evaluation_a, "B": evaluation_b},
        "delta_B_minus_A": {
            key: evaluation_b["metrics"][key]-evaluation_a["metrics"][key] for key in METRICS
        },
        "scope": manifest["scope"],
    }
    save_json(output / "summary.json", result)
    print("\n=== Paired DN-classification decoder intervention ===")
    print("A (reused native)", evaluation_a["metrics"])
    print("B (DN blocked)", evaluation_b["metrics"])
    print("B - A", result["delta_B_minus_A"])
    print(f"[save] {output/'summary.json'}", flush=True)
    return result


def run(args):
    if args.execute_prepared:
        manifest = read_manifest(Path(args.output_dir) / "manifest.json")
        if Path(manifest["output_dir"]).resolve() != Path(args.output_dir).resolve():
            raise ValueError("Trial directory must not be moved")
        if (manifest["seed"], manifest["num_gpus"], manifest["cpu_threads"]) != (
                args.seed, args.num_gpus, args.cpu_threads):
            raise ValueError("Prepared trial options differ")
        if Path(manifest["reference_A"]["directory"]).resolve() != Path(
                args.reference_trial).expanduser().resolve():
            raise ValueError("Prepared reference A differs")
    else:
        manifest = prepare(args)
    if args.prepare_only:
        print("[prepare-only] No GPU inference or training was launched", flush=True)
        return manifest
    return execute(manifest)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-trial", required=True,
                        help="Completed no_radius_aux_decoder_ab_500 directory")
    parser.add_argument("--output-dir", required=True, help="NEW directory; do not pre-create")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--execute-prepared", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
