#!/usr/bin/env python3
"""Paired 8ep short continuation: which Eq. (2) training formula helps rare AP?

Three arms share the same full checkpoint, optimizer/scheduler/AMP state,
images, augmentation/FedLoss draws, effective batch and LR timeline.  TPA
geometry is frozen exactly.  Every arm is then evaluated on full LVIS using the
same calibrated inference equation.  This is a bounded screen, not a sweep and
not an automatic launch of a longer run.
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
from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.decoder_aux_ablation_ops import HORIZON, START, state_digest, validate_resume
from tools.diagnose_rare_fp_regions import fingerprint
from tools.evaluate_decoder_rollback import (
    endpoint_state,
    evaluation_command,
    run_evaluation,
    verified_identity,
)
from tools.summarize_eq2_counterfactual import METRICS, read_last_result
from tools.tpa_formula_screen_ops import ARMS, verify_formula_pair
from tools.tpa_geometry_audit_ops import extract_shared_tpa
from tools.train_tpa_formula_arm import read_manifest


CODE_FILES = (
    "lami_dino/prototype_ops.py",
    "lami_dino/modeling/text_classifier.py",
    "lami_dino/configs/models/dino_convnextl.py",
    "tools/train_net.py",
    "tools/decoder_aux_ablation_ops.py",
    "tools/decoder_loss_audit_ops.py",
    "tools/tpa_geometry_audit_ops.py",
    "tools/tpa_formula_screen_ops.py",
    "tools/train_tpa_formula_arm.py",
    "tools/run_tpa_formula_screen.py",
    "tools/evaluate_decoder_rollback.py",
)


def prepare(args):
    output = Path(args.output_dir).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    config_path = Path(args.config_file).expanduser().resolve()
    if output.exists():
        raise ValueError(
            "Use a NEW output directory (do not pre-create or tee into it). "
            "For an intact prepared/completed stage use --execute-prepared."
        )
    if args.seed != 42 or not 1 <= args.updates <= 500 or args.num_gpus != 4 or args.cpu_threads < 1:
        raise ValueError("Locked screen: seed42, four GPUs, 1..500 updates, positive CPU threads")
    if not checkpoint_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("Checkpoint/config does not exist")
    if (
        output == checkpoint_path.parent
        or output in checkpoint_path.parent.parents
        or checkpoint_path.parent in output.parents
    ):
        raise ValueError("Output must not overwrite or contain the source checkpoint directory")

    checkpoint = load_trusted_torch_file(checkpoint_path)
    state = endpoint_state(checkpoint, START - 1)
    resume = validate_resume(checkpoint)
    tpa, info = extract_shared_tpa(checkpoint)
    if info["iteration"] != START - 1:
        raise ValueError("Expected the completed 8ep endpoint")
    geometry = state_digest(tpa)
    code = {relative: file_identity(ROOT / relative)["sha256"] for relative in CODE_FILES}
    manifest = {
        "output_dir": str(output),
        "checkpoint": file_identity(checkpoint_path),
        "config": file_identity(config_path),
        "resume": resume,
        "model_digest": state_digest(state),
        "tpa_geometry_digest": geometry,
        "tpa_geometry": {
            "prototype_shape": list(tpa["prototype_queries"].shape),
            "slot_prior_strength": float(tpa["slot_prior_strength"]),
            "prototype_mode_strength": float(tpa["prototype_mode_strength"]),
        },
        "arms": ARMS,
        "seed": args.seed,
        "updates": args.updates,
        "num_gpus": 4,
        "cpu_threads": args.cpu_threads,
        "torch_version": str(torch.__version__),
        "code": code,
        "start": START,
        "lr_horizon": HORIZON,
        "effective_batch": 32,
        "scope": [
            "A=calibrated, B=calibrated+log(5), C=legacy training logits.",
            "TPA geometry parameters are frozen in all arms; APR/routing/gradient clipping remain computed natively.",
            "Detector/query parameters retain the full 8ep optimizer moments, scheduler and AMP scaler.",
            "All arms use paired new images/augmentation/FedLoss/RNG with workers=0; this is not a historical sampler replay.",
            "All formal LVIS evaluations use calibrated Eq. (2), current fusion and native boxes/ROI/top-300.",
            "One 500-update seed is a screening result; it does not establish the best full-training formula.",
        ],
    }
    manifest["fingerprint"] = fingerprint(manifest)
    del state, tpa, checkpoint
    gc.collect()

    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    # Three full checkpoints and prediction JSONs plus an explicit safety margin.
    required = 3 * checkpoint_path.stat().st_size + 5 * 1024**3
    if shutil.disk_usage(parent).free < required:
        raise ValueError(f"Need at least {required / 1024**3:.1f} GiB free for the three-arm screen")
    output.mkdir(parents=True, exist_ok=False)
    for arm in ARMS:
        directory = output / arm
        directory.mkdir()
        with (directory / "last_checkpoint").open("x") as handle:
            handle.write(str(checkpoint_path))
    save_json(output / "manifest.json", manifest)
    print(
        f"[prepared] full 8ep resume at {START}; {args.updates} updates/arm; "
        f"stop={START + args.updates}; LR horizon={HORIZON}; effective batch32; TPA geometry frozen",
        flush=True,
    )
    return manifest


def worker_command(manifest, arm):
    return [
        sys.executable,
        "-u",
        str(ROOT / "tools/train_tpa_formula_arm.py"),
        "--manifest",
        str(Path(manifest["output_dir"]) / "manifest.json"),
        "--arm",
        arm,
    ]


def verify_arm(manifest, arm):
    directory = Path(manifest["output_dir"]) / arm
    expected_initial = {key: manifest["resume"][key] for key in ("optimizer", "scheduler", "scaler")}
    expected_initial["model"] = manifest["model_digest"]
    receipts = []
    for rank in range(4):
        receipt = load_json(directory / f"complete_rank{rank}.json")
        if (
            not receipt.get("complete")
            or receipt["arm"] != arm
            or receipt["aggregation"] != ARMS[arm]
            or receipt["rank"] != rank
            or receipt["updates"] != manifest["updates"]
            or receipt["start"] != START
            or receipt["stop"] != START + manifest["updates"]
            or receipt["manifest_fingerprint"] != manifest["fingerprint"]
            or receipt["initial_state"] != expected_initial
            or receipt["tpa_geometry_digest"] != manifest["tpa_geometry_digest"]
        ):
            raise ValueError("Incomplete/unmatched formula-training receipt")
        transcript = directory / f"pairing_rank{rank}.jsonl"
        update_log = directory / f"updates_rank{rank}.jsonl"
        verified_identity(receipt["transcript"], f"{arm} rank{rank} pairing transcript")
        verified_identity(receipt["update_log"], f"{arm} rank{rank} update log")
        with transcript.open() as handle:
            rows = [json.loads(line) for line in handle]
        if len(rows) != 2 * manifest["updates"]:
            raise ValueError("Incomplete paired microbatch count")
        for index, row in enumerate(rows):
            if (
                row["iteration"] != START + index // 2
                or row["micro"] != index % 2
                or row["lrs"] != manifest["resume"]["lrs"]
            ):
                raise ValueError("Pairing transcript iteration/microbatch/LR mismatch")
        if arm != "A":
            with (directory.parent / "A" / transcript.name).open() as handle:
                reference = [json.loads(line) for line in handle]
            if len(reference) != len(rows):
                raise ValueError("Formula transcript lengths differ")
            for row, prior in zip(rows, reference):
                verify_formula_pair(row, prior)
        with update_log.open() as handle:
            updates = [json.loads(line) for line in handle]
        if len(updates) != manifest["updates"] or any(
            row != {
                "iteration": START + index,
                "update": index + 1,
                "aggregation": ARMS[arm],
                "tpa_geometry_digest": manifest["tpa_geometry_digest"],
            }
            for index, row in enumerate(updates)
        ):
            raise ValueError("TPA freeze/update log is incomplete or inconsistent")
        receipts.append(receipt)

    path = directory / "model_final.pth"
    saved = load_trusted_torch_file(path)
    endpoint_state(saved, START + manifest["updates"] - 1)
    final_tpa, _ = extract_shared_tpa(saved)
    if state_digest(final_tpa) != manifest["tpa_geometry_digest"]:
        raise ValueError("Final checkpoint TPA geometry changed despite freeze")
    trainer = saved["trainer"]
    trial = trainer.get("tpa_formula_trial", {})
    expected_trial = {
        "arm": arm,
        "aggregation": ARMS[arm],
        "start": START,
        "updates": manifest["updates"],
        "manifest_fingerprint": manifest["fingerprint"],
        "tpa_geometry_frozen": True,
    }
    if (
        trainer.get("iteration") != START + manifest["updates"] - 1
        or trainer.get("lr_scheduler_max_iter") != HORIZON
        or trainer.get("gradient_accumulation_steps") != 2
        or trainer.get("hooks", {}).get("LRScheduler", {}).get("last_epoch")
        != START + manifest["updates"]
        or not trainer.get("optimizer", {}).get("state")
        or not trainer.get("grad_scaler")
        or trial != expected_trial
    ):
        raise ValueError("Final checkpoint does not certify the completed frozen-TPA trial")
    del saved, final_tpa
    gc.collect()
    return {
        "checkpoint": file_identity(path),
        "receipts": receipts,
        "tpa_geometry_unchanged": True,
    }


def collect_evaluation(directory):
    console = (directory / "console.log").read_text(errors="replace")
    if "Skipping evaluation" in console or "Traceback (most recent call last)" in console:
        raise RuntimeError("Native evaluation failed/skipped; no successful summary")
    metrics = dict(zip(METRICS, read_last_result(directory / "log.txt")))
    if len(metrics) != 9 or any(not math.isfinite(value) or not 0 <= value <= 100 for value in metrics.values()):
        raise ValueError("Missing/invalid official LVIS metrics")
    predictions = directory / "lvis_instances_results.json"
    if not predictions.is_file() or predictions.stat().st_size <= 2:
        raise ValueError("Missing nonempty full-validation predictions")
    return {"metrics": metrics, "predictions": file_identity(predictions)}


def execute(manifest):
    output = Path(manifest["output_dir"])
    verified_identity(manifest["checkpoint"], "original full 8ep checkpoint")
    stages = {}
    for arm in ARMS:
        directory = output / arm
        if not (directory / "complete_rank0.json").exists():
            if {path.name for path in directory.iterdir()} != {"last_checkpoint"}:
                raise ValueError(f"Incomplete {arm} run; retain it and use a fresh trial directory")
            print(
                f"[train {arm}] {ARMS[arm]}, {manifest['updates']} optimizer updates, TPA geometry frozen",
                flush=True,
            )
            run_evaluation(worker_command(manifest, arm), directory)
        stages[arm] = verify_arm(manifest, arm)

    evaluations = {}
    for arm in ARMS:
        directory = output / f"eval_{arm}"
        checkpoint = stages[arm]["checkpoint"]
        receipt_path = directory / "verified_result.json"
        if receipt_path.exists():
            prior = load_json(receipt_path)
            if prior["checkpoint"] != checkpoint or prior["manifest_fingerprint"] != manifest["fingerprint"]:
                raise ValueError("Stale evaluation receipt")
            evaluations[arm] = collect_evaluation(directory)
            if evaluations[arm] != prior["result"]:
                raise ValueError("Saved evaluation outputs changed")
            continue
        if directory.exists():
            raise ValueError(f"Incomplete evaluation {arm}; inspect it before explicitly rerunning")
        directory.mkdir()
        command = evaluation_command(manifest["config"]["path"], checkpoint["path"], directory, 4)
        save_json(directory / "command.json", command)
        print(f"[evaluate {arm}] full LVIS, common calibrated inference", flush=True)
        run_evaluation(command, directory)
        evaluations[arm] = collect_evaluation(directory)
        save_json(
            receipt_path,
            {
                "checkpoint": checkpoint,
                "manifest_fingerprint": manifest["fingerprint"],
                "result": evaluations[arm],
            },
        )

    verified_identity(manifest["checkpoint"], "original checkpoint still unchanged")
    deltas = {
        arm: {
            metric: evaluations[arm]["metrics"][metric] - evaluations["A"]["metrics"][metric]
            for metric in METRICS
        }
        for arm in ARMS
        if arm != "A"
    }
    ranking = sorted(
        (
            {
                "arm": arm,
                "aggregation": ARMS[arm],
                "APr": evaluations[arm]["metrics"]["APr"],
                "AP": evaluations[arm]["metrics"]["AP"],
            }
            for arm in ARMS
        ),
        key=lambda row: (row["APr"], row["AP"]),
        reverse=True,
    )
    result = {
        "complete": True,
        "manifest_fingerprint": manifest["fingerprint"],
        "updates_per_arm": manifest["updates"],
        "training": stages,
        "evaluations": evaluations,
        "delta_vs_calibrated_A": deltas,
        "ranking_by_APr_then_AP": ranking,
        "all_tpa_geometry_unchanged": all(
            stage["tpa_geometry_unchanged"] for stage in stages.values()
        ),
        "scope": manifest["scope"],
    }
    save_json(output / "summary.json", result)
    print("\n=== Frozen-TPA training-formula screen ===")
    for arm in ARMS:
        print(arm, ARMS[arm], evaluations[arm]["metrics"])
    for arm, delta in deltas.items():
        print(f"{arm} - A", delta)
    print("ranking", ranking)
    print("TPA geometry unchanged in every arm: True")
    print(f"[save] {output / 'summary.json'}", flush=True)
    return result


def run(args):
    if args.cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    if args.execute_prepared:
        manifest = read_manifest(Path(args.output_dir) / "manifest.json")
        if Path(manifest["output_dir"]).resolve() != Path(args.output_dir).resolve():
            raise ValueError("Trial directory must not be moved")
        if (
            manifest["updates"],
            manifest["seed"],
            manifest["num_gpus"],
            manifest["cpu_threads"],
        ) != (args.updates, args.seed, args.num_gpus, args.cpu_threads):
            raise ValueError("Prepared trial options differ")
        if file_identity(args.checkpoint) != manifest["checkpoint"]:
            raise ValueError("Source checkpoint changed")
        if file_identity(args.config_file) != manifest["config"]:
            raise ValueError("Config changed")
    else:
        manifest = prepare(args)
    if args.prepare_only:
        print("[prepare-only] No GPU training or inference was launched", flush=True)
        return manifest
    return execute(manifest)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Full 8ep model_0056799.pth")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--output-dir", required=True, help="NEW directory; do not pre-create")
    parser.add_argument("--updates", type=int, default=500, help="1..500; smaller is smoke-only")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--execute-prepared", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
