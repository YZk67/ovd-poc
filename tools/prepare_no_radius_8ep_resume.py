"""CPU-only preflight and optional evidence snapshot for a trusted 4ep run.

This never changes a checkpoint, its last_checkpoint pointer, or starts training.
Only use trusted local training artifacts: the compatibility loader unpickles
full optimizer/trainer state. No detector, dataset or CUDA imports are required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lami_dino.checkpoint_init import load_trusted_torch_file  # noqa: E402


FOUR_EP_ITERATION = 28399
EIGHT_EP_STOP = 56800
LR_HORIZON = 85200
SNAPSHOT_NAME = "four_ep_snapshot"


def validate_checkpoint(checkpoint):
    """Require a real no-radius 4ep training checkpoint, not weights-only init."""
    if checkpoint.get("iteration") != FOUR_EP_ITERATION:
        raise ValueError("Expected the completed 4ep checkpoint at iteration 28399. "
                         "Do not run this preparation again on a later checkpoint.")
    trainer = checkpoint.get("trainer", {})
    for key, expected in (
        ("iteration", FOUR_EP_ITERATION),
        ("lr_scheduler_max_iter", LR_HORIZON),
        ("gradient_accumulation_steps", 2),
    ):
        if trainer.get(key) != expected:
            raise ValueError(f"trainer.{key}: expected {expected}, got {trainer.get(key)!r}")

    optimizer = trainer.get("optimizer", {})
    groups = optimizer.get("param_groups", [])
    if not optimizer.get("state") or not groups:
        raise ValueError("Full optimizer state is missing; weights-only loading is not resume.")
    scheduler = trainer.get("hooks", {}).get("LRScheduler", {})
    if scheduler.get("last_epoch") != FOUR_EP_ITERATION + 1:
        raise ValueError("LRScheduler is missing or its last_epoch is not 28400.")
    base_lrs = scheduler.get("base_lrs", [])
    if len(base_lrs) != len(groups):
        raise ValueError("Saved scheduler base_lrs do not match optimizer groups.")
    # This stage must remain on the original high-LR plateau, not a compressed
    # 4ep schedule followed by an LR jump on resume.
    for group, base_lr in zip(groups, base_lrs):
        if not math.isfinite(base_lr) or base_lr <= 0 or not math.isclose(
            group.get("lr", float("nan")), base_lr, rel_tol=1e-6, abs_tol=1e-12
        ):
            raise ValueError("Saved LR is not on the 12ep high-LR plateau; inspect the schedule.")

    state = checkpoint.get("model", {})
    queries = {key: value for key, value in state.items() if key.endswith(".tpa.prototype_queries")}
    if not queries:
        raise ValueError("No TPA prototype_queries found in the checkpoint.")
    for key, value in queries.items():
        prefix = key.removesuffix("prototype_queries")
        if tuple(value.shape) != (5, 256):
            raise ValueError(f"{key}: expected K=5, hidden_dim=256, got {tuple(value.shape)}")
        for name, expected in (("prototype_mode_strength", 0.0), ("slot_prior_strength", 0.2)):
            saved = state.get(prefix + name)
            if saved is None or saved.numel() != 1 or not math.isclose(
                float(saved.item()), expected, rel_tol=1e-6, abs_tol=1e-8
            ):
                raise ValueError(f"{prefix}{name}: expected {expected}; wrong experiment checkpoint.")
    return {"iteration": FOUR_EP_ITERATION, "resume_iteration": FOUR_EP_ITERATION + 1,
            "stop_iteration": EIGHT_EP_STOP, "lr_scheduler_max_iter": LR_HORIZON,
            "gradient_accumulation_steps": 2, "prototype_mode_strength": 0.0,
            "optimizer_lrs": sorted(set(float(group["lr"]) for group in groups))}


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def prepare(output_dir, *, snapshot=False):
    output_dir = Path(output_dir).resolve(strict=True)
    marker = output_dir / "last_checkpoint"
    if not marker.is_file():
        raise ValueError("Missing last_checkpoint: --resume could silently start from iteration 0.")
    target = marker.read_text().strip()
    checkpoint_path = (output_dir / target).resolve(strict=True)
    if not target or checkpoint_path.parent != output_dir or not checkpoint_path.is_file():
        raise ValueError("last_checkpoint must point to a checkpoint file directly in this run directory.")
    print(f"[load CPU] {checkpoint_path}", flush=True)
    checkpoint = load_trusted_torch_file(checkpoint_path)
    report = validate_checkpoint(checkpoint)
    del checkpoint
    report["checkpoint"] = checkpoint_path.name
    print(f"[OK] {json.dumps(report, sort_keys=True)}", flush=True)
    if not snapshot:
        print("[check only] No files changed. Use --snapshot before first continuation.", flush=True)
        return report

    # Independent copies: hard links are unsafe if training truncates model_final
    # or its prediction/config files. Require the complete 4ep evidence set.
    sources = [checkpoint_path, marker] + [output_dir / name for name in (
        "config.yaml", "log.txt", "metrics.json", "lvis_instances_results.json")]
    for path in sources:
        if not path.is_file():
            raise ValueError(f"Missing 4ep evidence: {path}; archive/restore it before continuation.")
    destination = output_dir / SNAPSHOT_NAME
    if destination.exists():
        manifest = json.loads((destination / "manifest.json").read_text())
        if manifest["resume"] != report or set(manifest["files"]) != {p.name for p in sources}:
            raise ValueError("Existing snapshot belongs to a different source; nothing overwritten.")
        for source in sources:
            expected = manifest["files"][source.name]
            if fingerprint(source) != expected or fingerprint(destination / source.name) != expected:
                raise ValueError(f"Snapshot/source mismatch for {source.name}; nothing overwritten.")
        print(f"[snapshot OK] Existing identical archive: {destination}", flush=True)
        return report

    required = sum(path.stat().st_size for path in sources)
    if shutil.disk_usage(output_dir).free < required + 16 * 1024 * 1024:
        raise ValueError(f"Snapshot needs {required / 1024**3:.2f} GiB plus safety margin; insufficient space.")
    staging = Path(tempfile.mkdtemp(prefix=".four_ep_snapshot-", dir=output_dir))
    print(f"[snapshot] Independent copies: {required / 1024**3:.2f} GiB", flush=True)
    files = {}
    try:
        for source in sources:
            print(f"[copy] {source.name}", flush=True)
            expected = fingerprint(source)
            shutil.copy2(source, staging / source.name)
            if fingerprint(staging / source.name) != expected or fingerprint(source) != expected:
                raise ValueError(f"{source.name} changed during snapshot; stop other writers first.")
            files[source.name] = expected
        (staging / "manifest.json").write_text(
            json.dumps({"resume": report, "files": files}, indent=2) + "\n")
        staging.rename(destination)
    except Exception:
        print(f"[incomplete snapshot retained] {staging}; no source files changed.", flush=True)
        raise
    print(f"[snapshot saved] {destination}", flush=True)
    print("[ready] Resume the SAME directory with the 8ep no-radius config and --resume.", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--snapshot", action="store_true", help="Copy/verify 4ep evidence before first resume")
    args = parser.parse_args()
    prepare(args.output_dir, snapshot=args.snapshot)
