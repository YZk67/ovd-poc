#!/usr/bin/env python3
"""Capture a bounded single-GPU loss probe, then audit its TPA directions on CPU.

Never trains/updates a model. --reuse-gradients skips all image/GPU work.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from lami_dino.checkpoint_init import load_trusted_torch_file
from tools.capture_tpa_gradients import capture
from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.diagnose_rare_fp_regions import fingerprint, read_manifest, validate_sample
from tools.tpa_geometry_audit_ops import extract_shared_tpa, reconstruct_tpa, validate_reconstructed_bank
from tools.tpa_gradient_audit_ops import (
    CENTERS, LOGITS, SCALARS, direction_jvp, gradient_directions, loss_gradients,
    parameter_block_norms, split_jvp_summary,
)


def save_gradients(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = stream.name
            torch.save(value, stream)
        os.replace(temporary, path)
    finally:
        if temporary and Path(temporary).exists():
            Path(temporary).unlink()


def load_queries(geometry, state, prompts):
    categories = [{k: c[k] for k in ("id", "name", "frequency")} for c in geometry["classes"]]
    indices = {c["name"]: i for i, c in enumerate(categories)}
    reconstruction = reconstruct_tpa(prompts, state, geometry["protocol"]["tpa_tau"])
    bank, cache = None, {}
    features, query_indices, records, seen = [], [], [], set()
    for row in geometry["regions"]:
        if len(row["queries"]) != 1:
            continue
        entry = row["queries"][0]
        identity = (row["category"], row["kind"], row["image_id"], entry["query_id"])
        if identity in seen:
            continue
        seen.add(identity)
        directory = Path(row["cache_directory"])
        sample_key = str(directory), row["image_id"]
        if sample_key not in cache:
            manifest = read_manifest(directory / "manifest.json")
            if (manifest["fingerprint"] != geometry["parent_fingerprint"]
                    and manifest["inputs"].get("parent_fingerprint") != geometry["parent_fingerprint"]):
                raise ValueError("Query cache is unrelated to geometry report")
            candidate = load_trusted_torch_file(directory / "new" / "bank.pt")
            if candidate["category_ids"] != [c["id"] for c in categories]:
                raise ValueError("Cached category order differs")
            validate_reconstructed_bank(reconstruction, candidate)
            if bank is None:
                bank = candidate
            elif any(candidate[k] != bank[k] for k in ("temperature", "logit_scale", "cls_bias", "tpa_tau")):
                raise ValueError("Cached classifier protocol differs")
            sample = load_trusted_torch_file(directory / "new" / f"{row['image_id']}.pt")
            validate_sample(sample, candidate, manifest["fingerprint"], "new", row["image_id"])
            cache[sample_key] = sample
        sample = cache[sample_key]
        q = entry["query_id"]
        if not 0 <= q < len(sample["features"]):
            raise ValueError("Invalid cached query ID")
        if not torch.allclose(sample["query_boxes"][q].double(), torch.tensor(entry["box_xyxy"], dtype=torch.float64), atol=.05, rtol=0):
            raise ValueError("Cached query box differs from geometry audit")
        features.append(sample["features"][q].cpu())
        query_indices.append(indices[row["category"]])
        records.append({"category": row["category"], "kind": row["kind"], "image_id": row["image_id"],
                        "query_id": q, "native_cache_logit": entry["native_cache_logit"]})
    if not records:
        raise ValueError("No unambiguous selected native queries")
    return categories, bank, torch.stack(features), torch.tensor(query_indices, dtype=torch.long), records


def capture_identity(args, geometry):
    paths = (set(ROOT.glob("lami_dino/**/*.py")) | set(ROOT.glob("configs/common/**/*.py"))
             | set(ROOT.glob("detrex/modeling/**/*.py")) | set(ROOT.glob("detrex/data/**/*.py"))
             | {ROOT / "tools/train_net.py", ROOT / "tools/capture_tpa_gradients.py"})
    return {"schema_version": 1, "checkpoint_sha256": geometry["checkpoint"]["sha256"],
            "geometry_sha256": file_identity(args.geometry_json)["sha256"],
            "config_sha256": file_identity(args.config_file)["sha256"],
            "training_code_sha256": {str(p.relative_to(ROOT)): file_identity(p)["sha256"] for p in sorted(paths)},
            "loss_grouping_sha256": hashlib.sha256(inspect.getsource(loss_gradients).encode()).hexdigest(),
            "windows": args.windows, "microbatches": args.microbatches, "batch_size": args.batch_size,
            "seed": args.seed, "device": args.device}


def analyze_capture(captured, state, prompts, bank, categories, features, query_classes, records):
    reports = []
    for window in captured["windows"]:
        directions, stats = gradient_directions(window["gradients"], captured["clip_max_norm"])
        report = {"window": window["window"], "mean_losses": window["mean_losses"],
                  "microbatches": window["microbatches"], "gradient_summary": stats, "jvps": {}}
        print(f"\n=== Window {window['window'] + 1}: averaged gradients before projection ===", flush=True)
        print(stats["routing"], flush=True)
        for name, gradient in directions.items():
            summary = stats["directions"][name]
            summary["parameter_block_norms"] = parameter_block_norms(gradient, state)
            jvp = direction_jvp(state, prompts, bank, features, query_classes, gradient)
            error = max(abs(row[3] - record["native_cache_logit"]) for row, record in zip(jvp["query_logits"], records))
            if error > 2e-3:
                raise ValueError(f"JVP forward differs from cached native logits: {error}")
            jvp["native_logit_max_abs_error"] = error
            jvp["by_split"], jvp["selected_queries"] = split_jvp_summary(jvp, categories, records)
            # A common clip coefficient preserves additive gradient attribution.
            # This is still a Euclidean gradient probe, NOT AdamW's update.
            factor = summary["norm"] * stats["directions"]["routed_total"]["clip_coefficient_if_used_alone"]
            jvp["raw_to_common_clipped_factor"] = factor
            for row in jvp["selected_queries"]:
                row["common_clipped_gradient_derivative_mean"] = {k: v * factor for k, v in row["derivative_mean"].items()}
            report["jvps"][name] = jvp
            rare = jvp["by_split"].get("r", {})
            scalars = rare.get("scalar_derivative_mean", {})
            angular = rare.get("center_angular_speed_mean_deg", {})
            print(f"[direction] {name} norm={summary['norm']:.6g} "
                  f"rare d_shift={scalars.get('center_shift_ratio')} "
                  f"d_spread={scalars.get('pre_spread_ratio')} "
                  f"d_unit_mean={scalars.get('post_unit_mean_norm')} "
                  f"d_cos={scalars.get('post_pairwise_cos')} "
                  f"value-center angular-speed={angular.get('value_center')}", flush=True)
        reports.append(report)
    return reports


def display_queries(windows):
    for window in windows:
        print(f"\n=== Window {window['window'] + 1}: selected-query LOGIT derivatives per unit descent ===", flush=True)
        print("direction         class             kind     N    center   mean-corr   LME-uplift     logit  common-clip logit", flush=True)
        for name in ("task", "apr", "unprojected_total", "routed_total", "projection_added"):
            for row in window["jvps"][name]["selected_queries"]:
                d = row["derivative_mean"]
                print(f"{name:17} {row['category']:17} {row['kind']:5} {row['queries']:4} "
                      f"{d['center_response']:9.5f} {d['center_to_mean_correction']:11.5f} "
                      f"{d['dispersion_uplift']:12.5f} {d['native_logit']:9.5f} "
                      f"{row['common_clipped_gradient_derivative_mean']['native_logit']:18.5f}", flush=True)


def run(args):
    if min(args.windows, args.microbatches, args.batch_size, args.cpu_threads) < 1:
        raise ValueError("Probe sizes and threads must be positive")
    if args.windows * args.microbatches * args.batch_size > 128:
        raise ValueError("Hard audit budget is 128 training-image exposures; no unbounded runs")
    torch.set_num_threads(args.cpu_threads)
    geometry = load_json(args.geometry_json)
    if geometry.get("complete") is not True or geometry.get("side") != "new":
        raise ValueError("Use the completed NEW-checkpoint geometry audit")
    checkpoint_path = Path(args.checkpoint or geometry["checkpoint"]["path"]).resolve()
    prompt_path = Path(geometry["prompt_bank"]["path"]).resolve()
    output = Path(args.output).resolve()
    gradient_path = Path(args.gradient_cache).resolve() if args.gradient_cache else output.with_suffix(".gradients.pt")
    protected = {Path(args.geometry_json).resolve(), checkpoint_path, prompt_path,
                 Path(args.config_file).resolve(), Path(geometry["source_report"]["path"]).resolve(),
                 Path(geometry["annotations"]["path"]).resolve()}
    cache_dirs = {Path(r["cache_directory"]).resolve() for r in geometry["regions"]}
    if output == gradient_path or any(p in protected or any(p == d or d in p.parents for d in cache_dirs)
                                      for p in (output, gradient_path)):
        raise ValueError("Audit outputs must not overwrite source inputs/caches")
    print("[identity] checking checkpoint/prompt SHA256 against the geometry audit", flush=True)
    for path, expected in ((checkpoint_path, geometry["checkpoint"]), (prompt_path, geometry["prompt_bank"])):
        if file_identity(path)["sha256"] != expected["sha256"]:
            raise ValueError(f"Input identity differs from geometry audit: {path}")
    checkpoint = load_trusted_torch_file(checkpoint_path)
    state, state_info = extract_shared_tpa(checkpoint)
    del checkpoint
    if state_info["iteration"] != geometry["tpa_state"]["iteration"]:
        raise ValueError("Checkpoint iteration mismatch")
    prompts = torch.from_numpy(np.load(prompt_path, allow_pickle=False)).float()
    print("[preflight] checking all selected query caches before any GPU work", flush=True)
    categories, bank, features, query_classes, records = load_queries(geometry, state, prompts)
    inputs = capture_identity(args, geometry)
    signature = fingerprint(inputs)
    if args.reuse_gradients:
        cached = load_trusted_torch_file(gradient_path)
        if cached.get("fingerprint") != signature or cached.get("inputs") != inputs:
            raise ValueError("Gradient cache provenance differs; do not mix captures")
        captured = cached["capture"]
        print(f"[reuse] {gradient_path}; no images/GPU", flush=True)
    else:
        if gradient_path.exists():
            raise ValueError("Gradient cache already exists; use --reuse-gradients or a new output path")
        captured = capture(args, geometry, state)
        save_gradients(gradient_path, {"inputs": inputs, "fingerprint": signature, "capture": captured})
        print(f"[saved gradients] {gradient_path}; CPU analysis can be rerun with --reuse-gradients", flush=True)
        gc.collect()
    print("[JVP] CPU text-side directional derivatives; parameters are never updated", flush=True)
    windows = analyze_capture(captured, state, prompts, bank, categories, features, query_classes, records)
    report = {"complete": True, "schema_version": 1, "inputs": inputs, "gradient_cache": file_identity(gradient_path),
              "capture_metadata": {k: v for k, v in captured.items() if k != "windows"},
              "classes": categories, "selected_query_records": records, "windows": windows,
              "definitions": {"scalar_columns": list(SCALARS), "center_columns": list(CENTERS), "logit_columns": list(LOGITS),
                              "jvp": "J[-g/||g||], a local derivative per unit parameter-space descent, NOT a finite optimizer update",
                              "magnitude": "Multiply unit derivative by the direction norm for raw -g; for hypothetical clipped SGD also multiply by clip coefficient and LR. NOT AdamW.",
                              "angular_speed": "Unsigned angular speed, not movement toward correct semantics",
                              "zero_centers": "Undefined directions flagged; excluded from mean angular speeds",
                              "scope": "Local TPA-only derivatives at one final checkpoint, not historical training causation or AP prediction",
                              "queries": "Validation labels only label frozen diagnostic queries, never training-loss targets. Query fusion/features/boxes are held fixed in score JVP.",
                              "comparisons": "Unit-normalized direction rows do not add linearly; recover raw magnitudes before comparing component sums."}}
    save_json(output, report)
    display_queries(windows)
    print("\nScope: local gradient audit, not AdamW replay or training ablation. No weights changed, no AP estimate.", flush=True)
    print(f"[save] {output}", flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-json", required=True)
    parser.add_argument("--config-file", default="lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py")
    parser.add_argument("--checkpoint", help="Relocated identical checkpoint (SHA256 checked)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gradient-cache")
    parser.add_argument("--reuse-gradients", action="store_true")
    parser.add_argument("--windows", type=int, default=2)
    parser.add_argument("--microbatches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
