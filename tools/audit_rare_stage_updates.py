#!/usr/bin/env python3
"""Bounded 8ep/10ep ranking audit: native regions, terminal-bank swaps, local TPA gradients.

Uses one GPU only for selected-image forwards and <=128 train_norare image
exposures across BOTH endpoints. Never creates an optimizer or updates weights.
Endpoint gradients are local evidence, NOT a replay of historical AdamW updates.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from lami_dino.checkpoint_init import load_trusted_torch_file
from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.rare_stage_update_ops import (
    endpoint_effects, flat_delta, grouped_margins, local_margin_audit, probe_pairing, select_ranking_classes,
)
from tools.tpa_geometry_audit_ops import extract_shared_tpa, validate_reconstructed_bank, reconstruct_tpa


def preflight(args):
    if not (1 <= args.max_images <= 32 and min(args.windows, args.microbatches, args.batch_size, args.cpu_threads) > 0):
        raise ValueError("Positive probe sizes required; validation image budget is 1..32")
    if 2 * args.windows * args.microbatches * args.batch_size > 128:
        raise ValueError("Hard budget: <=128 training-image exposures across BOTH checkpoints")
    source_paths = [Path(getattr(args, key)).resolve() for key in (
        "comparison", "old_checkpoint", "new_checkpoint", "old_predictions", "new_predictions",
        "annotations", "prompt_bank", "config_file")]
    output = Path(args.output_dir).resolve()
    if any(not p.is_file() for p in source_paths):
        raise FileNotFoundError(f"Missing inputs: {[str(p) for p in source_paths if not p.is_file()]}")
    if any(p == output or output in p.parents for p in source_paths):
        raise ValueError("Use a separate output directory, not an input directory")
    comparison = load_json(args.comparison)
    for key, expected in (("old_apr", 42.8843), ("new_apr", 42.3031)):
        actual = comparison.get(key)
        if actual is None or not np.isfinite(actual) or abs(actual - expected) > .02:
            raise ValueError(f"Expected no-radius 8ep/10ep {key}={expected}, found {actual}")
    names = select_ranking_classes(comparison, args.classes)
    identities = {}
    for key in ("comparison", "old_checkpoint", "new_checkpoint", "old_predictions", "new_predictions",
                "annotations", "prompt_bank", "config_file"):
        print(f"[identity] hashing {key}: {getattr(args, key)}", flush=True)
        identities[key] = file_identity(getattr(args, key))
    states = {}
    for side, iteration in (("old", 56799), ("new", 70999)):
        print(f"[preflight] {side} checkpoint: iteration/buffers/shared TPA aliases", flush=True)
        checkpoint = load_trusted_torch_file(getattr(args, side + "_checkpoint"))
        state, info = extract_shared_tpa(checkpoint)
        if info["iteration"] != iteration:
            raise ValueError(f"{side}: expected iteration {iteration}, found {info['iteration']}")
        if float(state["prototype_mode_strength"]) != 0. or abs(float(state["slot_prior_strength"]) - .2) > 1e-6:
            raise ValueError("Both endpoints must be the no-radius, slot-prior=.2 run")
        if state["prototype_queries"].shape != (5, 256):
            raise ValueError("Expected five 256D prototype queries")
        states[side] = state
        del checkpoint
    return names, identities, states


def prepare(args, names, identities, output):
    from tools.diagnose_detector_tpa_pairing import locked_manifest
    from tools.inspect_rare_pre_tp_fps import run as inspect_fps
    from tools.rare_region_pairing_ops import collect_regions

    identity = {"schema_version": 1, "sources": identities, "categories": names,
                "max_images": args.max_images, "seed": args.seed,
                "windows": args.windows, "microbatches": args.microbatches, "batch_size": args.batch_size}
    # Lock every source before any result can be reused/overwritten.
    locked_manifest(output / "manifest.json", identity)
    path = output / "fp_details.json"
    if not path.exists():
        inspect_fps(SimpleNamespace(comparison=args.comparison, old_predictions=args.old_predictions,
            new_predictions=args.new_predictions, annotations=args.annotations, categories=names,
            max_dets=300, print_fp_classes=names, print_fp_limit=5, output=str(path)))
    details = load_json(path)
    for key in ("comparison", "old_predictions", "new_predictions", "annotations"):
        if details["sources"][key]["sha256"] != identities[key]["sha256"]:
            raise ValueError("Saved FP details refer to changed inputs")
    regions = collect_regions(details, names)
    image_ids = sorted({row["image_id"] for row in regions})
    if not 1 <= len(image_ids) <= args.max_images:
        raise ValueError(f"Selected {len(image_ids)} images; budget {args.max_images}. "
                         "No GPU work started. Use fewer --classes and a fresh output directory.")
    print(f"[budget] categories={names}, validation images={len(image_ids)} x 2 checkpoints; "
          f"training image exposures={2 * args.windows * args.microbatches * args.batch_size}; no optimizer", flush=True)
    return details, image_ids


def dump_and_pair(args, names, identities, image_ids, output):
    from tools.diagnose_detector_tpa_pairing import dump_checkpoint, input_asset_hashes, locked_manifest
    from tools.diagnose_rare_fp_regions import model_code_hash, run as pair_regions

    cache = output / "pairing_cache"
    dump_args = SimpleNamespace(**vars(args), phase="dump", alpha=0., beta=.3, novel_scale=3.,
                                tpa_tau=.004375, cls_tau=.07, max_dets=300, query_chunk_size=128,
                                log_interval=5)
    inputs = {"schema_version": 1, "annotations_sha256": identities["annotations"]["sha256"],
              "asset_sha256": input_asset_hashes(dump_args, cache), "code_sha256": model_code_hash(args.config_file),
              "image_ids": image_ids, "alpha": 0., "beta": .3, "novel_scale": 3.,
              "tpa_tau": .004375, "cls_tau": .07, "max_dets": 300, "iou_thresholds": [.5, .75]}
    for side in ("old", "new"):
        inputs[side + "_checkpoint"] = identities[side + "_checkpoint"]["path"]
        inputs[side + "_sha256"] = identities[side + "_checkpoint"]["sha256"]
    signature = locked_manifest(cache / "manifest.json", inputs)
    dataset = load_json(args.annotations)
    for side in ("old", "new"):
        dump_checkpoint(dump_args, side, {"image_ids": image_ids}, dataset, signature, cache)
    return pair_regions(SimpleNamespace(
        fp_details=str(output / "fp_details.json"), pairing_cache=str(cache),
        output=str(output / "regions.json"), categories=names, match_iou=.5,
        max_images=args.max_images, cpu_threads=args.cpu_threads, device="cpu", fill_missing=False,
        dump_device=args.device, annotations=args.annotations, old_checkpoint=args.old_checkpoint,
        new_checkpoint=args.new_checkpoint, config_file=args.config_file))


def load_panel(regions, cache, states, prompts):
    from tools.diagnose_rare_fp_regions import read_manifest, validate_sample
    from tools.rare_logit_decomposition_ops import decompose_entry

    manifest = read_manifest(cache / "manifest.json")
    banks = {side: load_trusted_torch_file(cache / side / "bank.pt") for side in ("old", "new")}
    for side in banks:
        validate_reconstructed_bank(reconstruct_tpa(prompts, states[side], banks[side]["tpa_tau"]), banks[side])
    by_id = {c: i for i, c in enumerate(banks["new"]["category_ids"])}
    dataset = load_json(regions["annotations"]["path"])
    by_name = {c["name"]: by_id[c["id"]] for c in dataset["categories"]}
    features, records, clips = ({s: [] for s in ("old", "new")} for _ in range(3))
    indices, samples, seen, excluded = [], {}, set(), []
    for row in regions["regions"]:
        if row["source_side"] != "new":
            continue
        if len(row["source_candidates"]) != 1 or row["other_best_score_box"] is None:
            excluded.append({"category": row["category"], "image_id": row["image_id"],
                             "kind": row["kind"], "reason": "ambiguous_source_or_no_old_IoU_ge_0.5_counterpart"})
            continue
        entries = {"new": row["source_candidates"][0], "old": row["other_best_score_box"]}
        identity = row["category"], row["image_id"], entries["new"]["query_id"]
        if identity in seen:
            continue
        seen.add(identity)
        index = by_name[row["category"]]
        indices.append(index)
        for side, entry in entries.items():
            key = side, row["image_id"]
            if key not in samples:
                samples[key] = load_trusted_torch_file(cache / side / f"{row['image_id']}.pt")
                validate_sample(samples[key], banks[side], manifest["fingerprint"], *key)
            sample = samples[key]
            checked = decompose_entry(entry, sample, banks[side], index, regions["protocol"])
            features[side].append(sample["features"][entry["query_id"]])
            clips[side].append(entry["clip_log_probability"])
            records[side].append({"category": row["category"], "kind": row["kind"],
                                  "image_id": row["image_id"], "query_id": entry["query_id"],
                                  "native_cache_logit": checked["logit_terms"]["native_logit"],
                                  "region_iou": entry["region_iou"], "box_xyxy": entry["box_xyxy"]})
    if not indices:
        raise ValueError("No unambiguous paired regions; cannot audit margins")
    counts = grouped_margins(torch.zeros(len(indices)), records["new"])
    if not any(c["pairs"] for c in counts.values()):
        raise ValueError("No class has paired TP AND FP regions; no training-gradient probes started")
    print(f"[paired panel] counts={counts}; excluded regions={len(excluded)}", flush=True)
    return (banks, {s: torch.stack(v) for s, v in features.items()}, torch.tensor(indices), records,
            {s: torch.tensor(v, dtype=torch.float64) for s, v in clips.items()}, excluded)


def capture_endpoints(args, output, states):
    from tools.audit_tpa_geometry import run as geometry_run
    from tools.audit_tpa_gradients import capture_identity, save_gradients
    from tools.capture_tpa_gradients import capture
    from tools.diagnose_rare_fp_regions import fingerprint

    captures = {}
    for side in ("old", "new"):
        geometry_path = output / f"{side}_geometry.json"
        geometry = geometry_run(SimpleNamespace(source_json=str(output / "regions.json"),
            output=str(geometry_path), side=side, checkpoint=getattr(args, side + "_checkpoint"),
            prompt_bank=args.prompt_bank, annotations=args.annotations, cpu_threads=args.cpu_threads))
        probe_args = SimpleNamespace(config_file=args.config_file, checkpoint=getattr(args, side + "_checkpoint"),
            windows=args.windows, microbatches=args.microbatches, batch_size=args.batch_size,
            seed=args.seed, device=args.device, geometry_json=str(geometry_path))
        identity = capture_identity(probe_args, geometry)
        signature = fingerprint(identity)
        path = output / f"{side}_gradients.pt"
        if path.exists():
            saved = load_trusted_torch_file(path)
            if saved.get("fingerprint") != signature or saved.get("inputs") != identity:
                raise ValueError("Gradient cache identity changed; refusing mixed endpoint probes")
            captures[side] = saved["capture"]
            annotation = captures[side]["train_annotations"]
            if file_identity(annotation["path"])["sha256"] != annotation["sha256"]:
                raise ValueError("Training annotations changed since cached gradient probe")
            print(f"[reuse gradients] {side}, no additional training-image probes", flush=True)
        else:
            captures[side] = capture(probe_args, geometry, states[side])
            save_gradients(path, {"inputs": identity, "fingerprint": signature, "capture": captures[side]})
            print(f"[save gradients] {path}", flush=True)
        if not captures[side]["weights_unchanged"] or captures[side]["optimizer_created"]:
            raise ValueError("Probe does not certify frozen checkpoint / no optimizer")
        gc.collect()
    return captures


def display(report):
    print("\n=== Frozen regional TP-FP margin change: 10ep - 8ep ===", flush=True)
    print("class                   terminal-TPA   query/bias   CLIP/ROI       total", flush=True)
    effects = report["endpoint_effects"]["margin_change"]
    for name, summary in effects["total"].items():
        values = [effects[k][name]["mean_tp_minus_fp"] for k in
                  ("terminal_tpa_bank", "query_and_bias_path", "clip_roi_path", "total")]
        print(f"{name:23} " + " ".join(f"{v:+11.6f}" if v is not None else "   NO_PAIRS" for v in values), flush=True)
    print("\n=== Local TPA descent: derivative of fused-log TP-FP margin ===", flush=True)
    print("Positive = opens margin; negative = closes it. Common clipping, no LR/AdamW step.", flush=True)
    for side, windows in report["local_gradient_audit"].items():
        for window in windows:
            for direction in ("detector", "rpsa", "apr", "projection_added", "unprojected_total", "routed_total"):
                row = window["directions"][direction]
                for name, margin in row["margin_derivative"].items():
                    print(f"{side} window={window['window']} {direction:18} {name:23} "
                          f"dMargin={margin['mean_tp_minus_fp']} cos(-g,deltaTPA)="
                          f"{row['cos_descent_with_8ep_to_10ep_tpa_delta']}", flush=True)
    print(f"[probe pairing] {report['probe_pairing']}", flush=True)
    print("Scope: selected 10ep FP/TP labels carried to IoU-paired 8ep regions. Terminal-bank swap "
          "does not isolate upstream TPA influence. Endpoint gradients are local Euclidean probes, "
          "not historical DDP/AdamW updates, AP predictions or causal attribution of the full APr gap.", flush=True)


def run(args):
    torch.set_num_threads(max(1, args.cpu_threads))
    names, identities, states = preflight(args)
    output = Path(args.output_dir).resolve()
    _, image_ids = prepare(args, names, identities, output)
    if args.prepare_only:
        print("[prepared] CPU selection only; rerun without --prepare-only to allow the bounded GPU audit", flush=True)
        return
    regions = dump_and_pair(args, names, identities, image_ids, output)
    prompts = torch.from_numpy(np.load(args.prompt_bank, allow_pickle=False)).float()
    banks, features, indices, records, clip, excluded = load_panel(regions, output / "pairing_cache", states, prompts)
    effects = endpoint_effects(features, indices, banks, records["new"], clip, .3)
    captures = capture_endpoints(args, output, states)
    report = {"schema_version": 1, "complete": False, "sources": identities,
              "categories": names, "selected_validation_images": image_ids,
              "gradient_probe_image_exposures": 2 * args.windows * args.microbatches * args.batch_size,
              "paired_regions": records, "excluded_regions": excluded,
              "endpoint_effects": effects, "probe_pairing": probe_pairing(captures["old"], captures["new"]),
              "local_gradient_audit": {}, "optimizer_created": False,
              "probe_metadata": {s: {k: v for k, v in c.items() if k != "windows"}
                                 for s, c in captures.items()},
              "scope": "selected regional score attribution and local TPA-only gradients; NOT historical optimizer replay or full-validation AP"}
    save_json(output / "report.json", report)
    delta = flat_delta(states["old"], states["new"])
    for side in ("old", "new"):
        print(f"[CPU JVP] {side}; frozen paired queries/boxes/CLIP, no parameter update", flush=True)
        report["local_gradient_audit"][side] = local_margin_audit(
            captures[side], states[side], prompts, banks[side], features[side], indices, records[side], .3, delta)
        save_json(output / "report.json", report)
    report["complete"] = True
    save_json(output / "report.json", report)
    display(report)
    print(f"[done] {output / 'report.json'}; no training or AP sweep", flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("comparison", "old-checkpoint", "new-checkpoint", "old-predictions", "new-predictions", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--config-file", default="lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_no_radius.py")
    parser.add_argument("--annotations", default="dataset/lvis/lvis_v1_val.json")
    parser.add_argument("--prompt-bank", default="dataset/metadata/lvis_claude_prompts_convnextl.npy")
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--windows", type=int, default=2)
    parser.add_argument("--microbatches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
