#!/usr/bin/env python3
"""Audit fixed-radius TPA geometry on CPU using one checkpoint and cached queries.

Only the text-side TPA forward is reconstructed. No detector/image forward,
gradient audit, training, top-300 reselection or AP evaluation is performed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from lami_dino.checkpoint_init import load_trusted_torch_file
from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.diagnose_rare_fp_regions import compare_banks, read_manifest, validate_sample
from tools.tpa_geometry_audit_ops import (
    BUFFERS, class_geometry, extract_shared_tpa, fixed_query_audit, reconstruct_tpa,
    tensor_digest, validate_reconstructed_bank,
)


GEOMETRY_KEYS = ("mean_shift_over_value_center_norm", "pre_post_center_angle_deg",
                 "post_value_center_angle_deg", "mean_unit_residual_norm",
                 "pre_post_unit_mean_angle_deg")


def summarize_classes(classes):
    result = {}
    for split in ("all", "r", "c", "f"):
        rows = [c for c in classes if split == "all" or c["frequency"] == split]
        summary = {"classes": len(rows), "statistics": {}}
        for key in GEOMETRY_KEYS:
            values = [r[key] for r in rows if r[key] is not None]
            summary["statistics"][key] = ({"count": len(values), "mean": mean(values),
                                           "median": median(values), "max": max(values)}
                                          if values else {"count": 0})
        result[split] = summary
    return result


def summarize_queries(regions):
    groups = defaultdict(dict)
    for row in regions:
        # Ambiguous identities are reported but excluded from group estimates.
        if len(row["queries"]) == 1:
            entry = row["queries"][0]
            groups[row["category"], row["kind"]][row["image_id"], entry["query_id"]] = entry
    output = []
    for (category, kind), identities in sorted(groups.items()):
        entries = list(identities.values())
        output.append({"category": category, "kind": kind, "unique_queries": len(entries),
                       "after_minus_before_mean": {
                           k: mean(e["after_minus_before"][k] for e in entries)
                           for k in entries[0]["after_minus_before"]}})
    ordering = []
    for category in sorted({c for c, _ in groups}):
        fps, tps = groups[category, "fp"], groups[category, "tp"]
        row = {"category": category, "fp_queries": len(fps), "tp_queries": len(tps),
               "selected_pairs": len(fps) * len(tps)}
        for stage in ("before", "after"):
            differences = [fp["scores"][stage]["fused_log_score"] - tp["scores"][stage]["fused_log_score"]
                           for fp in fps.values() for tp in tps.values()]
            row["fp_above_tp_" + stage] = sum(d > 1e-8 for d in differences)
            row["ties_" + stage] = sum(abs(d) <= 1e-8 for d in differences)
        ordering.append(row)
    return output, ordering


def display(report):
    print("\n=== Fixed-radius TPA geometry: one checkpoint, before -> after ===", flush=True)
    print(f"side={report['side']} iteration={report['tpa_state']['iteration']} "
          f"slot_prior={report['parameters']['slot_prior_strength']} "
          f"mode_strength={report['parameters']['prototype_mode_strength']}", flush=True)
    print("c = mean(projected prompts), a = mean(pre-radius slots), b = mean(post-radius slots)", flush=True)
    print("Enabled transform: b = c + radius * mean(unit(pre-radius slot - c)); b need not equal c or a.", flush=True)
    print("split classes  mean ||b-a||/||c||  median angle(a,b)  max angle(a,b)", flush=True)
    for split, group in report["geometry_summary"].items():
        s = group["statistics"]
        def value(key, field):
            return s[key].get(field, float("nan"))
        print(f"{split:5} {group['classes']:7d} "
              f"{value('mean_shift_over_value_center_norm', 'mean'):18.5f} "
              f"{value('pre_post_center_angle_deg', 'median'):18.3f} "
              f"{value('pre_post_center_angle_deg', 'max'):15.3f}", flush=True)
    print("\n=== Selected classes: center rotation and normalization ===", flush=True)
    print("class             pre/post center angle  pre/post unit-mean norm  raw/unit center angle pre/post", flush=True)
    selected = {r["category"] for r in report["regions"]}
    for row in report["classes"]:
        if row["name"] not in selected:
            continue
        def angle(value):
            return "undefined" if value is None else f"{value:.3f}"
        print(f"{row['name']:17} {angle(row['pre_post_center_angle_deg']):>21}  "
              f"{row['before']['unit_slot_mean_norm']:.5f}/{row['after']['unit_slot_mean_norm']:.5f}  "
              f"{angle(row['before']['normalization_center_angle_deg'])}/"
              f"{angle(row['after']['normalization_center_angle_deg'])}", flush=True)
    print("\n=== Fixed-query effect of applying radius (after - before; means) ===", flush=True)
    print("class             kind     N   dCenter     dMode    dLogit   dMeanCorr    dUplift", flush=True)
    for row in report["query_summary"]:
        d = row["after_minus_before_mean"]
        print(f"{row['category']:17} {row['kind']:5} {row['unique_queries']:4d} "
              f"{d['center_response']:9.4f} {d['mode_delta']:9.4f} {d['native_logit']:9.4f} "
              f"{d['center_to_mean_correction']:11.4f} {d['dispersion_uplift']:10.4f}", flush=True)
    print("\n=== Selected FP > TP ordering; fixed queries/boxes/CLIP ===", flush=True)
    for row in report["selected_ordering"]:
        print(f"{row['category']}: pairs={row['selected_pairs']} "
              f"before={row['fp_above_tp_before']} after={row['fp_above_tp_after']} "
              f"ties={row['ties_before']}/{row['ties_after']}", flush=True)
    print("\nScope: direct forward effect with already-trained weights. NOT a training ablation, "
          "NOT full-model mode_scale=0, and NOT an AP prediction. Selected FP/TP labels are not rematched.", flush=True)


def run(args):
    if args.cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    source_path, output = Path(args.source_json).resolve(), Path(args.output).resolve()
    source = load_json(source_path)
    if source.get("complete") is not True or not source.get("regions"):
        raise ValueError("Need a complete FP-region report")
    side = args.side
    rows = [r for r in source["regions"] if r["source_side"] == side]
    if not rows:
        raise ValueError(f"No source regions for {side}")
    checkpoint_path = Path(args.checkpoint or source["checkpoints"][side]["path"]).resolve()
    prompt_path = Path(args.prompt_bank).resolve()
    annotation_path = Path(args.annotations or source["annotations"]["path"]).resolve()
    sources = {}
    for entry in source["sample_sources"]:
        key = (entry["side"], entry["image_id"])
        if key in sources:
            raise ValueError("Duplicate sample-source identity")
        sources[key] = entry
    cache_dirs = {Path(e["directory"]).resolve() for e in sources.values()}
    protected = {source_path, checkpoint_path, prompt_path, annotation_path}
    protected.update(Path(e["path"]).resolve() for e in source["checkpoints"].values())
    if source.get("source_details", {}).get("path"):
        protected.add(Path(source["source_details"]["path"]).resolve())
    if output in protected or any(output == d or d in output.parents for d in cache_dirs):
        raise ValueError("Output must not overwrite inputs or caches")
    print(f"[load] side={side}; hashing checkpoint and reading cached text/queries; CPU only", flush=True)
    checkpoint_identity = file_identity(checkpoint_path)
    if checkpoint_identity["sha256"] != source["checkpoints"][side]["sha256"]:
        raise ValueError("Checkpoint identity differs from source report")
    if file_identity(annotation_path)["sha256"] != source["annotations"]["sha256"]:
        raise ValueError("Annotation identity differs from source report")
    categories = sorted(load_json(annotation_path)["categories"], key=lambda c: c["id"])
    class_index = {c["name"]: i for i, c in enumerate(categories)}
    selected_dirs = {Path(sources[side, r["image_id"]]["directory"]).resolve() for r in rows}
    banks, manifests = {}, {}
    for directory in sorted(selected_dirs):
        m = read_manifest(directory / "manifest.json")
        if m["fingerprint"] == source["parent_fingerprint"]:
            if any(m["inputs"][k] != v for k, v in source["protocol"].items()):
                raise ValueError("Source protocol differs from parent cache")
            if (m["inputs"]["annotations_sha256"] != source["annotations"]["sha256"]
                    or m["inputs"][side + "_sha256"] != checkpoint_identity["sha256"]):
                raise ValueError("Parent cache annotation/checkpoint identity differs")
        elif (m["inputs"].get("parent_fingerprint") != source["parent_fingerprint"]
              or m["inputs"].get("details_sha256") != source["source_details"]["sha256"]):
            raise ValueError("Unrelated supplemental cache")
        bank = load_trusted_torch_file(directory / side / "bank.pt")
        if bank["fingerprint"] != m["fingerprint"] or bank["label"] != side:
            raise ValueError("Bank identity differs from manifest")
        if bank["category_ids"] != [c["id"] for c in categories]:
            raise ValueError("Cached class order differs from annotations")
        if not torch.equal(bank["novel_mask"], torch.tensor([c["frequency"] == "r" for c in categories])):
            raise ValueError("Cached novel mask differs from annotations")
        if bank["tpa_tau"] != source["protocol"]["tpa_tau"] or bank["temperature"] != source["protocol"]["cls_tau"]:
            raise ValueError("Cached temperatures differ from source protocol")
        if banks:
            compare_banks(next(iter(banks.values())), bank)
        banks[directory], manifests[directory] = bank, m
    bank = next(iter(banks.values()))
    checkpoint = load_trusted_torch_file(checkpoint_path)
    state, state_info = extract_shared_tpa(checkpoint)
    del checkpoint
    if state_info["iteration"] != bank["iteration"]:
        raise ValueError("Checkpoint iteration differs from cache")
    for key in BUFFERS:
        if abs(float(state[key]) - float(bank[key])) > 1e-7:
            raise ValueError(f"Checkpoint/cache {key} differs")
    prompts = torch.from_numpy(np.load(prompt_path, allow_pickle=False)).float()
    if tensor_digest(prompts) != bank["prompt_sha256"]:
        raise ValueError("Prompt tensor identity differs from cached eval_text_feats")
    print(f"[reconstruct] prompts={tuple(prompts.shape)}, shared aliases={len(state_info['prefixes'])}", flush=True)
    reconstruction = reconstruct_tpa(prompts, state, bank["tpa_tau"])
    checks = validate_reconstructed_bank(reconstruction, bank)
    print(f"[native-bank check] {checks}", flush=True)
    report = {"schema_version": 1, "side": side, "complete": False,
              "source_report": file_identity(source_path), "checkpoint": checkpoint_identity,
              "prompt_bank": file_identity(prompt_path), "annotations": source["annotations"],
              "protocol": source["protocol"], "parent_fingerprint": source["parent_fingerprint"],
              "tpa_state": state_info, "parameters": {k: float(state[k]) for k in BUFFERS},
              "native_bank_checks": checks,
              "definitions": {
                  "before": "attention-weighted projected prompts; learned weights and slot prior unchanged",
                  "after": "existing fixed-radius semantic-mode forward applied to before",
                  "value_center": "mean of projected prompts, NOT mean of pre/post prototype slots",
                  "shift": "post prototype mean minus pre prototype mean",
                  "normalization": "each raw slot is normalized separately, then unit slots averaged",
                  "scope": "direct forward intervention on coadapted weights, not a training causal attribution",
                  "queries": "source-side native FP/TP only; ambiguous IDs excluded from summaries; fixed boxes and CLIP",
                  "ordering": "post-hoc selected FP/TP pairs only; no reselection/rematching or AP estimate"},
              "classes": [{**c, **class_geometry(reconstruction, i)} for i, c in enumerate(categories)],
              "regions": []}
    by_image = defaultdict(list)
    for row in rows:
        by_image[row["image_id"]].append(row)
    for number, (image_id, image_rows) in enumerate(sorted(by_image.items()), 1):
        source_entry = sources[side, image_id]
        directory = Path(source_entry["directory"]).resolve()
        manifest = manifests[directory]
        if (source_entry["fingerprint"] != manifest["fingerprint"]
                or image_id not in manifest["inputs"]["image_ids"]):
            raise ValueError("Sample-source identity differs from manifest")
        sample = load_trusted_torch_file(directory / side / f"{image_id}.pt")
        validate_sample(sample, banks[directory], manifest["fingerprint"], side, image_id)
        for row in image_rows:
            if not row["source_candidates"]:
                raise ValueError("No source queries for selected region")
            report["regions"].append({
                "category": row["category"], "kind": row["kind"], "image_id": image_id,
                "detection_id": row["detection_id"], "cache_directory": str(directory),
                "queries": [fixed_query_audit(e, sample, banks[directory], class_index[row["category"]],
                                              source["protocol"], reconstruction)
                            for e in row["source_candidates"]]})
        print(f"[queries] {number}/{len(by_image)} image={image_id} native replay: PASS", flush=True)
    report["geometry_summary"] = summarize_classes(report["classes"])
    report["query_summary"], report["selected_ordering"] = summarize_queries(report["regions"])
    report["complete"] = True
    save_json(output, report)
    display(report)
    print(f"[save] {output}", flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", required=True, help="Completed diagnose_rare_fp_regions report.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--side", choices=("new", "old"), default="new")
    parser.add_argument("--checkpoint", help="Optional relocated checkpoint; SHA256 must match source")
    parser.add_argument("--prompt-bank", default="dataset/metadata/lvis_claude_prompts_convnextl.npy")
    parser.add_argument("--annotations", help="Optional relocated annotations; SHA256 must match source")
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
