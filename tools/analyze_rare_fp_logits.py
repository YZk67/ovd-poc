#!/usr/bin/env python3
"""CPU-only logit decomposition on queries in an existing FP-region report.

No detector/checkpoint forward, image loading, top-300 reselection or AP evaluation.
Missing caches fail explicitly; this command never regenerates them.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from lami_dino.checkpoint_init import load_trusted_torch_file
from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.diagnose_rare_fp_regions import read_manifest, validate_sample
from tools.rare_logit_decomposition_ops import decompose_entry, paired_delta, summarize


def display(report):
    print("\n=== Fixed-query detector logit decomposition ===", flush=True)
    print("native logit = normalized raw-center response + signed mode delta + bias", flush=True)
    print("mode delta = center-to-mean correction + nonnegative LME dispersion uplift", flush=True)
    for row in report["regions"]:
        print(f"\n{row['category']} {row['source_side']}_{row['kind']} "
              f"image={row['image_id']} det={row['detection_id']}", flush=True)
        print("endpoint            q    center      mode      bias     logit   mean-corr  LME-uplift   fused/native->center", flush=True)
        entries = [(row["source_side"] + f"_source{i}", e) for i, e in enumerate(row["source_candidates"])]
        entries += [(row["other_side"] + "_nearest", row["other_nearest_box"]),
                    (row["other_side"] + "_best", row["other_best_score_box"])]
        for name, entry in entries:
            if entry is None:
                print(name + ": NO_MATCH", flush=True)
                continue
            t, c = entry["logit_terms"], entry["fixed_query_controls"]
            print(f"{name:17} {entry['query_id']:3d} {t['center_response']:9.4f} {t['mode_delta']:9.4f} "
                  f"{t['bias']:9.4f} {t['native_logit']:9.4f} {t['center_to_mean_correction']:11.4f} "
                  f"{t['dispersion_uplift']:11.4f}  {c['native']['fused_score']:.6g}->{c['centroid']['fused_score']:.6g}", flush=True)
    print("\n=== Paired regional new-old LOGIT deltas (means; new-source regions only) ===", flush=True)
    print("class             region   pairs   dCenter     dMode     dBias    dLogit   dMeanCorr   dUplift", flush=True)
    for row in report["paired_summary"]:
        d = row["delta_mean"]
        print(f"{row['category']:17} {row['region_kind']:7} {row['pairs']:5d} "
              f"{d['center_response']:9.4f} {d['mode_delta']:9.4f} {d['bias']:9.4f} "
              f"{d['native_logit']:9.4f} {d['center_to_mean_correction']:11.4f} "
              f"{d['dispersion_uplift']:9.4f}", flush=True)
    print("\n=== Selected new-FP/new-TP ordering, fixed queries and CLIP ===", flush=True)
    for row in report["selected_ordering"]:
        print(f"{row['category']}: FP={row['fp_queries']} TP={row['tp_queries']} "
              f"pairs={row['selected_pairs']} FP>TP native={row['fp_above_tp_native']} "
              f"centroid={row['fp_above_tp_centroid']} "
              f"native-only inversions={row['native_only_inversions']} "
              f"centroid-only inversions={row['centroid_only_inversions']} "
              f"ties native/centroid={row['native_ties']}/{row['centroid_ties']}", flush=True)
    print("\nScope: selected errors/controls, not full PR/AP. No top-300 reselection or TP/FP rematching. "
          "Centroid changes only the terminal classifier, NOT full model mode_scale=0. "
          "Paired checkpoint queries/boxes differ; component differences do not identify a training cause.", flush=True)


def run(args):
    if args.cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    source_path, output = Path(args.source_json).resolve(), Path(args.output).resolve()
    source = load_json(source_path)
    if source.get("complete") is not True or not source.get("regions"):
        raise ValueError("Need a complete, nonempty FP-region report")
    annotation_path = Path(args.annotations or source["annotations"]["path"]).resolve()
    sources = {}
    protected = {source_path, annotation_path}
    for name in ("source_details",):
        if source.get(name, {}).get("path"):
            protected.add(Path(source[name]["path"]).resolve())
    protected.update(Path(value["path"]).resolve() for value in source.get("checkpoints", {}).values() if value.get("path"))
    for entry in source["sample_sources"]:
        key = (entry["side"], entry["image_id"])
        if key in sources:
            raise ValueError("Duplicate sample-source identity")
        sources[key] = entry
    cache_dirs = {Path(entry["directory"]).resolve() for entry in sources.values()}
    if output in protected or any(output == root or root in output.parents for root in cache_dirs):
        raise ValueError("Output must not overwrite inputs or cache files")
    if file_identity(annotation_path)["sha256"] != source["annotations"]["sha256"]:
        raise ValueError("Annotation identity differs from source report")
    categories = {c["name"]: c["id"] for c in load_json(annotation_path)["categories"]}
    manifests = {directory: read_manifest(directory / "manifest.json") for directory in cache_dirs}
    for directory, manifest in manifests.items():
        if manifest["fingerprint"] == source["parent_fingerprint"]:
            if any(manifest["inputs"][k] != v for k, v in source["protocol"].items()):
                raise ValueError("Source protocol differs from parent cache")
            if manifest["inputs"]["annotations_sha256"] != source["annotations"]["sha256"]:
                raise ValueError("Parent-cache annotations differ from source report")
        elif (manifest["inputs"].get("parent_fingerprint") != source["parent_fingerprint"]
              or manifest["inputs"].get("details_sha256") != source["source_details"]["sha256"]):
            raise ValueError(f"Unrelated supplemental cache: {directory}")
    bank_cache, by_image = {}, defaultdict(list)
    for row in source["regions"]:
        by_image[row["image_id"]].append(row)
    report = {"schema_version": 1, "source_report": file_identity(source_path),
              "protocol": source["protocol"], "parent_fingerprint": source["parent_fingerprint"],
              "sample_sources": source["sample_sources"], "regions": [],
              "definitions": {
                  "center": "scale * dot(normalize(feature), normalize(mean(raw_prototypes)))",
                  "mode_delta": "native_logit - bias - center; SIGNED",
                  "mean_slot": "scale * mean(dot(normalize(feature), normalize(each raw prototype)))",
                  "dispersion_uplift": "native_logit - bias - mean_slot; NONNEGATIVE",
                  "center_to_mean_correction": "mean_slot - center",
                  "fixed_query_centroid": "sigmoid(center+bias), with same native query/box/CLIP; no reselection",
              }, "scope": {
                  "selection": "post-hoc selected errors and TP controls, not the full validation population",
                  "paired_labels": "new_fp/new_tp labels describe new-source regions; old counterparts are not rematched TP/FP",
                  "summary": "new-source regions only; repeated old queries and overlapping regions are not independent observations",
                  "counterfactual": "terminal classifier only, with frozen native queries/boxes/CLIP; not full mode_scale=0",
                  "ordering": "selected fixed-query FP/TP pairs, no global top-k reselection, ignore-rule rematching or AP estimate",
                  "attribution": "logit arithmetic, not an identification of feature vs classifier training causes",
              }, "complete": False}
    print(f"[load] {len(source['regions'])} regions on {len(by_image)} images; CPU cache only, no forward", flush=True)
    for number, (image_id, rows) in enumerate(sorted(by_image.items()), 1):
        samples, banks = {}, {}
        for side in ("old", "new"):
            entry = sources[side, image_id]
            directory = Path(entry["directory"]).resolve()
            if entry["fingerprint"] != manifests[directory]["fingerprint"]:
                raise ValueError("Saved sample fingerprint differs from its manifest")
            if image_id not in manifests[directory]["inputs"]["image_ids"]:
                raise ValueError("Image absent from cache manifest")
            bank_key = (side, directory)
            if bank_key not in bank_cache:
                bank_cache[bank_key] = load_trusted_torch_file(directory / side / "bank.pt")
            banks[side] = bank_cache[bank_key]
            samples[side] = load_trusted_torch_file(directory / side / f"{image_id}.pt")
            validate_sample(samples[side], banks[side], entry["fingerprint"], side, image_id)
            if banks[side]["category_ids"] != sorted(categories.values()):
                raise ValueError("Cached category order differs from annotations")
        memo = {}
        for row in rows:
            category_id = categories[row["category"]]
            def expand(entry, side):
                if entry is None:
                    return None
                # Cache exact repeated endpoints, but validate each distinct saved entry.
                key = (side, category_id, json.dumps(entry, sort_keys=True))
                if key not in memo:
                    memo[key] = decompose_entry(entry, samples[side], banks[side],
                                                banks[side]["category_ids"].index(category_id), source["protocol"])
                return memo[key]
            new_row = {**row, "source_candidates": [expand(e, row["source_side"]) for e in row["source_candidates"]],
                       "other_nearest_box": expand(row["other_nearest_box"], row["other_side"]),
                       "other_best_score_box": expand(row["other_best_score_box"], row["other_side"])}
            new_row["new_minus_old_logit_terms"] = (
                paired_delta(new_row["source_candidates"][0], new_row["other_best_score_box"], row["source_side"])
                if len(new_row["source_candidates"]) == 1 and new_row["other_best_score_box"] is not None else None
            )
            report["regions"].append(new_row)
        print(f"[decompose] {number}/{len(by_image)} image={image_id} native replay and additive checks: PASS", flush=True)
    report["paired_summary"], report["selected_ordering"] = summarize(report["regions"])
    report["complete"] = True
    save_json(output, report)
    display(report)
    print(f"[save] {output}", flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", required=True, help="Completed diagnose_rare_fp_regions report.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--annotations", help="Optional relocated annotation path; SHA256 must match")
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
