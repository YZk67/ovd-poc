#!/usr/bin/env python3
"""Compare native detector/CLIP scores around a small set of saved rare FP regions.

Existing pairing caches are read-only. --fill-missing permits one native forward
per missing image/checkpoint into a separate cache. No training or LVIS AP sweep.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from lami_dino.checkpoint_init import load_trusted_torch_file
from tools.compare_rare_pr_reports import file_identity, save_json
from tools.rare_region_pairing_ops import (
    collect_regions, compare_region, fp_overlap_report, replay_sample,
)


def fingerprint(inputs):
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def read_manifest(path):
    with Path(path).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if fingerprint(manifest["inputs"]) != manifest["fingerprint"]:
        raise ValueError(f"Manifest fingerprint mismatch: {path}")
    return manifest


def validate_sample(sample, bank, expected_fingerprint, side, image_id):
    for item in (sample, bank):
        if item["fingerprint"] != expected_fingerprint or item["label"] != side:
            raise ValueError("Image/bank cache fingerprint or side mismatch")
    if sample["image_id"] != image_id:
        raise ValueError("Cache image ID mismatch")
    checks = sample["native_replay_check"]
    if (not all(math.isfinite(checks[key]) for key in ("logit_max_abs_error", "score_max_abs_error"))
            or checks["logit_max_abs_error"] > 1e-4 or checks["score_max_abs_error"] > 5e-6):
        raise ValueError("Cache did not pass native classifier/fusion replay")


def compare_banks(parent, added):
    if set(parent) != set(added):
        raise ValueError("Supplemental prototype bank schema differs")
    errors = {}
    for key in parent:
        if key == "fingerprint":
            continue
        left, right = parent[key], added[key]
        if torch.is_tensor(left):
            if left.shape != right.shape or left.dtype != right.dtype:
                raise ValueError(f"Supplemental bank shape/dtype differs: {key}")
            if left.is_floating_point():
                error = float((left - right).abs().max()) if left.numel() else 0.0
                if not torch.isfinite(left).all() or not torch.isfinite(right).all() or error > 1e-6:
                    raise ValueError(f"Supplemental bank changed: {key}, max error={error}")
                errors[key] = error
            elif not torch.equal(left, right):
                raise ValueError(f"Supplemental bank changed: {key}")
        elif left != right:
            raise ValueError(f"Supplemental bank changed: {key}")
    return errors


def model_code_hash(config_file):
    # Exactly the source set used by the existing pairing manifest. Adding this
    # independent tool does not invalidate that cache's model code identity.
    sources = sorted(set(ROOT.glob("lami_dino/**/*.py")) | set(ROOT.glob("detrex/config/**/*.py"))
                     | set(ROOT.glob("configs/common/**/*.py")) | {
        ROOT / "tools/diagnose_detector_tpa_pairing.py", ROOT / "tools/pairing_lvis_support.py",
        ROOT / "tools/diagnose_tpa_usage.py", ROOT / config_file,
        ROOT / "detrex/modeling/backbone/convnext.py",
    })
    content = "".join(f"{path}:{file_identity(path)['sha256']}\n" for path in sources)
    return hashlib.sha256(content.encode()).hexdigest()


def fill_missing_images(args, parent, supplemental, missing, dataset, annotation_path, cache_dir):
    """Use the existing tested native dump, restricted to the missing IDs only."""
    inputs = parent["inputs"]
    if model_code_hash(args.config_file) != inputs["code_sha256"]:
        raise ValueError("Model/config code changed since pairing cache; refuse mixed forwards")
    for path, digest in inputs["asset_sha256"].items():
        if file_identity(path)["sha256"] != digest:
            raise ValueError(f"Model asset changed since pairing cache: {path}")
    checkpoint_paths = {}
    for side in ("old", "new"):
        checkpoint_paths[side] = getattr(args, side + "_checkpoint") or inputs[side + "_checkpoint"]
        if missing[side] and file_identity(checkpoint_paths[side])["sha256"] != inputs[side + "_sha256"]:
            raise ValueError(f"{side} checkpoint differs from pairing cache")
    if args.dump_device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("Missing-image forward needs CUDA in this interpreter; use the lami Python")
    from tools.diagnose_detector_tpa_pairing import dump_checkpoint

    dump_args = SimpleNamespace(
        config_file=args.config_file, annotations=str(annotation_path), device=args.dump_device,
        old_checkpoint=checkpoint_paths["old"], new_checkpoint=checkpoint_paths["new"],
        query_chunk_size=128, log_interval=1,
        **{key: inputs[key] for key in ("alpha", "beta", "novel_scale", "tpa_tau", "cls_tau", "max_dets")},
    )
    torch.manual_seed(inputs["seed"])
    for side in ("old", "new"):
        if missing[side]:
            dump_checkpoint(dump_args, side, {"image_ids": missing[side]}, dataset,
                            supplemental["fingerprint"], cache_dir)


def _display(report):
    print("\n=== New FP box overlaps within each image/category ===", flush=True)
    for group in report["fp_overlaps"]:
        print(f"{group['category']} image={group['image_id']} n={len(group['detection_ids'])} "
              f"pairs>=.5:{group['pairs_iou_ge_050']} >=.75:{group['pairs_iou_ge_075']} "
              f"maxIoU={group['maximum_pair_iou']}", flush=True)
        if len(group["detection_ids"]) > 1:
            print("  detection_ids:", group["detection_ids"], flush=True)
            print("  IoU matrix:", [[round(v, 3) for v in row] for row in group["iou_matrix"]], flush=True)
    print("\n=== Native region scores: source vs other checkpoint ===", flush=True)
    print("Other-nearest uses geometry only; other-best selects highest same-class score "
          "among boxes meeting region IoU. Different query IDs/boxes are not a fixed-ROI counterfactual.", flush=True)
    for row in report["regions"]:
        print(f"\n{row['category']} {row['source_side']}_{row['kind']} image={row['image_id']} "
              f"det={row['detection_id']} class_rank={row['class_global_rank']} "
              f"identity={row['source_query_identity']} "
              f"same-class region top300 counts source/other="
              f"{row['source_retained_query_count']}/{row['other_retained_query_count']}", flush=True)
        named = [(f"source_{i}", entry) for i, entry in enumerate(row["source_candidates"])]
        named += [(row["other_side"] + "_nearest", row["other_nearest_box"]),
                  (row["other_side"] + "_best", row["other_best_score_box"])]
        for label, item in named:
            if item is None:
                print(f"  {label}: NO_MATCH", flush=True)
                continue
            print(f"  {label}: q={item['query_id']} IoU={item['region_iou']:.4f} "
                  f"det={item['detector_probability']:.6g} CLIP={item['clip_probability']:.6g} "
                  f"fused={item['fused_score']:.6g} det_rank={item['detector_rank']} "
                  f"CLIP_rank={item['clip_rank']} top300={item['in_image_topk']} "
                  f"CLIP_top1={item['clip_top1']} box={item['box_xyxy']}", flush=True)
            if "overlapping_annotation_class" in item:
                print("    overlapping annotation category:", item["overlapping_annotation_class"], flush=True)
        if row["new_minus_old_log_score_components"] is not None:
            print("  new-old weighted log contributions (paired native boxes):",
                  row["new_minus_old_log_score_components"], flush=True)


def run(args):
    if args.cpu_threads < 1 or args.max_images < 1 or not 0 < args.match_iou <= 1:
        raise ValueError("Invalid threads, image budget or IoU threshold")
    torch.set_num_threads(args.cpu_threads)
    details_path, parent_dir = Path(args.fp_details).resolve(), Path(args.pairing_cache).resolve()
    output = Path(args.output).resolve()
    with details_path.open(encoding="utf-8") as stream:
        details = json.load(stream)
    parent = read_manifest(parent_dir / "manifest.json")
    inputs = parent["inputs"]
    if inputs["schema_version"] != 1:
        raise ValueError("Unsupported parent pairing-cache schema")
    annotation_path = Path(args.annotations or details["sources"]["annotations"]["path"]).resolve()
    protected = {details_path, annotation_path}
    for source in details.get("sources", {}).values():
        if isinstance(source, dict) and source.get("path"):
            protected.add(Path(source["path"]).resolve())
    for side in ("old", "new"):
        for value in (inputs.get(side + "_checkpoint"), getattr(args, side + "_checkpoint", None)):
            if value:
                protected.add(Path(value).resolve())
    protected.update(Path(path).resolve() for path in inputs.get("asset_sha256", {}))
    if output in protected or parent_dir in output.parents:
        raise ValueError("Output must not overwrite source details/annotations or parent pairing cache")
    annotation_identity = file_identity(annotation_path)
    if annotation_identity["sha256"] != inputs["annotations_sha256"] or (
        annotation_identity["sha256"] != details["sources"]["annotations"]["sha256"]
    ):
        raise ValueError("FP details, pairing cache and annotations differ")
    if details["max_dets"] != inputs["max_dets"]:
        raise ValueError("FP report and pairing cache max_dets differ")
    with annotation_path.open(encoding="utf-8") as stream:
        dataset = json.load(stream)
    categories = {category["id"]: category for category in dataset["categories"]}
    selected = collect_regions(details, args.categories)
    image_ids = sorted({row["image_id"] for row in selected})
    if not image_ids or len(image_ids) > args.max_images:
        raise ValueError(f"Selected {len(image_ids)} images; allowed budget is 1..{args.max_images}")
    print(f"[select] classes={args.categories} regions={len(selected)} images={image_ids}", flush=True)
    banks = {side: load_trusted_torch_file(parent_dir / side / "bank.pt") for side in ("old", "new")}
    for side, bank in banks.items():
        if bank["fingerprint"] != parent["fingerprint"] or bank["label"] != side:
            raise ValueError("Parent bank identity mismatch")
        if bank["category_ids"] != sorted(categories):
            raise ValueError("Bank vocabulary differs from annotations")
    for key in ("category_ids", "vlm_temperature"):
        if banks["old"][key] != banks["new"][key]:
            raise ValueError(f"Old/new bank {key} differs")
    for key in ("vlm_text", "novel_mask"):
        if not torch.equal(banks["old"][key], banks["new"][key]):
            raise ValueError(f"Old/new shared CLIP input differs: {key}")
    supplement_dir = output.parent / "region_cache"
    if supplement_dir == parent_dir or parent_dir in supplement_dir.parents:
        raise ValueError("Supplemental cache must be separate from parent cache")
    supplemental_inputs = {
        "schema_version": 1, "parent_fingerprint": parent["fingerprint"],
        "details_sha256": file_identity(details_path)["sha256"], "image_ids": image_ids,
    }
    supplemental = {"inputs": supplemental_inputs, "fingerprint": fingerprint(supplemental_inputs)}
    manifest_path = supplement_dir / "manifest.json"
    if manifest_path.exists() and read_manifest(manifest_path) != supplemental:
        raise ValueError("Supplemental cache belongs to another selection; use a new output directory")
    availability, missing = {}, {"old": [], "new": []}
    for side in ("old", "new"):
        for image_id in image_ids:
            if image_id in inputs["image_ids"] and (parent_dir / side / f"{image_id}.pt").exists():
                availability[side, image_id] = (parent_dir, parent["fingerprint"])
            elif (manifest_path.exists() and (supplement_dir / side / "bank.pt").exists()
                  and (supplement_dir / side / f"{image_id}.pt").exists()):
                availability[side, image_id] = (supplement_dir, supplemental["fingerprint"])
            else:
                missing[side].append(image_id)
    print(f"[cache] reused={len(availability)} missing={missing}", flush=True)
    report = {
        "complete": False, "selected_image_ids": image_ids, "missing_images": missing,
        "parent_fingerprint": parent["fingerprint"], "protocol": {
            key: inputs[key] for key in ("alpha", "beta", "novel_scale", "tpa_tau", "cls_tau", "max_dets")
        },
        "source_details": file_identity(details_path), "annotations": annotation_identity,
        "checkpoints": {side: {"path": inputs.get(side + "_checkpoint"), "sha256": inputs.get(side + "_sha256")}
                        for side in ("old", "new")},
        "sample_sources": [],
        "fp_overlaps": fp_overlap_report(selected), "regions": [],
        "scope": "selected saved errors and TP controls; not full-validation AP or training attribution",
        "pairing_policy": "source recovered by bbox+score; other-model boxes paired by IoU, not query index",
    }
    save_json(output, report)
    if any(missing.values()):
        if not args.fill_missing:
            print("[incomplete] Missing cached images. Add --fill-missing to allow only these native forwards.", flush=True)
            return report
        save_json(manifest_path, supplemental)
        fill_missing_images(args, parent, supplemental, missing, dataset, annotation_path, supplement_dir)
        for side in ("old", "new"):
            for image_id in missing[side]:
                availability[side, image_id] = (supplement_dir, supplemental["fingerprint"])
    active_banks = {(side, parent_dir): bank for side, bank in banks.items()}
    for side in ("old", "new"):
        if any(directory == supplement_dir for (label, _), (directory, _) in availability.items() if label == side):
            added = load_trusted_torch_file(supplement_dir / side / "bank.pt")
            compare_banks(banks[side], added)
            active_banks[side, supplement_dir] = added
    regions_by_image = defaultdict(list)
    for region in selected:
        regions_by_image[region["image_id"]].append(region)
    for number, image_id in enumerate(image_ids, 1):
        replays = {}
        for side in ("old", "new"):
            directory, expected_fp = availability[side, image_id]
            sample = load_trusted_torch_file(directory / side / f"{image_id}.pt")
            bank = active_banks[side, directory]
            validate_sample(sample, bank, expected_fp, side, image_id)
            report["sample_sources"].append({"side": side, "image_id": image_id,
                                             "directory": str(directory), "fingerprint": expected_fp,
                                             "native_replay_check": sample["native_replay_check"]})
            replays[side] = replay_sample(sample, bank, inputs, device=args.device)
        for region in regions_by_image[image_id]:
            report["regions"].append(compare_region(region, replays, categories, inputs, match_iou=args.match_iou))
        print(f"[replay] {number}/{len(image_ids)} image={image_id} saved bbox/score checks: PASS", flush=True)
        del replays
    report["complete"] = True
    report["missing_images"] = {"old": [], "new": []}
    report["ambiguous_source_detections"] = sum(row["source_query_identity"] != "unique" for row in report["regions"])
    save_json(output, report)
    _display(report)
    print(f"[save] {output} complete=True", flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp-details", required=True)
    parser.add_argument("--pairing-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--categories", nargs="+", default=["koala", "roller_skate", "joystick"])
    parser.add_argument("--match-iou", type=float, default=.5)
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", default="cpu", help="Device for cached score replay")
    parser.add_argument("--fill-missing", action="store_true", help="Allow only selected missing-image native forwards")
    parser.add_argument("--dump-device", default="cuda:0")
    parser.add_argument("--annotations")
    parser.add_argument("--old-checkpoint")
    parser.add_argument("--new-checkpoint")
    parser.add_argument("--config-file", default="lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py")
    return parser.parse_args(argv)


if __name__ == "__main__":
    result = run(parse_args())
    sys.exit(0 if result["complete"] else 2)
