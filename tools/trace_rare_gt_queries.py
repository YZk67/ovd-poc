#!/usr/bin/env python3
"""Trace scarecrow 137708 before top-300 at no-radius 8ep/12ep.

Reuse authenticated dense pairing caches, otherwise at most ONE validation
image forward per checkpoint. No training, full-validation inference, swaps,
threshold sweep, or AP estimate. Reproduce every saved image prediction first.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from lami_dino.checkpoint_init import load_trusted_torch_file
from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from tools.diagnose_detector_tpa_pairing import (
    dump_checkpoint, input_asset_hashes, locked_manifest, save_tensor_file,
)
from tools.diagnose_rare_fp_regions import model_code_hash, read_manifest, validate_sample
from tools.rare_gt_query_ops import analyze_queries, selected_predictions, verify_predictions
from tools.rare_region_pairing_ops import replay_sample
from tools.tpa_geometry_audit_ops import extract_shared_tpa


PROTOCOL = dict(alpha=0., beta=.3, novel_scale=3., tpa_tau=.004375, cls_tau=.07, max_dets=300)
ITERATIONS = {"old": 56799, "new": 85199}
IMAGE_ID, GT_ID = 218917, 137708


def validate_trace(trace):
    if (trace.get("complete") is not True or trace.get("category") != "scarecrow"
            or trace.get("endpoint_labels") != {"A": "8ep", "B": "12ep"}
            or trace["gt"]["id"] != GT_ID or trace["gt"]["image_id"] != IMAGE_ID
            or trace["image"]["id"] != IMAGE_ID):
        raise ValueError("Expected completed single scarecrow GT trace for 8ep -> 12ep")
    for side in ITERATIONS:
        endpoint = trace["endpoints"][side]
        rows = endpoint["selected_detections_by_gt_iou"]
        if (len(rows) != 300 or endpoint["selected_predictions_count"] != 300
                or any(r["image_id"] != IMAGE_ID for r in rows)
                or len({r["detection_id"] for r in rows}) != 300):
            raise ValueError("Trace must contain all 300 distinct selected detections per endpoint")
        verify_predictions(rows, rows)


def validate_checkpoint(checkpoint, side):
    state, info = extract_shared_tpa(checkpoint)
    keys = set()
    for key in checkpoint.get("model", checkpoint):
        while key.startswith(("module.", "model.")):
            key = key.split(".", 1)[1]
        keys.add(key)
    if any(prefix + name not in keys for prefix in info["prefixes"]
           for name in ("slot_prior_strength", "prototype_mode_strength")):
        raise ValueError("No-radius buffers must be explicit; constructor fallback is not allowed")
    if info["iteration"] != ITERATIONS[side]:
        raise ValueError(f"{side}: expected iteration {ITERATIONS[side]}, found {info['iteration']}")
    if (float(state["prototype_mode_strength"]) != 0.
            or abs(float(state["slot_prior_strength"]) - .2) > 1e-6
            or tuple(state["prototype_queries"].shape) != (5, 256)):
        raise ValueError("Expected K5 no-radius checkpoint, slot_prior=.2; not the radius/legacy run")


def cache_compatible(inputs, expected, side, checkpoint_sha):
    """Labels and filenames alone cannot establish endpoint identity."""
    return (inputs.get(side + "_sha256") == checkpoint_sha
            and IMAGE_ID in inputs.get("image_ids", [])
            and inputs.get("annotations_sha256") == expected["annotations_sha256"]
            and inputs.get("code_sha256") == expected["code_sha256"]
            and inputs.get("asset_sha256") == expected["asset_sha256"]
            and all(inputs.get(k) == v for k, v in PROTOCOL.items()))


def candidate_manifests(args, own_cache):
    paths = [own_cache / "manifest.json"]
    for directory in args.reuse_cache:
        path = Path(directory).resolve() / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Explicit cache manifest missing: {path}")
        paths.append(path)
    # Bounded directory listings, NOT recursive raw-cache scanning.
    if args.cache_search_root:
        root = Path(args.cache_search_root).resolve()
        for pattern in ("*/pairing_cache/manifest.json", "*/cache/manifest.json"):
            paths.extend(sorted(root.glob(pattern)))
    return list(dict.fromkeys(p for p in paths if p.is_file()))


def find_caches(manifests, expected, identities):
    found = {}
    for path in manifests:
        raw = load_json(path)
        inputs = raw.get("inputs", {}) if isinstance(raw, dict) else {}
        if not isinstance(inputs, dict) or not {"old_sha256", "new_sha256", "image_ids"}.issubset(inputs):
            print(f"[skip cache] not a dense two-endpoint pairing manifest: {path}", flush=True)
            continue
        manifest = read_manifest(path)
        for label in ITERATIONS:
            for target in ITERATIONS:
                if target in found or not cache_compatible(
                        manifest["inputs"], expected, label, identities[target + "_checkpoint"]["sha256"]):
                    continue
                branch = path.parent / label
                if (branch / "bank.pt").is_file() and (branch / f"{IMAGE_ID}.pt").is_file():
                    found[target] = (path, label)
                    print(f"[reuse] {target}: {branch}", flush=True)
    return found


def load_cache(path, label, target, dataset, image):
    manifest = read_manifest(path)
    branch = path.parent / label
    bank = load_trusted_torch_file(branch / "bank.pt")
    sample = load_trusted_torch_file(branch / f"{IMAGE_ID}.pt")
    validate_sample(sample, bank, manifest["fingerprint"], label, IMAGE_ID)
    ids = sorted(c["id"] for c in dataset["categories"])
    rare = {c["id"] for c in dataset["categories"] if c["frequency"] == "r"}
    if (bank["iteration"] != ITERATIONS[target] or bank["category_ids"] != ids
            or bank["prototype_mode_strength"] != 0.
            or abs(bank["slot_prior_strength"] - .2) > 1e-6
            or tuple(bank["prototypes"].shape) != (1203, 5, 256)
            or abs(bank["tpa_tau"] - PROTOCOL["tpa_tau"]) > 1e-8
            or abs(bank["temperature"] - PROTOCOL["cls_tau"]) > 1e-8
            or not torch.equal(bank["novel_mask"].bool(), torch.tensor([c in rare for c in ids]))
            or tuple(sample["features"].shape) != (900, 256)
            or tuple(sample["query_boxes"].shape) != (900, 4)
            or tuple(sample["roi_features"].shape) != (900, 768)
            or (sample["width"], sample["height"]) != (image["width"], image["height"])):
        raise ValueError("Dense cache bank/image does not match this no-radius endpoint")
    for value in (*sample.values(), *bank.values()):
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise ValueError("Nonfinite dense cache")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Nonfinite cache setting")
    return sample, bank, {"manifest": file_identity(path),
                          "bank": file_identity(branch / "bank.pt"),
                          "sample": file_identity(branch / f"{IMAGE_ID}.pt"), "source_label": label}


def run(args):
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    trace = load_json(args.trace_json)
    validate_trace(trace)
    output = Path(args.output_dir).resolve()
    annotation = args.annotations or trace["sources"]["annotations"]["path"]
    paths = dict(trace=args.trace_json, annotations=annotation, config=args.config_file,
                 old_checkpoint=args.old_checkpoint, new_checkpoint=args.new_checkpoint)
    if any(Path(p).resolve() == output or output in Path(p).resolve().parents for p in paths.values()):
        raise ValueError("Use a separate output directory, not an input directory")
    identities = {}
    for key, path in paths.items():
        print(f"[identity] {key}: {path}", flush=True)
        identities[key] = file_identity(path)
    if identities["annotations"]["sha256"] != trace["sources"]["annotations"]["sha256"]:
        raise ValueError("Annotations changed since saved prediction trace")
    dataset = load_json(annotation)
    matching_gt = [r for r in dataset["annotations"] if r["id"] == GT_ID]
    matching_image = [r for r in dataset["images"] if r["id"] == IMAGE_ID]
    if matching_gt != [trace["gt"]] or matching_image != [trace["image"]]:
        raise ValueError("Target GT/image differs from source annotations")
    if sum(c["category_id"] == trace["gt"]["category_id"] for c in dataset["annotations"]) != 1:
        raise ValueError("This diagnostic requires the unique validation GT from the trace")
    for side in ITERATIONS:
        checkpoint = load_trusted_torch_file(paths[side + "_checkpoint"])
        validate_checkpoint(checkpoint, side)
        del checkpoint

    cache = output / "pairing_cache"
    dump_args = SimpleNamespace(**PROTOCOL, config_file=args.config_file, annotations=annotation,
        old_checkpoint=args.old_checkpoint, new_checkpoint=args.new_checkpoint,
        phase="dump", device=args.device, query_chunk_size=128, log_interval=1)
    expected = dict(PROTOCOL, schema_version=1,
                    annotations_sha256=identities["annotations"]["sha256"],
                    code_sha256=model_code_hash(args.config_file),
                    asset_sha256=input_asset_hashes(dump_args, cache),
                    image_ids=[IMAGE_ID], seed=42)
    for side in ITERATIONS:
        expected[side + "_checkpoint"] = identities[side + "_checkpoint"]["path"]
        expected[side + "_sha256"] = identities[side + "_checkpoint"]["sha256"]
    found = find_caches(candidate_manifests(args, cache), expected, identities)
    missing = [s for s in ITERATIONS if s not in found]
    print(f"[budget] image={IMAGE_ID}; cached={list(found)}; new forwards={len(missing)} <= 2; "
          "training updates=0; no full-validation inference", flush=True)
    if missing and args.cache_only:
        raise FileNotFoundError(f"--cache-only: no compatible dense cache for {missing}; NO forward started")
    if missing and args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("Missing-cache forwards need CUDA in the lami interpreter")
    signature = locked_manifest(output / "manifest.json", {
        "schema_version": 1, "sources": identities, "native_capture": expected,
        "analysis_code": {name: file_identity(ROOT / "tools" / name)["sha256"] for name in
                          ("trace_rare_gt_queries.py", "rare_gt_query_ops.py", "rare_region_pairing_ops.py")}})
    cache_signature = locked_manifest(cache / "manifest.json", expected)
    categories = {c["id"]: c for c in dataset["categories"]}
    report = {"complete": False, "fingerprint": signature, "sources": identities,
              "saved_prediction_identities": {s: trace["sources"][s + "_predictions"] for s in ITERATIONS},
              "prediction_verification_scope": "All selected image rows embedded in the hashed source trace; "
              "the full-validation prediction files are not re-read or re-evaluated.",
              "gt": trace["gt"], "image": trace["image"], "protocol": expected,
              "new_forward_calls": 0, "training_updates": 0, "endpoints": {},
              "limits": ["One post-hoc selected GT; not AP or a cross-class training cause.",
                         "Query IDs are endpoint-local; the GT is the geometric anchor.",
                         "Eligible-box loss concerns FINAL pre-top300 queries, not all encoder proposals.",
                         "Scores/ranks do not identify the loss term or historical optimizer update responsible."]}
    save_json(output / "report.json", report)
    for side in ITERATIONS:
        if side not in found:
            torch.manual_seed(42)
            # Checkpoint buffers were validated above and overwrite constructor defaults.
            # This existing capture receives a SINGLE image ID, never a full panel.
            dump_checkpoint(dump_args, side, {"image_ids": [IMAGE_ID]}, dataset, cache_signature, cache)
            report["new_forward_calls"] += 1
            found[side] = (cache / "manifest.json", side)
        sample, bank, provenance = load_cache(*found[side], side, dataset, trace["image"])
        replay = replay_sample(sample, bank, PROTOCOL, device="cpu")
        saved = trace["endpoints"][side]["selected_detections_by_gt_iou"]
        check = verify_predictions(selected_predictions(replay, IMAGE_ID), saved)
        print(f"[verified {side}] all {check['matched']} saved predictions reproduced; "
              f"box_error={check['max_box_error']:.6g}, score_error={check['max_score_error']:.6g}", flush=True)
        analysis = analyze_queries(replay, trace["gt"], categories)
        for key, data in analysis["by_iou"].items():
            saved_coverage = trace["endpoints"][side]["by_iou"][key]
            if (data["retained_true_class_queries"] != saved_coverage["selected_true_class_eligible"]
                    or data["retained_any_class_pairs"] != saved_coverage["selected_any_class_eligible"]):
                raise ValueError("Replayed target coverage differs from saved matching; inspect IoU boundary precision")
        # Keep every query x category score, not only the target class/top-k union.
        dense_path = output / f"{side}_all_query_scores.pt"
        save_tensor_file(dense_path, {"fingerprint": signature, "image_id": IMAGE_ID,
            "iteration": ITERATIONS[side], "category_ids": replay["category_ids"],
            "boxes_xyxy": replay["boxes"], "detector_logits": replay["det_logits"],
            "clip_logits": replay["clip_logits"], "fused_scores": replay["scores"],
            "selected": replay["selected"], "note": "Native-checked dense float32 replay, not counterfactual scoring"})
        report["endpoints"][side] = dict(analysis, iteration=ITERATIONS[side], cache=provenance,
            saved_predictions_check=check, native_replay_check=sample["native_replay_check"],
            dense_scores=file_identity(dense_path))
        save_json(output / "report.json", report)
    report["complete"] = True
    save_json(output / "report.json", report)
    print("\n=== One-GT pre-top300 trace: 8ep -> 12ep ===", flush=True)
    for side, data in report["endpoints"].items():
        print(f"{side} iter={data['iteration']} Q={data['raw_queries']} "
              f"best-IoU={data['best_iou_query']['region_iou']:.4f} cutoff={data['image_cutoff']:.6f}")
        for key, row in data["by_iou"].items():
            print(f"  IoU={key}: {row['reason']}; raw eligible={row['raw_eligible_queries']}; "
                  f"retained correct={row['retained_true_class_queries']}")
            best = row["best_fused_true_class_eligible"]
            if best:
                print(f"    best eligible q={best['query_id']} det={best['detector_probability']:.6f} "
                      f"CLIP={best['clip_probability']:.6f} fused={best['fused_score']:.6f} "
                      f"det/CLIP class ranks={best['detector_rank']}/{best['clip_rank']} "
                      f"image pair rank={best['image_pair_rank_interval']}")
    print(f"[save] {output / 'report.json'}", flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-json", required=True)
    parser.add_argument("--old-checkpoint", required=True, help="No-radius model_0056799.pth")
    parser.add_argument("--new-checkpoint", required=True, help="No-radius model_final.pth, iteration 85199")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-file", default="lami_dino/configs/dino_convnext_large_4scale_12ep_lvis_no_radius.py")
    parser.add_argument("--annotations", help="Defaults to the source trace's annotation file")
    parser.add_argument("--reuse-cache", nargs="*", default=[], help="Explicit dense pairing_cache directories")
    parser.add_argument("--cache-search-root", help="Inspect only */pairing_cache and */cache manifests under this directory")
    parser.add_argument("--cache-only", action="store_true", help="Refuse any missing-cache model forward")
    parser.add_argument("--device", default="cuda:0", help="Only missing-cache forwards use this device; replay is CPU")
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
