#!/usr/bin/env python
"""Replay cached detector/CLIP score fusion and run official LVIS bbox AP.

This script performs no neural-network inference. It consumes the compact
candidate cache produced by ``dump_ovd_raw_scores.py``. Every fusion profile in
the dump manifest is exact for its top-300 detections.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lami_dino.diagnostic_ops import fuse_sparse_detector_vlm_scores  # noqa: E402


METRICS = ("AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf")


def load_tensor_file(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalized_boxes_to_lvis(boxes, width, height):
    cx, cy, box_width, box_height = boxes.unbind(-1)
    x1 = (cx - 0.5 * box_width) * width
    y1 = (cy - 0.5 * box_height) * height
    x2 = (cx + 0.5 * box_width) * width
    y2 = (cy + 0.5 * box_height) * height
    x1.clamp_(0, width)
    x2.clamp_(0, width)
    y1.clamp_(0, height)
    y2.clamp_(0, height)
    return torch.stack((x1, y1, (x2 - x1).clamp_min(0), (y2 - y1).clamp_min(0)), dim=-1)


def predictions_for_profile(files, profile, novel_mask, max_dets):
    predictions = []
    for index, path in enumerate(files):
        payload = load_tensor_file(path)
        if payload.get("schema_version") != 1:
            raise ValueError(f"unsupported raw dump schema in {path}")
        query_ids = payload["candidate_query_ids"].long()
        class_ids = payload["candidate_class_ids"].long()
        detector_source = profile.get("detector_source", "logmeanexp")
        if "detector_logits_by_source" in payload:
            logits_by_source = payload["detector_logits_by_source"]
            if detector_source not in logits_by_source:
                raise ValueError(
                    f"detector source {detector_source!r} is missing from {path}"
                )
            detector_logits = logits_by_source[detector_source]
        elif detector_source == "logmeanexp" and "detector_logits" in payload:
            # Backward compatibility for schema-1 caches made before the
            # one-vector prototype controls were added.
            detector_logits = payload["detector_logits"]
        else:
            raise ValueError(
                f"dump {path} cannot evaluate detector source {detector_source!r}"
            )
        log_scores = fuse_sparse_detector_vlm_scores(
            detector_logits,
            payload["vlm_logits"],
            payload["vlm_log_normalizer"],
            query_ids,
            class_ids,
            novel_mask,
            fusion=profile["fusion"],
            base_weight=float(profile["base_weight"]),
            novel_weight=float(profile["novel_weight"]),
            novel_scale=float(profile["novel_scale"]),
            detector_temperature=float(profile.get("detector_temperature", 1.0)),
            vlm_temperature=float(profile.get("vlm_temperature", 1.0)),
        )
        count = min(max_dets, log_scores.numel())
        selected = log_scores.topk(count).indices
        selected_queries = query_ids[selected]
        selected_classes = class_ids[selected]
        selected_scores = log_scores[selected].exp()
        selected_boxes = normalized_boxes_to_lvis(
            payload["query_boxes"][selected_queries].float(),
            int(payload["width"]),
            int(payload["height"]),
        )
        valid = (selected_boxes[:, 2] > 0) & (selected_boxes[:, 3] > 0)
        image_id = int(payload["image_id"])
        for box, score, class_id in zip(
            selected_boxes[valid].tolist(),
            selected_scores[valid].tolist(),
            selected_classes[valid].tolist(),
        ):
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": int(class_id) + 1,
                    "bbox": box,
                    "score": float(score),
                }
            )
        if (index + 1) % 500 == 0:
            print(f"  loaded {index + 1}/{len(files)} images", flush=True)
    return predictions


def evaluate_lvis(lvis_gt, predictions, max_dets):
    from lvis import LVISEval, LVISResults

    lvis_results = LVISResults(lvis_gt, predictions, max_dets=max_dets)
    evaluator = LVISEval(lvis_gt, lvis_results, "bbox")
    evaluator.run()
    evaluator.print_results()
    raw = evaluator.get_results()
    return {metric: 100.0 * float(raw[metric]) for metric in METRICS}


def print_summary(results):
    print("\n=== Fusion summary ===")
    print(f"{'profile':>24} " + " ".join(f"{metric:>8}" for metric in METRICS))
    for name, row in results.items():
        print(
            f"{name:>24} "
            + " ".join(f"{row[metric]:8.4f}" for metric in METRICS)
        )

    baseline = results.get("current_power")
    if baseline is not None and len(results) > 1:
        print("\n=== Delta from current_power (positive is better) ===")
        print(f"{'profile':>24} {'delta_AP':>10} {'delta_APr':>10}")
        for name, row in results.items():
            if name == "current_power":
                continue
            print(
                f"{name:>24} "
                f"{row['AP'] - baseline['AP']:+10.4f} "
                f"{row['APr'] - baseline['APr']:+10.4f}"
            )


def save_metric_report(path, results):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[save] {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument(
        "--profiles",
        nargs="*",
        default=None,
        help="profile names from manifest.json; default evaluates all",
    )
    parser.add_argument("--max-dets", type=int, default=300)
    parser.add_argument("--output", default=None, help="small JSON metric report")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse profiles already present in --output and append new results",
    )
    parser.add_argument(
        "--save-predictions-dir",
        default=None,
        help="optional; writes very large per-profile LVIS JSON files",
    )
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    manifest = json.loads((dump_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported manifest schema")
    if args.max_dets > int(manifest["topk_per_profile"]):
        raise ValueError(
            f"--max-dets={args.max_dets} exceeds the exact cached top-k "
            f"{manifest['topk_per_profile']}"
        )

    files = sorted(dump_dir.glob("raw_*.pth"))
    expected = int(manifest["num_dataset_images"])
    if len(files) != expected:
        raise RuntimeError(
            f"incomplete dump: found {len(files)} raw files, expected {expected}; "
            "resume dump_ovd_raw_scores.py before evaluating"
        )

    profiles_by_name = {profile["name"]: profile for profile in manifest["profiles"]}
    selected_names = args.profiles or list(profiles_by_name)
    missing = set(selected_names) - set(profiles_by_name)
    if missing:
        raise ValueError(
            f"unknown profiles {sorted(missing)}; available={list(profiles_by_name)}"
        )

    num_classes = max(manifest["novel_class_ids"]) + 1
    # LVIS has contiguous classifier indices [0, 1202]. Infer the full count
    # from the annotation categories in case the highest class is not novel.
    from lvis import LVIS

    lvis_gt = LVIS(manifest["lvis_json"])
    num_classes = max(num_classes, len(lvis_gt.dataset["categories"]))
    novel_mask = torch.zeros(num_classes, dtype=torch.bool)
    novel_mask[torch.tensor(manifest["novel_class_ids"], dtype=torch.long)] = True

    results = {}
    if args.resume:
        if not args.output:
            raise ValueError("--resume requires --output")
        output_path = Path(args.output)
        if output_path.exists():
            results = json.loads(output_path.read_text(encoding="utf-8"))
    for name in selected_names:
        if args.resume and name in results:
            print(f"[skip] {name} already exists in {args.output}")
            continue
        profile = profiles_by_name[name]
        print(f"\n[evaluate] {name}: {profile}")
        predictions = predictions_for_profile(
            files, profile, novel_mask, args.max_dets
        )
        if args.save_predictions_dir:
            prediction_dir = Path(args.save_predictions_dir)
            prediction_dir.mkdir(parents=True, exist_ok=True)
            prediction_path = prediction_dir / f"{name}.json"
            prediction_path.write_text(json.dumps(predictions), encoding="utf-8")
            print(f"[save] {prediction_path}")
        results[name] = evaluate_lvis(lvis_gt, predictions, args.max_dets)
        del predictions
        gc.collect()
        if args.output:
            save_metric_report(args.output, results)

    print_summary(results)


if __name__ == "__main__":
    main()
