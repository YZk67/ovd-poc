#!/usr/bin/env python
"""Decompose OVD recall loss using a compact raw-score cache.

For each LVIS GT instance this script measures:

1. whether any class-agnostic decoder query reaches an IoU threshold;
2. whether the correct class paired with the best-IoU query survives top-k;
3. whether any correct-class query reaching the threshold survives top-k.

The first-to-third gap localizes recall loss before versus after proposal
generation. Because the compact cache does not contain all Q x C logits, this
cannot recover the exact 1203-way rank of a true-class pair below top-k and it
does not diagnose false-positive precision errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lami_dino.diagnostic_ops import (  # noqa: E402
    detection_stage_hits,
    fuse_sparse_detector_vlm_scores,
)


SPLITS = ("all", "r", "c", "f")


def load_tensor_file(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalized_cxcywh_to_xyxy(boxes, width, height):
    cx, cy, box_width, box_height = boxes.float().unbind(-1)
    x1 = ((cx - 0.5 * box_width) * width).clamp(0, width)
    y1 = ((cy - 0.5 * box_height) * height).clamp(0, height)
    x2 = ((cx + 0.5 * box_width) * width).clamp(0, width)
    y2 = ((cy + 0.5 * box_height) * height).clamp(0, height)
    return torch.stack((x1, y1, x2, y2), dim=-1)


def xywh_to_xyxy(boxes):
    result = boxes.float().clone()
    result[:, 2] = boxes[:, 0] + boxes[:, 2]
    result[:, 3] = boxes[:, 1] + boxes[:, 3]
    return result


def category_frequency(category):
    frequency = category.get("frequency")
    if frequency in {"r", "c", "f"}:
        return frequency
    image_count = category.get("image_count")
    if image_count is None:
        raise ValueError(
            f"category {category.get('id')} lacks frequency and image_count"
        )
    return "r" if image_count <= 10 else "c" if image_count <= 100 else "f"


def detector_logits_for_source(payload, source, path):
    if "detector_logits_by_source" in payload:
        sources = payload["detector_logits_by_source"]
        if source not in sources:
            raise ValueError(f"detector source {source!r} is missing from {path}")
        return sources[source]
    if source == "logmeanexp" and "detector_logits" in payload:
        return payload["detector_logits"]
    raise ValueError(f"dump {path} cannot provide detector source {source!r}")


def selected_pairs(payload, profile, novel_mask, max_dets, path):
    query_ids = payload["candidate_query_ids"].long()
    class_ids = payload["candidate_class_ids"].long()
    detector_logits = detector_logits_for_source(
        payload, profile.get("detector_source", "logmeanexp"), path
    )
    scores = fuse_sparse_detector_vlm_scores(
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
    count = min(int(max_dets), scores.numel())
    selected = scores.topk(count).indices
    return query_ids[selected], class_ids[selected]


def make_accumulator(thresholds):
    return {
        split: {
            "gt_count": 0,
            "best_ious": [],
            "thresholds": {
                threshold: defaultdict(int) for threshold in thresholds
            },
        }
        for split in SPLITS
    }


def update_accumulator(accumulator, frequencies, best_iou, stages):
    for index, frequency in enumerate(frequencies):
        for split in ("all", frequency):
            row = accumulator[split]
            row["gt_count"] += 1
            row["best_ious"].append(float(best_iou[index]))
            for threshold, hits in stages.items():
                counters = row["thresholds"][threshold]
                for key, values in hits.items():
                    counters[key] += int(values[index])


def safe_ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else None


def finalize(accumulator, thresholds):
    report = {}
    for split in SPLITS:
        source = accumulator[split]
        count = int(source["gt_count"])
        best_ious = torch.tensor(source["best_ious"], dtype=torch.float32)
        split_report = {
            "gt_count": count,
            "mean_best_iou": float(best_ious.mean()) if count else None,
            "median_best_iou": float(best_ious.median()) if count else None,
            "thresholds": {},
        }
        for threshold in thresholds:
            counters = source["thresholds"][threshold]
            proposal = int(counters["proposal"])
            best_pair = int(counters["best_query_pair"])
            class_aware = int(counters["class_aware_topk"])
            split_report["thresholds"][f"{threshold:.2f}"] = {
                "proposal_hits": proposal,
                "best_query_pair_hits": best_pair,
                "class_aware_topk_hits": class_aware,
                "proposal_recall": safe_ratio(proposal, count),
                "best_query_pair_recall": safe_ratio(best_pair, count),
                "class_aware_topk_recall": safe_ratio(class_aware, count),
                "conditional_topk_retention": safe_ratio(class_aware, proposal),
                "localization_miss_rate": safe_ratio(count - proposal, count),
                "post_proposal_miss_rate": safe_ratio(proposal - class_aware, count),
            }
        report[split] = split_report
    return report


def percentage(value):
    return "-" if value is None else f"{100.0 * value:.2f}"


def print_report(report, thresholds, profile, max_dets):
    print("\n=== Recall-side error decomposition ===")
    print(f"profile={profile}, topk={max_dets}")
    print(
        f"{'split':>6} {'IoU':>5} {'GT':>8} {'proposal%':>10} "
        f"{'best-pair%':>11} {'correct-topk%':>14} {'cond-keep%':>11} "
        f"{'loc-miss%':>10} {'post-miss%':>11}"
    )
    for split in SPLITS:
        row = report[split]
        for threshold in thresholds:
            values = row["thresholds"][f"{threshold:.2f}"]
            print(
                f"{split:>6} {threshold:5.2f} {row['gt_count']:8d} "
                f"{percentage(values['proposal_recall']):>10} "
                f"{percentage(values['best_query_pair_recall']):>11} "
                f"{percentage(values['class_aware_topk_recall']):>14} "
                f"{percentage(values['conditional_topk_retention']):>11} "
                f"{percentage(values['localization_miss_rate']):>10} "
                f"{percentage(values['post_proposal_miss_rate']):>11}"
            )

    rare = report["r"]["thresholds"].get("0.50")
    if rare is not None:
        localization = rare["localization_miss_rate"]
        post_proposal = rare["post_proposal_miss_rate"]
        print("\n=== Rare recall-side verdict at IoU=0.50 ===")
        if localization is None or post_proposal is None:
            print("verdict: INSUFFICIENT_RARE_GT")
        elif post_proposal > 1.25 * localization:
            print("verdict: CLASSIFICATION_OR_TOPK_RANKING_DOMINANT")
        elif localization > 1.25 * post_proposal:
            print("verdict: LOCALIZATION_OR_PROPOSAL_DOMINANT")
        else:
            print("verdict: MIXED_RECALL_BOTTLENECK")
        print(f"rare_localization_miss_rate: {localization}")
        print(f"rare_post_proposal_miss_rate: {post_proposal}")
    print(
        "\nScope: coverage upper bounds only. This report does not measure false-positive "
        "precision errors or exact 1203-way ranks below the cached top-k."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--profile", default="current_power")
    parser.add_argument("--max-dets", type=int, default=300)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.5, 0.75])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    thresholds = sorted(set(float(value) for value in args.iou_thresholds))
    if any(value <= 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("IoU thresholds must be within (0,1]")
    dump_dir = Path(args.dump_dir)
    manifest = json.loads((dump_dir / "manifest.json").read_text(encoding="utf-8"))
    if args.max_dets > int(manifest["topk_per_profile"]):
        raise ValueError(
            f"--max-dets={args.max_dets} exceeds cached exact top-k "
            f"{manifest['topk_per_profile']}"
        )
    profiles = {profile["name"]: profile for profile in manifest["profiles"]}
    if args.profile not in profiles:
        raise ValueError(
            f"unknown profile {args.profile!r}; available={sorted(profiles)}"
        )
    profile = profiles[args.profile]

    files = sorted(dump_dir.glob("raw_*.pth"))
    expected = int(manifest["num_dataset_images"])
    if len(files) != expected:
        raise RuntimeError(f"incomplete dump: found {len(files)}, expected {expected}")

    annotation_data = json.loads(Path(manifest["lvis_json"]).read_text(encoding="utf-8"))
    categories = {int(item["id"]): item for item in annotation_data["categories"]}
    frequencies = {
        category_id: category_frequency(category)
        for category_id, category in categories.items()
    }
    annotations_by_image = defaultdict(list)
    for annotation in annotation_data["annotations"]:
        if annotation.get("iscrowd", 0):
            continue
        bbox = annotation["bbox"]
        if bbox[2] <= 0 or bbox[3] <= 0:
            continue
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    num_classes = max(max(categories), max(manifest["novel_class_ids"]) + 1)
    novel_mask = torch.zeros(num_classes, dtype=torch.bool)
    novel_mask[torch.tensor(manifest["novel_class_ids"], dtype=torch.long)] = True
    accumulator = make_accumulator(thresholds)

    for index, path in enumerate(files):
        payload = load_tensor_file(path)
        image_id = int(payload["image_id"])
        annotations = annotations_by_image.get(image_id, [])
        if annotations:
            gt_boxes = xywh_to_xyxy(
                torch.tensor([item["bbox"] for item in annotations], dtype=torch.float32)
            )
            gt_classes = torch.tensor(
                [int(item["category_id"]) - 1 for item in annotations],
                dtype=torch.long,
            )
            query_boxes = normalized_cxcywh_to_xyxy(
                payload["query_boxes"],
                int(payload["width"]),
                int(payload["height"]),
            )
            selected_query_ids, selected_class_ids = selected_pairs(
                payload, profile, novel_mask, args.max_dets, path
            )
            best_iou, stages = detection_stage_hits(
                query_boxes,
                gt_boxes,
                gt_classes,
                selected_query_ids,
                selected_class_ids,
                thresholds=thresholds,
            )
            update_accumulator(
                accumulator,
                [frequencies[int(item["category_id"])] for item in annotations],
                best_iou,
                stages,
            )
        if (index + 1) % 500 == 0:
            print(f"loaded {index + 1}/{len(files)} images", flush=True)

    report = finalize(accumulator, thresholds)
    print_report(report, thresholds, args.profile, args.max_dets)
    payload = {
        "dump_dir": str(dump_dir),
        "profile": args.profile,
        "max_dets": args.max_dets,
        "report": report,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[save] {output_path}")


if __name__ == "__main__":
    main()
