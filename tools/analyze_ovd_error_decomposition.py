#!/usr/bin/env python
"""Decompose OVD recall loss using a compact raw-score cache.

For each LVIS GT instance this script measures:

1. whether any class-agnostic decoder query reaches an IoU threshold;
2. whether the correct class paired with the best-IoU query survives top-k;
3. whether any correct-class query reaching the threshold survives top-k;
4. which rare false-positive type occupies the selected top-k.

The first-to-third gap localizes recall loss before versus after proposal
generation.  Rare precision errors are matched over the complete validation
set using LVIS positive/negative/not-exhaustive image annotations.  This is a
diagnostic decomposition, not a replacement for official LVIS AP.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
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
    return query_ids[selected], class_ids[selected], scores[selected]


def pairwise_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0).prod(dim=-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0).prod(dim=-1)
    left_top = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (right_bottom - left_top).clamp_min(0).prod(dim=-1)
    return intersection / (
        area1[:, None] + area2[None, :] - intersection
    ).clamp_min(1e-12)


def classify_rare_detections(
    detection_boxes,
    detection_scores,
    detection_classes,
    gt_boxes,
    gt_classes,
    *,
    negative_classes,
    not_exhaustive_classes,
    iou_threshold,
    background_iou,
):
    """Greedily classify score-ordered rare detections into TP/FP reasons.

    Category ids are zero based.  Unmatched detections for categories that LVIS
    does not exhaustively verify in an image are marked ignored rather than
    counted as false positives.
    """
    if not 0.0 <= background_iou < iou_threshold <= 1.0:
        raise ValueError("require 0 <= background_iou < iou_threshold <= 1")
    if (
        detection_boxes.shape[0] != detection_scores.numel()
        or detection_scores.numel() != detection_classes.numel()
    ):
        raise ValueError("detection tensors must contain the same number of rows")
    if gt_boxes.shape[0] != gt_classes.numel():
        raise ValueError("GT tensors must contain the same number of rows")

    order = detection_scores.argsort(descending=True)
    overlaps = pairwise_iou(detection_boxes, gt_boxes)
    matched_gt = set()
    positive_classes = set(gt_classes.tolist())
    results = []
    for detection_index in order.tolist():
        category = int(detection_classes[detection_index])
        score = float(detection_scores[detection_index])
        row = overlaps[detection_index]
        same_indices = torch.nonzero(
            gt_classes == category, as_tuple=False
        ).flatten()
        same_ious = row[same_indices] if same_indices.numel() else row.new_empty((0,))

        available = [
            int(gt_index)
            for gt_index, iou in zip(same_indices.tolist(), same_ious.tolist())
            if iou >= iou_threshold and int(gt_index) not in matched_gt
        ]
        if available:
            best_match = max(available, key=lambda index: float(row[index]))
            matched_gt.add(best_match)
            outcome = "tp"
        elif category in not_exhaustive_classes:
            outcome = "ignored_not_exhaustive"
        elif category not in positive_classes and category not in negative_classes:
            outcome = "ignored_unknown"
        elif same_ious.numel() and float(same_ious.max()) >= iou_threshold:
            outcome = "duplicate"
        else:
            different = row[gt_classes != category]
            if different.numel() and float(different.max()) >= iou_threshold:
                outcome = "classification"
            elif row.numel() and float(row.max()) >= background_iou:
                outcome = "localization"
            else:
                outcome = "background"
        # Keep millions of full-validation records compact in memory.
        results.append((category, score, outcome))
    return results


def _summarize_outcomes(records):
    counts = Counter(record[2] for record in records)
    evaluated = sum(
        count for outcome, count in counts.items() if not outcome.startswith("ignored_")
    )
    true_positives = counts["tp"]
    false_positives = evaluated - true_positives
    fp_types = {
        key: int(counts[key])
        for key in ("duplicate", "classification", "localization", "background")
    }
    return {
        "detections": len(records),
        "evaluated": evaluated,
        "true_positives": int(true_positives),
        "false_positives": int(false_positives),
        "diagnostic_precision": safe_ratio(true_positives, evaluated),
        "false_positive_types": fp_types,
        "false_positive_fractions": {
            key: safe_ratio(value, false_positives) for key, value in fp_types.items()
        },
        "ignored_not_exhaustive": int(counts["ignored_not_exhaustive"]),
        "ignored_unknown": int(counts["ignored_unknown"]),
    }


def summarize_rare_false_positives(records):
    """Summarize all selected predictions and the AP-relevant score prefix.

    For each category, predictions below its last true positive cannot create a
    later recall increase.  The per-category prefix through the last TP is
    therefore more informative than counting every low-score tail prediction.
    Categories without a TP retain their complete evaluated list.
    """
    by_class = defaultdict(list)
    for record in records:
        by_class[int(record[0])].append(record)
    relevant = []
    categories_without_tp = 0
    for class_records in by_class.values():
        ordered = sorted(class_records, key=lambda item: item[1], reverse=True)
        evaluated_indices = [
            index
            for index, item in enumerate(ordered)
            if not item[2].startswith("ignored_")
        ]
        true_positive_indices = [
            index for index in evaluated_indices if ordered[index][2] == "tp"
        ]
        if true_positive_indices:
            relevant.extend(ordered[: max(true_positive_indices) + 1])
        else:
            categories_without_tp += 1
            relevant.extend(ordered)
    return {
        "all_selected": _summarize_outcomes(records),
        "through_last_true_positive_per_category": _summarize_outcomes(relevant),
        "categories_with_predictions": len(by_class),
        "categories_without_true_positive": categories_without_tp,
    }


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
        "\nRecall scope: coverage upper bounds only; the compact cache cannot "
        "recover exact 1203-way ranks below its candidate pool."
    )


def print_false_positive_report(report, iou_threshold, background_iou):
    print(f"\n=== Rare top-300 precision-side decomposition @IoU={iou_threshold:.2f} ===")
    print(
        "LVIS unknown/not-exhaustive detections are ignored. 'localization' "
        f"means max GT IoU is in [{background_iou:.2f}, {iou_threshold:.2f})."
    )
    print(
        f"{'scope':>38} {'eval':>9} {'TP':>8} {'FP':>9} {'prec%':>8} "
        f"{'dup%':>8} {'cls%':>8} {'loc%':>8} {'bg%':>8}"
    )
    for label, key in (
        ("all selected rare detections", "all_selected"),
        ("per-class prefix through last TP", "through_last_true_positive_per_category"),
    ):
        row = report[key]
        fractions = row["false_positive_fractions"]
        print(
            f"{label:>38} {row['evaluated']:9d} {row['true_positives']:8d} "
            f"{row['false_positives']:9d} "
            f"{100.0 * (row['diagnostic_precision'] or 0.0):8.2f} "
            f"{100.0 * (fractions['duplicate'] or 0.0):8.2f} "
            f"{100.0 * (fractions['classification'] or 0.0):8.2f} "
            f"{100.0 * (fractions['localization'] or 0.0):8.2f} "
            f"{100.0 * (fractions['background'] or 0.0):8.2f}"
        )
    all_selected = report["all_selected"]
    print(f"ignored_not_exhaustive: {all_selected['ignored_not_exhaustive']}")
    print(f"ignored_unknown: {all_selected['ignored_unknown']}")
    print(f"categories_with_predictions: {report['categories_with_predictions']}")
    print(
        "categories_without_true_positive: "
        f"{report['categories_without_true_positive']}"
    )
    print(
        "The prefix table is diagnostic, not official AP: it identifies which "
        "high-ranked FP types precede useful recall."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--profile", default="current_power")
    parser.add_argument("--max-dets", type=int, default=300)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.5, 0.75])
    parser.add_argument("--fp-iou-threshold", type=float, default=0.5)
    parser.add_argument("--fp-background-iou", type=float, default=0.1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    thresholds = sorted(set(float(value) for value in args.iou_thresholds))
    if any(value <= 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("IoU thresholds must be within (0,1]")
    if not 0.0 <= args.fp_background_iou < args.fp_iou_threshold <= 1.0:
        raise ValueError(
            "require 0 <= --fp-background-iou < --fp-iou-threshold <= 1"
        )
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
    images_by_id = {
        int(image["id"]): image for image in annotation_data["images"]
    }

    num_classes = max(max(categories), max(manifest["novel_class_ids"]) + 1)
    novel_mask = torch.zeros(num_classes, dtype=torch.bool)
    novel_mask[torch.tensor(manifest["novel_class_ids"], dtype=torch.long)] = True
    accumulator = make_accumulator(thresholds)
    rare_detection_records = []

    for index, path in enumerate(files):
        payload = load_tensor_file(path)
        image_id = int(payload["image_id"])
        annotations = annotations_by_image.get(image_id, [])
        query_boxes = normalized_cxcywh_to_xyxy(
            payload["query_boxes"],
            int(payload["width"]),
            int(payload["height"]),
        )
        (
            selected_query_ids,
            selected_class_ids,
            selected_log_scores,
        ) = selected_pairs(payload, profile, novel_mask, args.max_dets, path)
        if annotations:
            gt_boxes = xywh_to_xyxy(
                torch.tensor([item["bbox"] for item in annotations], dtype=torch.float32)
            )
            gt_classes = torch.tensor(
                [int(item["category_id"]) - 1 for item in annotations],
                dtype=torch.long,
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

        rare_prediction_mask = torch.tensor(
            [frequencies[int(class_id) + 1] == "r" for class_id in selected_class_ids],
            dtype=torch.bool,
        )
        if rare_prediction_mask.any():
            rare_query_ids = selected_query_ids[rare_prediction_mask]
            rare_classes = selected_class_ids[rare_prediction_mask]
            rare_boxes = query_boxes[rare_query_ids]
            rare_scores = selected_log_scores[rare_prediction_mask].exp()
            if annotations:
                fp_gt_boxes = gt_boxes
                fp_gt_classes = gt_classes
            else:
                fp_gt_boxes = query_boxes.new_empty((0, 4))
                fp_gt_classes = selected_class_ids.new_empty((0,))
            image_metadata = images_by_id[image_id]
            rare_detection_records.extend(
                classify_rare_detections(
                    rare_boxes,
                    rare_scores,
                    rare_classes,
                    fp_gt_boxes,
                    fp_gt_classes,
                    negative_classes={
                        int(value) - 1
                        for value in image_metadata.get("neg_category_ids", [])
                    },
                    not_exhaustive_classes={
                        int(value) - 1
                        for value in image_metadata.get(
                            "not_exhaustive_category_ids", []
                        )
                    },
                    iou_threshold=args.fp_iou_threshold,
                    background_iou=args.fp_background_iou,
                )
            )
        if (index + 1) % 500 == 0:
            print(f"loaded {index + 1}/{len(files)} images", flush=True)

    report = finalize(accumulator, thresholds)
    print_report(report, thresholds, args.profile, args.max_dets)
    rare_false_positive_report = summarize_rare_false_positives(
        rare_detection_records
    )
    print_false_positive_report(
        rare_false_positive_report,
        args.fp_iou_threshold,
        args.fp_background_iou,
    )
    payload = {
        "dump_dir": str(dump_dir),
        "profile": args.profile,
        "max_dets": args.max_dets,
        "report": report,
        "rare_false_positive_decomposition": rare_false_positive_report,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[save] {output_path}")


if __name__ == "__main__":
    main()
