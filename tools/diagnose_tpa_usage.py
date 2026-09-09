#!/usr/bin/env python
"""Diagnose whether non-collapsed TPA modes improve final-query recognition.

The diagnostic matches final decoder boxes to LVIS ground-truth boxes and then
compares the true-class ranking of three detector-side text representations:

* ``logmeanexp``: the current K-prototype Eq. (2) classifier;
* ``prototype_mean``: the same learned prototypes averaged into one vector;
* ``prompt_mean``: the original prompt bank averaged into one vector.

It additionally measures the exact 1203-way category rank after the current
detector/CLIP fusion and whether that best-query/category pair survives the
image-level top-k selection.

It also measures posterior sharpness and slot usage for the true class. This
separates "the prototype vectors have high rank" from "visual instances
actually select and benefit from different prototype modes".
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lami_dino.diagnostic_ops import (  # noqa: E402
    fuse_detector_vlm_scores,
    prototype_variant_logits,
    true_class_mode_weights,
)


def install_capture_hooks(model, capture):
    """Capture the final decoder classifier inputs and normalized query boxes."""
    final_index = model.transformer.decoder.num_layers - 1
    classifier = model.class_embed[final_index]
    original_logits = classifier._compute_tpa_logits

    def capture_logits(x, *, content_inds, additional_class):
        result = original_logits(
            x,
            content_inds=content_inds,
            additional_class=additional_class,
        )
        prototypes = classifier._external_prototypes
        if prototypes is None:
            prototypes = classifier._cached_eval
        if prototypes is not None:
            # _compute_tpa_logits receives the output of classifier.linear;
            # applying the projection a second time would be dimensionally
            # invalid (and was the main flaw in the old diagnostic hook).
            capture["projected_features"] = x.detach()
            capture["detector_logits"] = result.detach()
            capture["prototypes"] = prototypes.detach()
            capture["prompt_features"] = classifier.eval_text_feats.detach()
        return result

    classifier._compute_tpa_logits = capture_logits

    original_extract = model.extract_region_feature

    def capture_region_feature(features, bbox, layer_name):
        result = original_extract(features, bbox, layer_name)
        if layer_name == "p3":
            capture["roi_features"] = result.detach()
        return result

    model.extract_region_feature = capture_region_feature

    original_inference = model.inference

    def capture_inference(box_cls, box_pred, image_sizes, wo_sigmoid=False):
        capture["query_boxes"] = box_pred.detach()
        capture["image_sizes"] = tuple(image_sizes)
        capture["final_scores"] = (
            box_cls.detach() if wo_sigmoid else box_cls.detach().sigmoid()
        )
        return original_inference(
            box_cls,
            box_pred,
            image_sizes,
            wo_sigmoid=wo_sigmoid,
        )

    model.inference = capture_inference
    return classifier


def box_cxcywh_to_xyxy(boxes):
    cx, cy, width, height = boxes.unbind(-1)
    return torch.stack(
        (cx - 0.5 * width, cy - 0.5 * height, cx + 0.5 * width, cy + 0.5 * height),
        dim=-1,
    )


def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0).prod(dim=-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0).prod(dim=-1)
    left_top = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (right_bottom - left_top).clamp_min(0).prod(dim=-1)
    return intersection / (area1[:, None] + area2[None, :] - intersection).clamp_min(1e-12)


def update_ranking_stats(accumulator, logits, class_ids, frequencies):
    true_logits = logits.gather(1, class_ids[:, None])
    ranks = 1 + (logits > true_logits).sum(dim=-1)
    for rank, frequency in zip(ranks.tolist(), frequencies):
        for split in ("all", frequency):
            row = accumulator[split]
            row["count"] += 1
            row["top1"] += int(rank <= 1)
            row["top5"] += int(rank <= 5)
            row["top10"] += int(rank <= 10)
            row["top20"] += int(rank <= 20)
            row["reciprocal_rank"] += 1.0 / float(rank)
            row["rank_sum"] += int(rank)
    return ranks


def finalize_ranking_stats(accumulator):
    result = {}
    for split in ("all", "r", "c", "f"):
        row = accumulator[split]
        count = int(row["count"])
        if count == 0:
            result[split] = {"count": 0}
            continue
        result[split] = {
            "count": count,
            "top1": row["top1"] / count,
            "top5": row["top5"] / count,
            "top10": row["top10"] / count,
            "top20": row["top20"] / count,
            "mrr": row["reciprocal_rank"] / count,
            "mean_rank": row["rank_sum"] / count,
        }
    return result


def normalized_entropy(probabilities):
    probabilities = probabilities.float()
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    return entropy / math.log(probabilities.shape[-1])


def format_percent(value):
    return "-" if value is None else f"{100.0 * value:6.2f}"


def print_ranking_table(results):
    print("\n=== Oracle-query category recognition ===")
    print("The best-IoU query for each GT isolates text classification from proposal recall.\n")
    print(
        f"{'variant':>18} {'split':>6} {'N':>7} "
        f"{'top1':>8} {'top5':>8} {'top10':>8} {'top20':>8} "
        f"{'MRR':>8} {'mean_rank':>11}"
    )
    for variant, splits in results.items():
        for split in ("all", "r", "c", "f"):
            row = splits[split]
            if not row.get("count"):
                print(
                    f"{variant:>18} {split:>6} {0:7d} "
                    f"{'-':>8} {'-':>8} {'-':>8} {'-':>8} {'-':>8} {'-':>11}"
                )
                continue
            print(
                f"{variant:>18} {split:>6} {row['count']:7d} "
                f"{format_percent(row['top1']):>8} "
                f"{format_percent(row['top5']):>8} "
                f"{format_percent(row['top10']):>8} "
                f"{format_percent(row['top20']):>8} "
                f"{row['mrr']:8.4f} {row['mean_rank']:11.2f}"
            )


def frequency_lookup(metadata, num_classes):
    counts = [None] * num_classes
    for item in metadata.class_image_count:
        category_index = int(item["id"]) - 1
        if 0 <= category_index < num_classes:
            counts[category_index] = int(item["image_count"])
    if any(value is None for value in counts):
        raise ValueError("LVIS class_image_count does not cover every classifier category")
    return ["r" if value <= 10 else "c" if value <= 100 else "f" for value in counts]


def select_dataset_records(records, num_images, seed, sampling, frequencies):
    if sampling == "rare":
        records = [
            record
            for record in records
            if any(
                frequencies[annotation["category_id"]] == "r"
                for annotation in record.get("annotations", [])
            )
        ]
    if num_images <= 0 or num_images >= len(records):
        return records
    generator = np.random.default_rng(seed)
    indices = np.sort(generator.choice(len(records), size=num_images, replace=False))
    return [records[int(index)] for index in indices]


def make_selection_accumulator():
    return {
        split: {
            "count": 0,
            "ranks": [],
            "global_topk": 0,
            "rank_le_cutoff_global_hit": 0,
            "rank_le_cutoff_global_miss": 0,
            "rank_gt_cutoff_global_hit": 0,
            "rank_gt_cutoff_global_miss": 0,
        }
        for split in ("all", "r", "c", "f")
    }


def update_selection_accumulator(accumulator, ranks, survives, frequencies, cutoff):
    for rank, survived, frequency in zip(
        ranks.tolist(), survives.tolist(), frequencies
    ):
        for split in ("all", frequency):
            row = accumulator[split]
            row["count"] += 1
            row["ranks"].append(int(rank))
            row["global_topk"] += int(survived)
            if rank <= cutoff:
                key = (
                    "rank_le_cutoff_global_hit"
                    if survived
                    else "rank_le_cutoff_global_miss"
                )
            else:
                key = (
                    "rank_gt_cutoff_global_hit"
                    if survived
                    else "rank_gt_cutoff_global_miss"
                )
            row[key] += 1


def finalize_selection_accumulator(accumulator, cutoff):
    result = {}
    for split, row in accumulator.items():
        count = int(row["count"])
        if count == 0:
            result[split] = {"count": 0}
            continue
        ranks = np.asarray(row["ranks"], dtype=np.float64)
        missed = count - int(row["global_topk"])
        low_rank_miss = int(row["rank_gt_cutoff_global_miss"])
        high_rank_miss = int(row["rank_le_cutoff_global_miss"])
        result[split] = {
            "count": count,
            "category_rank_cutoff": cutoff,
            "median_category_rank": float(np.median(ranks)),
            "p90_category_rank": float(np.percentile(ranks, 90)),
            "global_topk_recall": row["global_topk"] / count,
            "global_missed": missed,
            "rank_le_cutoff_global_hit": int(row["rank_le_cutoff_global_hit"]),
            "rank_le_cutoff_global_miss": high_rank_miss,
            "rank_gt_cutoff_global_hit": int(row["rank_gt_cutoff_global_hit"]),
            "rank_gt_cutoff_global_miss": low_rank_miss,
            "miss_due_to_category_rank_fraction": (
                low_rank_miss / missed if missed else 0.0
            ),
            "miss_despite_rank_cutoff_fraction": (
                high_rank_miss / missed if missed else 0.0
            ),
        }
    return result


def print_selection_report(report, topk, cutoff):
    print("\n=== Best-query fused category rank vs image-level selection ===")
    print(
        f"Only GTs with best-query IoU >= threshold are included; image top-k={topk}."
    )
    print(
        f"{'split':>6} {'N':>7} {'median-rank':>12} {'p90-rank':>10} "
        f"{'pair-topk%':>11} {'miss-rank>k%':>13} {'miss-rank<=k%':>14}"
    )
    for split in ("all", "r", "c", "f"):
        row = report[split]
        if not row.get("count"):
            continue
        print(
            f"{split:>6} {row['count']:7d} {row['median_category_rank']:12.1f} "
            f"{row['p90_category_rank']:10.1f} "
            f"{100.0 * row['global_topk_recall']:11.2f} "
            f"{100.0 * row['miss_due_to_category_rank_fraction']:13.2f} "
            f"{100.0 * row['miss_despite_rank_cutoff_fraction']:14.2f}"
        )
    rare = report.get("r", {})
    print("\n=== Rare best-query bottleneck verdict ===")
    if not rare.get("count") or not rare.get("global_missed"):
        print("verdict: INSUFFICIENT_RARE_MISSES")
        return "INSUFFICIENT_RARE_MISSES"
    category_fraction = rare["miss_due_to_category_rank_fraction"]
    global_fraction = rare["miss_despite_rank_cutoff_fraction"]
    if category_fraction > 0.6:
        verdict = "WITHIN_QUERY_CATEGORY_RANK_DOMINANT"
    elif global_fraction > 0.6:
        verdict = "IMAGE_LEVEL_TOPK_DOMINANT"
    else:
        verdict = "MIXED_CATEGORY_AND_GLOBAL_RANKING"
    print(f"verdict: {verdict}")
    print(
        f"Among missed best-query pairs, {100.0 * category_fraction:.2f}% have "
        f"true-class rank > {cutoff}; {100.0 * global_fraction:.2f}% are already "
        f"within top-{cutoff} but are displaced at image-level top-k."
    )
    return verdict


def make_component_accumulator():
    metrics = (
        "detector_probability",
        "vlm_probability",
        "fused_score",
        "topk_threshold",
        "score_margin",
        "log_score_margin",
        "best_iou",
        "category_rank",
    )
    return {
        split: {
            status: {metric: [] for metric in metrics}
            for status in ("hit", "miss", "all")
        }
        for split in ("all", "r", "c", "f")
    }


def update_component_accumulator(
    accumulator,
    *,
    detector_probabilities,
    vlm_probabilities,
    fused_scores,
    topk_threshold,
    best_ious,
    category_ranks,
    survives,
    frequencies,
):
    eps = torch.finfo(torch.float32).tiny
    thresholds = fused_scores.new_full(fused_scores.shape, float(topk_threshold))
    values_by_metric = {
        "detector_probability": detector_probabilities,
        "vlm_probability": vlm_probabilities,
        "fused_score": fused_scores,
        "topk_threshold": thresholds,
        "score_margin": fused_scores - thresholds,
        "log_score_margin": fused_scores.clamp_min(eps).log()
        - thresholds.clamp_min(eps).log(),
        "best_iou": best_ious,
        "category_rank": category_ranks.float(),
    }
    cpu_values = {
        key: values.detach().float().cpu().tolist()
        for key, values in values_by_metric.items()
    }
    for index, (survived, frequency) in enumerate(
        zip(survives.tolist(), frequencies)
    ):
        status = "hit" if survived else "miss"
        for split in ("all", frequency):
            for metric, values in cpu_values.items():
                value = float(values[index])
                accumulator[split][status][metric].append(value)
                accumulator[split]["all"][metric].append(value)


def distribution_summary(values):
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
    }


def correlation(x_values, y_values):
    if len(x_values) < 2 or len(y_values) != len(x_values):
        return None
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def ordinal_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def finalize_component_accumulator(accumulator):
    report = {}
    for split, statuses in accumulator.items():
        split_report = {
            status: {
                metric: distribution_summary(values)
                for metric, values in metrics.items()
            }
            for status, metrics in statuses.items()
        }
        all_values = statuses["all"]
        split_report["correlation"] = {
            "pearson_iou_log_margin": correlation(
                all_values["best_iou"], all_values["log_score_margin"]
            ),
            "spearman_iou_log_margin": correlation(
                ordinal_ranks(all_values["best_iou"]),
                ordinal_ranks(all_values["log_score_margin"]),
            ),
        }
        report[split] = split_report
    return report


def component_branch_verdict(report, *, detector_weight, vlm_weight):
    rare = report["r"]
    hit = rare["hit"]
    miss = rare["miss"]
    if not hit["detector_probability"].get("count") or not miss[
        "detector_probability"
    ].get("count"):
        return {
            "verdict": "INSUFFICIENT_RARE_HIT_MISS_SAMPLES",
            "detector_weighted_log_separation": None,
            "vlm_weighted_log_separation": None,
        }

    eps = np.finfo(np.float64).tiny
    detector_separation = float(detector_weight) * (
        math.log(max(hit["detector_probability"]["median"], eps))
        - math.log(max(miss["detector_probability"]["median"], eps))
    )
    vlm_separation = float(vlm_weight) * (
        math.log(max(hit["vlm_probability"]["median"], eps))
        - math.log(max(miss["vlm_probability"]["median"], eps))
    )
    positive_detector = max(detector_separation, 0.0)
    positive_vlm = max(vlm_separation, 0.0)
    if positive_detector > 1.25 * max(positive_vlm, 1e-12):
        verdict = "DETECTOR_COMPONENT_DOMINANT"
    elif positive_vlm > 1.25 * max(positive_detector, 1e-12):
        verdict = "CLIP_COMPONENT_DOMINANT"
    else:
        verdict = "MIXED_DETECTOR_AND_CLIP_COMPONENTS"
    return {
        "verdict": verdict,
        "detector_weighted_log_separation": detector_separation,
        "vlm_weighted_log_separation": vlm_separation,
    }


def print_component_report(report, verdict):
    print("\n=== Correct-pair score components: hit vs miss ===")
    print("A hit means the best-IoU query/true-class pair survives image top-k.")
    print(
        f"{'split':>6} {'status':>6} {'N':>6} {'det-med':>10} {'vlm-med':>10} "
        f"{'fused-med':>11} {'thr-med':>10} {'logmargin-med':>14} {'IoU-med':>9}"
    )
    for split in ("all", "r", "c", "f"):
        for status in ("hit", "miss"):
            row = report[split][status]
            count = row["fused_score"].get("count", 0)
            if not count:
                continue
            print(
                f"{split:>6} {status:>6} {count:6d} "
                f"{row['detector_probability']['median']:10.6f} "
                f"{row['vlm_probability']['median']:10.6f} "
                f"{row['fused_score']['median']:11.6f} "
                f"{row['topk_threshold']['median']:10.6f} "
                f"{row['log_score_margin']['median']:14.4f} "
                f"{row['best_iou']['median']:9.4f}"
            )
    print("\n=== Rare score-component verdict ===")
    for key, value in verdict.items():
        print(f"{key}: {value}")
    print("rare_iou_margin_correlation:", report["r"]["correlation"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        "--config",
        dest="config_file",
        default="lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py",
    )
    parser.add_argument("--checkpoint", "--ckpt", dest="checkpoint", required=True)
    parser.add_argument("--num-images", type=int, default=500, help="0 means all images")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-iou", type=float, default=0.5)
    parser.add_argument("--min-per-class", type=int, default=5)
    parser.add_argument(
        "--sampling",
        choices=("random", "rare"),
        default="random",
        help="rare keeps only validation images containing at least one rare GT",
    )
    parser.add_argument("--image-topk", type=int, default=300)
    parser.add_argument("--category-rank-cutoff", type=int, default=5)
    parser.add_argument("--output", default=None, help="optional JSON report path")
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="LazyConfig overrides, e.g. model.beta=0.3 model.novel_scale=3.0",
    )
    args = parser.parse_args()
    if not 0.0 < args.min_iou <= 1.0:
        raise ValueError("--min-iou must be within (0,1]")
    if args.image_topk < 1:
        raise ValueError("--image-topk must be positive")
    if args.category_rank_cutoff < 1:
        raise ValueError("--category-rank-cutoff must be positive")

    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import LazyConfig, instantiate
    from detectron2.data import MetadataCatalog, build_detection_test_loader
    from detectron2.data import get_detection_dataset_dicts
    from detectron2.structures import BoxMode

    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    dataset_name = cfg.dataloader.test.dataset.names
    if not isinstance(dataset_name, str):
        if len(dataset_name) != 1:
            raise ValueError("prototype diagnostic requires exactly one test dataset")
        dataset_name = dataset_name[0]

    print(f"[load] config={args.config_file}")
    model = instantiate(cfg.model)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"[load] checkpoint={args.checkpoint}")
    DetectionCheckpointer(model).load(args.checkpoint)
    if not getattr(model, "score_ensemble", False):
        raise ValueError("exact fused-rank diagnosis requires model.score_ensemble=True")
    fusion_protocol = {
        "alpha": float(model.alpha),
        "beta": float(model.beta),
        "novel_scale": float(model.novel_scale),
        "vlm_temperature": float(model.vlm_temperature),
    }
    print(f"[protocol] {fusion_protocol}")

    metadata = MetadataCatalog.get(dataset_name)
    capture = {}
    classifier = install_capture_hooks(model, capture)
    frequencies_by_class = frequency_lookup(metadata, model.num_classes)
    records = get_detection_dataset_dicts(names=dataset_name, filter_empty=False)
    records = select_dataset_records(
        records,
        args.num_images,
        args.seed,
        args.sampling,
        frequencies_by_class,
    )
    records_by_id = {record["image_id"]: record for record in records}
    loader = build_detection_test_loader(
        dataset=records,
        mapper=instantiate(cfg.dataloader.test.mapper),
        num_workers=0,
    )

    ranking = {
        name: defaultdict(lambda: defaultdict(float))
        for name in (
            "fused_current",
            "logmeanexp",
            "prototype_mean",
            "prompt_mean",
        )
    }
    selection_accumulator = make_selection_accumulator()
    component_accumulator = make_component_accumulator()
    occupancy = {
        "images": 0,
        "unique_queries": [],
        "unique_classes": [],
        "max_pairs_per_query": [],
        "pairs_per_unique_query": [],
        "frequency_pair_counts": Counter(),
    }
    winners_by_class = defaultdict(list)
    mode_maxima = []
    mode_entropies = []
    routing_category_maxima = []
    routing_prototype_maxima = []
    matched_count = 0
    gt_count = 0
    lme_max_error = 0.0
    fusion_max_error = 0.0
    novel_mask = model.novel_idx.to(device=device)

    print(
        f"[run] {len(records)} images, sampling={args.sampling}, "
        f"min IoU={args.min_iou}"
    )
    with torch.no_grad():
        for batch_index, batched_inputs in enumerate(loader):
            capture.clear()
            _ = model(batched_inputs)
            required = {
                "projected_features",
                "detector_logits",
                "prototypes",
                "prompt_features",
                "query_boxes",
                "final_scores",
                "roi_features",
            }
            missing = required - set(capture)
            if missing:
                raise RuntimeError(f"capture hooks missed: {sorted(missing)}")

            features = capture["projected_features"]
            detector_logits = capture["detector_logits"]
            query_boxes = capture["query_boxes"]
            prototypes = capture["prototypes"]
            prompts = capture["prompt_features"]
            final_scores = capture["final_scores"].float()
            roi_features = capture["roi_features"].float()
            if final_scores.shape != detector_logits.shape:
                raise RuntimeError(
                    "final fused score shape does not match detector logits: "
                    f"{final_scores.shape} vs {detector_logits.shape}"
                )

            for local_index, model_input in enumerate(batched_inputs):
                image_scores = final_scores[local_index]
                selected_count = min(args.image_topk, image_scores.numel())
                top_values, top_flat_ids = image_scores.reshape(-1).topk(
                    selected_count
                )
                top_query_ids = torch.div(
                    top_flat_ids, image_scores.shape[-1], rounding_mode="floor"
                )
                top_class_ids = top_flat_ids % image_scores.shape[-1]
                unique_queries, pairs_per_query = torch.unique(
                    top_query_ids, return_counts=True
                )
                occupancy["images"] += 1
                occupancy["unique_queries"].append(int(unique_queries.numel()))
                occupancy["unique_classes"].append(
                    int(torch.unique(top_class_ids).numel())
                )
                occupancy["max_pairs_per_query"].append(
                    int(pairs_per_query.max()) if pairs_per_query.numel() else 0
                )
                occupancy["pairs_per_unique_query"].append(
                    selected_count / max(int(unique_queries.numel()), 1)
                )
                occupancy["frequency_pair_counts"].update(
                    frequencies_by_class[class_id]
                    for class_id in top_class_ids.tolist()
                )

                record = records_by_id[model_input["image_id"]]
                annotations = record.get("annotations", [])
                if not annotations:
                    continue
                gt_count += len(annotations)
                gt_boxes = torch.tensor(
                    [
                        BoxMode.convert(
                            annotation["bbox"],
                            annotation["bbox_mode"],
                            BoxMode.XYXY_ABS,
                        )
                        for annotation in annotations
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                gt_classes = torch.tensor(
                    [annotation["category_id"] for annotation in annotations],
                    dtype=torch.long,
                    device=device,
                )
                predicted = box_cxcywh_to_xyxy(query_boxes[local_index].float())
                scale = predicted.new_tensor(
                    [record["width"], record["height"], record["width"], record["height"]]
                )
                predicted = predicted * scale
                predicted[:, 0::2].clamp_(0, record["width"])
                predicted[:, 1::2].clamp_(0, record["height"])
                overlaps = box_iou(gt_boxes, predicted)
                best_iou, best_query = overlaps.max(dim=1)
                valid = best_iou >= args.min_iou
                if not valid.any():
                    continue

                selected_queries = best_query[valid]
                selected_classes = gt_classes[valid]
                selected_features = features[local_index, selected_queries]
                selected_frequencies = [
                    frequencies_by_class[class_id]
                    for class_id in selected_classes.tolist()
                ]
                matched_count += int(valid.sum())

                variants = prototype_variant_logits(
                    selected_features,
                    prototypes,
                    prompts,
                    temperature=classifier.tpa_cls_tau,
                    logit_scale=classifier.norm_temperature,
                )
                if classifier.use_bias:
                    variants = {
                        name: logits + classifier.cls_bias
                        for name, logits in variants.items()
                    }
                current_logits = detector_logits[local_index, selected_queries].float()
                lme_max_error = max(
                    lme_max_error,
                    float((variants["logmeanexp"] - current_logits).abs().max().item()),
                )
                variants["logmeanexp"] = current_logits
                for name, logits in variants.items():
                    update_ranking_stats(
                        ranking[name], logits, selected_classes, selected_frequencies
                    )

                fused_selected_scores = image_scores[selected_queries]
                fused_ranks = update_ranking_stats(
                    ranking["fused_current"],
                    fused_selected_scores,
                    selected_classes,
                    selected_frequencies,
                )
                true_pair_ids = (
                    selected_queries * image_scores.shape[-1] + selected_classes
                )
                survives_global_topk = (
                    true_pair_ids[:, None] == top_flat_ids[None, :]
                ).any(dim=1)
                update_selection_accumulator(
                    selection_accumulator,
                    fused_ranks,
                    survives_global_topk,
                    selected_frequencies,
                    args.category_rank_cutoff,
                )

                image_vlm_logits = (
                    roi_features[local_index]
                    @ model.vlm_content_query_embedding.t()
                    * float(model.vlm_temperature)
                )
                image_vlm_probabilities = image_vlm_logits.softmax(dim=-1)
                recomputed_fused_scores = fuse_detector_vlm_scores(
                    detector_logits[local_index].float(),
                    image_vlm_logits,
                    novel_mask,
                    fusion="power",
                    base_weight=float(model.alpha),
                    novel_weight=float(model.beta),
                    novel_scale=float(model.novel_scale),
                ).exp()
                fusion_max_error = max(
                    fusion_max_error,
                    float((recomputed_fused_scores - image_scores).abs().max().item()),
                )
                row_ids = torch.arange(
                    selected_classes.numel(), device=selected_classes.device
                )
                detector_true_probabilities = current_logits.sigmoid()[
                    row_ids, selected_classes
                ]
                vlm_true_probabilities = image_vlm_probabilities[
                    selected_queries, selected_classes
                ]
                fused_true_scores = fused_selected_scores[
                    row_ids, selected_classes
                ]
                update_component_accumulator(
                    component_accumulator,
                    detector_probabilities=detector_true_probabilities,
                    vlm_probabilities=vlm_true_probabilities,
                    fused_scores=fused_true_scores,
                    topk_threshold=float(top_values[-1]),
                    best_ious=best_iou[valid],
                    category_ranks=fused_ranks,
                    survives=survives_global_topk,
                    frequencies=selected_frequencies,
                )

                weights = true_class_mode_weights(
                    selected_features,
                    prototypes,
                    selected_classes,
                    temperature=classifier.tpa_cls_tau,
                )
                mode_maxima.extend(weights.max(dim=-1).values.cpu().tolist())
                mode_entropies.extend(normalized_entropy(weights).cpu().tolist())
                for class_id, winner in zip(
                    selected_classes.tolist(), weights.argmax(dim=-1).tolist()
                ):
                    winners_by_class[class_id].append(winner)

            routing = getattr(model.transformer, "last_query_fusion_stats", {})
            if routing:
                routing_category_maxima.append(float(routing["category_max_weight"]))
                routing_prototype_maxima.append(float(routing["prototype_max_weight"]))
            if (batch_index + 1) % 50 == 0:
                print(
                    f"  [{batch_index + 1}/{len(records)}] "
                    f"GT matched={matched_count}/{gt_count}"
                )

    finalized = {
        name: finalize_ranking_stats(stats) for name, stats in ranking.items()
    }
    print_ranking_table(finalized)

    selection_report = finalize_selection_accumulator(
        selection_accumulator, args.category_rank_cutoff
    )
    ranking_verdict = print_selection_report(
        selection_report,
        args.image_topk,
        args.category_rank_cutoff,
    )
    component_report = finalize_component_accumulator(component_accumulator)
    component_verdict = component_branch_verdict(
        component_report,
        detector_weight=1.0 - float(model.beta),
        vlm_weight=float(model.beta),
    )
    component_verdict["fusion_recompute_max_abs_error"] = fusion_max_error
    print_component_report(component_report, component_verdict)
    total_pairs = sum(occupancy["frequency_pair_counts"].values())
    occupancy_report = {
        "images": occupancy["images"],
        "mean_unique_queries": float(np.mean(occupancy["unique_queries"])),
        "mean_unique_classes": float(np.mean(occupancy["unique_classes"])),
        "mean_max_pairs_per_query": float(
            np.mean(occupancy["max_pairs_per_query"])
        ),
        "mean_pairs_per_unique_query": float(
            np.mean(occupancy["pairs_per_unique_query"])
        ),
        "frequency_pair_fraction": {
            split: occupancy["frequency_pair_counts"][split] / max(total_pairs, 1)
            for split in ("r", "c", "f")
        },
    }
    print("\n=== Image-level top-k occupancy ===")
    for key, value in occupancy_report.items():
        print(f"{key}: {value}")

    if matched_count == 0:
        raise RuntimeError(
            "no GT box matched a final query; lower --min-iou or inspect box capture"
        )
    prototype_count = int(prototypes.shape[1])
    winner_counts = Counter(
        winner for winners in winners_by_class.values() for winner in winners
    )
    winner_probabilities = torch.tensor(
        [winner_counts[index] for index in range(prototype_count)], dtype=torch.float32
    )
    winner_probabilities /= winner_probabilities.sum().clamp_min(1)
    class_usage_entropies = []
    for winners in winners_by_class.values():
        if len(winners) < args.min_per_class:
            continue
        counts = torch.tensor(
            [winners.count(index) for index in range(prototype_count)], dtype=torch.float32
        )
        probabilities = counts / counts.sum()
        class_usage_entropies.append(float(normalized_entropy(probabilities[None])[0]))

    mode_report = {
        "prototype_count": prototype_count,
        "matched_gt": matched_count,
        "total_gt": gt_count,
        "matched_ratio": matched_count / max(gt_count, 1),
        "mean_posterior_max": float(np.mean(mode_maxima)) if mode_maxima else None,
        "mean_posterior_normalized_entropy": (
            float(np.mean(mode_entropies)) if mode_entropies else None
        ),
        "winner_distribution": winner_probabilities.tolist(),
        "winner_normalized_entropy": float(
            normalized_entropy(winner_probabilities[None])[0]
        ),
        "classes_with_min_matches": len(class_usage_entropies),
        "mean_per_class_winner_entropy": (
            float(np.mean(class_usage_entropies)) if class_usage_entropies else None
        ),
        "query_fusion_category_max_weight": (
            float(np.mean(routing_category_maxima)) if routing_category_maxima else None
        ),
        "query_fusion_prototype_max_weight": (
            float(np.mean(routing_prototype_maxima)) if routing_prototype_maxima else None
        ),
        "classifier_recompute_max_abs_error": lme_max_error,
    }
    print("\n=== Mode utilization on true-class matched queries ===")
    for key, value in mode_report.items():
        print(f"{key}: {value}")

    current = finalized["logmeanexp"]["all"]
    baselines = [
        finalized["prototype_mean"]["all"],
        finalized["prompt_mean"]["all"],
    ]
    best_baseline_top1 = max(row.get("top1", 0.0) for row in baselines)
    gain = current.get("top1", 0.0) - best_baseline_top1
    diffuse = (
        mode_report["mean_posterior_normalized_entropy"] is not None
        and mode_report["mean_posterior_normalized_entropy"] > 0.9
    )
    print("\n=== Decision ===")
    if gain > 0.005 and not diffuse:
        verdict = "PROTOTYPES_USED"
        explanation = "K-prototype classification beats both one-vector controls and routing is not uniform."
    elif gain <= 0.005 and diffuse:
        verdict = "PROTOTYPES_NOT_EFFECTIVE"
        explanation = "K prototypes do not beat one-vector controls and per-instance mode weights are near-uniform."
    else:
        verdict = "MIXED"
        explanation = "Classification gain and mode utilization disagree; inspect split metrics before changing TPA."
    print(f"verdict: {verdict}")
    print(f"top1_gain_over_best_one_vector: {gain:+.6f}")
    print(explanation)

    report = {
        "config": args.config_file,
        "checkpoint": args.checkpoint,
        "num_images": len(records),
        "sampling": args.sampling,
        "min_iou": args.min_iou,
        "fusion_protocol": fusion_protocol,
        "ranking": finalized,
        "selection": selection_report,
        "selection_verdict": ranking_verdict,
        "score_components": component_report,
        "score_component_verdict": component_verdict,
        "topk_occupancy": occupancy_report,
        "mode_utilization": mode_report,
        "verdict": verdict,
        "top1_gain_over_best_one_vector": gain,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[save] {output_path}")


if __name__ == "__main__":
    main()
