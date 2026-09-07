#!/usr/bin/env python
"""Diagnose whether non-collapsed TPA modes improve final-query recognition.

The diagnostic matches final decoder boxes to LVIS ground-truth boxes and then
compares the true-class ranking of three detector-side text representations:

* ``logmeanexp``: the current K-prototype Eq. (2) classifier;
* ``prototype_mean``: the same learned prototypes averaged into one vector;
* ``prompt_mean``: the original prompt bank averaged into one vector.

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

    original_inference = model.inference

    def capture_inference(box_cls, box_pred, image_sizes, wo_sigmoid=False):
        capture["query_boxes"] = box_pred.detach()
        capture["image_sizes"] = tuple(image_sizes)
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
            row["reciprocal_rank"] += 1.0 / float(rank)
            row["rank_sum"] += int(rank)


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
        f"{'top1':>8} {'top5':>8} {'MRR':>8} {'mean_rank':>11}"
    )
    for variant, splits in results.items():
        for split in ("all", "r", "c", "f"):
            row = splits[split]
            if not row.get("count"):
                print(f"{variant:>18} {split:>6} {0:7d} {'-':>8} {'-':>8} {'-':>8} {'-':>11}")
                continue
            print(
                f"{variant:>18} {split:>6} {row['count']:7d} "
                f"{format_percent(row['top1']):>8} "
                f"{format_percent(row['top5']):>8} "
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


def select_dataset_records(records, num_images, seed):
    if num_images <= 0 or num_images >= len(records):
        return records
    generator = np.random.default_rng(seed)
    indices = np.sort(generator.choice(len(records), size=num_images, replace=False))
    return [records[int(index)] for index in indices]


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
    parser.add_argument("--output", default=None, help="optional JSON report path")
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="LazyConfig overrides, e.g. model.beta=0.3 model.novel_scale=3.0",
    )
    args = parser.parse_args()

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

    metadata = MetadataCatalog.get(dataset_name)
    records = get_detection_dataset_dicts(names=dataset_name, filter_empty=False)
    records = select_dataset_records(records, args.num_images, args.seed)
    records_by_id = {record["image_id"]: record for record in records}
    loader = build_detection_test_loader(
        dataset=records,
        mapper=instantiate(cfg.dataloader.test.mapper),
        num_workers=0,
    )

    capture = {}
    classifier = install_capture_hooks(model, capture)
    frequencies_by_class = frequency_lookup(metadata, model.num_classes)
    ranking = {
        name: defaultdict(lambda: defaultdict(float))
        for name in ("logmeanexp", "prototype_mean", "prompt_mean")
    }
    winners_by_class = defaultdict(list)
    mode_maxima = []
    mode_entropies = []
    routing_category_maxima = []
    routing_prototype_maxima = []
    matched_count = 0
    gt_count = 0
    lme_max_error = 0.0

    print(f"[run] {len(records)} images, min IoU={args.min_iou}")
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
            }
            missing = required - set(capture)
            if missing:
                raise RuntimeError(f"capture hooks missed: {sorted(missing)}")

            features = capture["projected_features"]
            detector_logits = capture["detector_logits"]
            query_boxes = capture["query_boxes"]
            prototypes = capture["prototypes"]
            prompts = capture["prompt_features"]

            for local_index, model_input in enumerate(batched_inputs):
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
        "min_iou": args.min_iou,
        "ranking": finalized,
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
