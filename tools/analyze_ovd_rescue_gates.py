#!/usr/bin/env python
"""Screen precision-constrained rare rescue gates and evaluate only survivors.

The cache must include explicit novel-only detector and VLM candidate pools.
This guarantees that top-k replay of

    max(current, detector_multiplier * detector, vlm_multiplier * vlm)

is exact while leaving base/common/frequent scores unchanged.  A cheap fixed-
IoU screen first rejects gates that add too many evaluated rare false positives;
only up to ``--evaluate-top`` survivors receive official LVIS evaluation.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lami_dino.diagnostic_ops import fuse_sparse_detector_vlm_scores  # noqa: E402
from tools.analyze_ovd_error_decomposition import (  # noqa: E402
    category_frequency,
    classify_rare_detections,
    detector_logits_for_source,
    normalized_cxcywh_to_xyxy,
    pairwise_iou,
    xywh_to_xyxy,
)
from tools.diagnose_tpa_usage import greedy_gt_topk_matches  # noqa: E402
from tools.evaluate_ovd_fusion import (  # noqa: E402
    evaluate_lvis,
    normalized_boxes_to_lvis,
)


REQUIRED_EXTENSIONS = {
    "detector_scaled_novel_only",
    "vlm_scaled_novel_only",
}
METRICS = ("AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf")


def load_tensor_file(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def profile_log_scores(payload, profile, novel_mask, path):
    query_ids = payload["candidate_query_ids"].long()
    class_ids = payload["candidate_class_ids"].long()
    detector_logits = detector_logits_for_source(
        payload, profile.get("detector_source", "logmeanexp"), path
    )
    return fuse_sparse_detector_vlm_scores(
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


def gated_log_scores(
    current_scores,
    detector_scores,
    vlm_scores,
    candidate_classes,
    novel_mask,
    detector_multiplier,
    vlm_multiplier,
):
    """Apply rescue branches to novel candidates only."""
    if not (
        current_scores.shape == detector_scores.shape == vlm_scores.shape
        and current_scores.ndim == 1
    ):
        raise ValueError("all sparse score tensors must be same-length vectors")
    if candidate_classes.shape != current_scores.shape:
        raise ValueError("candidate_classes must match sparse scores")
    if detector_multiplier < 0.0 or vlm_multiplier < 0.0:
        raise ValueError("gate multipliers must be non-negative")
    result = current_scores.clone()
    selected = novel_mask[candidate_classes.long()]
    if detector_multiplier > 0.0:
        result[selected] = torch.maximum(
            result[selected],
            detector_scores[selected] + math.log(float(detector_multiplier)),
        )
    if vlm_multiplier > 0.0:
        result[selected] = torch.maximum(
            result[selected],
            vlm_scores[selected] + math.log(float(vlm_multiplier)),
        )
    return result


def gate_name(detector_multiplier, vlm_multiplier):
    def token(value):
        return f"{float(value):g}".replace(".", "p")

    return f"gate_d{token(detector_multiplier)}_v{token(vlm_multiplier)}"


def build_gate_grid(detector_multipliers, vlm_multipliers):
    specs = [(0.0, 0.0)]
    specs.extend((float(value), 0.0) for value in detector_multipliers)
    specs.extend((0.0, float(value)) for value in vlm_multipliers)
    specs.extend(
        (float(detector), float(vlm))
        for detector in detector_multipliers
        for vlm in vlm_multipliers
    )
    result = []
    seen = set()
    for detector, vlm in specs:
        key = (detector, vlm)
        if key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "name": gate_name(detector, vlm),
                "detector_multiplier": detector,
                "vlm_multiplier": vlm,
            }
        )
    return result


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


def make_rescue_accumulator():
    metrics = (
        "detector_probability",
        "vlm_probability",
        "current_score",
        "branch_score",
        "current_topk_threshold",
        "branch_topk_threshold",
        "required_multiplier_vs_current_threshold",
        "component_margin",
        "matched_iou",
    )
    return {
        branch: {metric: [] for metric in metrics}
        for branch in ("detector_only", "vlm_only")
    }


def component_margin(payload, branch, query_id, class_id, true_probability):
    summary = payload.get("component_query_summary")
    if summary is None:
        raise ValueError(
            "raw payload lacks component_query_summary; regenerate the rescue cache"
        )
    prefix = "detector" if branch == "detector_only" else "vlm"
    top_classes = summary[f"{prefix}_top_classes"][query_id].long()
    top_probabilities = summary[f"{prefix}_top_probabilities"][query_id].float()
    competitor = (
        top_probabilities[1]
        if int(top_classes[0]) == int(class_id)
        else top_probabilities[0]
    )
    return float(true_probability - competitor)


def update_rescue_distribution(
    accumulator,
    branch,
    gt_indices,
    matched_queries,
    gt_classes,
    overlaps,
    candidate_flat_ids,
    detector_logits,
    vlm_logits,
    vlm_log_normalizer,
    current_scores,
    branch_scores,
    current_threshold,
    branch_threshold,
    payload,
    num_classes,
):
    row = accumulator[branch]
    for gt_index in gt_indices.tolist():
        query_id = int(matched_queries[gt_index])
        class_id = int(gt_classes[gt_index])
        flat_id = query_id * num_classes + class_id
        position = int(torch.searchsorted(candidate_flat_ids, flat_id))
        if (
            position >= candidate_flat_ids.numel()
            or int(candidate_flat_ids[position]) != flat_id
        ):
            raise RuntimeError(
                f"rescue pair ({query_id}, {class_id}) missing from exact cache"
            )
        detector_probability = float(detector_logits[position].sigmoid())
        vlm_probability = float(
            (
                vlm_logits[position]
                - vlm_log_normalizer[query_id]
            ).exp()
        )
        true_probability = (
            detector_probability if branch == "detector_only" else vlm_probability
        )
        branch_log_score = float(branch_scores[position])
        row["detector_probability"].append(detector_probability)
        row["vlm_probability"].append(vlm_probability)
        row["current_score"].append(math.exp(float(current_scores[position])))
        row["branch_score"].append(math.exp(branch_log_score))
        row["current_topk_threshold"].append(math.exp(float(current_threshold)))
        row["branch_topk_threshold"].append(math.exp(float(branch_threshold)))
        row["required_multiplier_vs_current_threshold"].append(
            math.exp(float(current_threshold) - branch_log_score)
        )
        row["component_margin"].append(
            component_margin(
                payload,
                branch,
                query_id,
                class_id,
                true_probability,
            )
        )
        row["matched_iou"].append(float(overlaps[gt_index, query_id]))


def finalize_rescue_accumulator(accumulator):
    return {
        branch: {
            metric: distribution_summary(values) for metric, values in row.items()
        }
        for branch, row in accumulator.items()
    }


def _outcome_counts(records):
    counts = Counter(record[2] for record in records)
    evaluated = sum(
        count
        for outcome, count in counts.items()
        if not outcome.startswith("ignored_")
    )
    true_positives = int(counts["tp"])
    return counts, evaluated, true_positives, evaluated - true_positives


def finalize_screen(screen, specs):
    baseline_name = gate_name(0.0, 0.0)
    baseline = screen[baseline_name]
    rows = []
    for spec in specs:
        source = screen[spec["name"]]
        evaluated = int(source["evaluated"])
        tp = int(source["tp"])
        fp = int(source["fp"])
        rescued = int(source["rescued_tp"])
        lost = int(source["lost_tp"])
        net_tp = tp - int(baseline["tp"])
        delta_fp = fp - int(baseline["fp"])
        added_evaluated = int(source["added_evaluated"])
        added_tp = int(source["added_tp"])
        added_fp = int(source["added_fp"])
        row = {
            **spec,
            "evaluated_rare_detections": evaluated,
            "rare_tp": tp,
            "rare_fp": fp,
            "diagnostic_precision": tp / evaluated if evaluated else None,
            "rescued_tp": rescued,
            "lost_tp": lost,
            "net_tp": net_tp,
            "delta_fp": delta_fp,
            "added_rare_candidates": int(source["added_candidates"]),
            "added_evaluated_rare_candidates": added_evaluated,
            "added_candidate_tp": added_tp,
            "added_candidate_fp": added_fp,
            "added_candidate_precision": (
                added_tp / added_evaluated if added_evaluated else None
            ),
            "added_candidate_outcomes": {
                outcome: int(source[f"added_{outcome}"])
                for outcome in (
                    "duplicate",
                    "classification",
                    "localization",
                    "background",
                    "ignored_not_exhaustive",
                    "ignored_unknown",
                )
            },
            "added_fp_per_net_tp": (
                max(delta_fp, 0) / net_tp if net_tp > 0 else None
            ),
            "selected_frequency_counts": {
                split: int(source[f"selected_{split}"])
                for split in ("r", "c", "f")
            },
        }
        rows.append(row)
    return rows


def choose_gates(rows, max_added_fp_per_net_tp, max_lost_tp, limit):
    candidates = [
        row
        for row in rows
        if (row["detector_multiplier"] > 0.0 or row["vlm_multiplier"] > 0.0)
        and row["net_tp"] > 0
        and row["lost_tp"] <= max_lost_tp
        and row["added_fp_per_net_tp"] <= max_added_fp_per_net_tp
    ]
    pareto = []
    for row in candidates:
        dominated = any(
            other["net_tp"] >= row["net_tp"]
            and other["delta_fp"] <= row["delta_fp"]
            and other["lost_tp"] <= row["lost_tp"]
            and (
                other["net_tp"] > row["net_tp"]
                or other["delta_fp"] < row["delta_fp"]
                or other["lost_tp"] < row["lost_tp"]
            )
            for other in candidates
        )
        if not dominated:
            pareto.append(row)
    pareto.sort(
        key=lambda row: (
            -row["net_tp"],
            row["added_fp_per_net_tp"],
            row["lost_tp"],
        )
    )
    return pareto[:limit]


def print_rescue_report(report):
    print("\n=== Isolated-component rescue distributions ===")
    print(
        f"{'branch':>14} {'N':>5} {'det-med':>10} {'vlm-med':>10} "
        f"{'branch-med':>12} {'margin-med':>11} {'required-gamma-med':>19}"
    )
    for branch in ("detector_only", "vlm_only"):
        row = report[branch]
        print(
            f"{branch:>14} {row['branch_score']['count']:5d} "
            f"{row['detector_probability'].get('median', float('nan')):10.6f} "
            f"{row['vlm_probability'].get('median', float('nan')):10.6f} "
            f"{row['branch_score'].get('median', float('nan')):12.6f} "
            f"{row['component_margin'].get('median', float('nan')):11.6f} "
            f"{row['required_multiplier_vs_current_threshold'].get('median', float('nan')):19.6f}"
        )


def print_rescue_partition(partition):
    print("\n=== Rare current-miss component partition @IoU=0.50 ===")
    print(
        "current_miss={current_miss} detector_only={detector_only} "
        "vlm_only={vlm_only} both={both} neither={neither}".format(**partition)
    )


def print_screen(rows, selected, max_added_fp_per_net_tp, max_lost_tp):
    print("\n=== Precision-constrained rare gate screen @IoU=0.50 ===")
    print(
        f"{'gate':>20} {'TP':>6} {'FP':>7} {'resc':>6} {'lost':>6} "
        f"{'netTP':>6} {'dFP':>7} {'newTP':>6} {'newFP':>7} "
        f"{'newPrec%':>9} {'dFP/netTP':>11}"
    )
    for row in rows:
        ratio = row["added_fp_per_net_tp"]
        precision = row["added_candidate_precision"]
        print(
            f"{row['name']:>20} {row['rare_tp']:6d} {row['rare_fp']:7d} "
            f"{row['rescued_tp']:6d} {row['lost_tp']:6d} "
            f"{row['net_tp']:6d} {row['delta_fp']:7d} "
            f"{row['added_candidate_tp']:6d} {row['added_candidate_fp']:7d} "
            f"{100.0 * precision if precision is not None else float('nan'):9.2f} "
            f"{ratio if ratio is not None else float('nan'):11.3f}"
        )
    print("\n=== Gates selected for official LVIS evaluation ===")
    print(f"max_added_fp_per_net_tp: {max_added_fp_per_net_tp}")
    print(f"max_lost_tp: {max_lost_tp}")
    if selected:
        for row in selected:
            print(
                f"{row['name']}: net_tp={row['net_tp']}, "
                f"delta_fp={row['delta_fp']}, lost_tp={row['lost_tp']}"
            )
    else:
        print("NONE: no gate satisfies the precision budget; formal evaluation skipped.")


def selected_gate_pairs(payload, sources, spec, novel_mask, max_dets):
    class_ids = payload["candidate_class_ids"].long()
    scores = gated_log_scores(
        sources["current"],
        sources["detector"],
        sources["vlm"],
        class_ids,
        novel_mask,
        spec["detector_multiplier"],
        spec["vlm_multiplier"],
    )
    selected = scores.topk(min(max_dets, scores.numel())).indices
    return selected, scores


def predictions_for_gate(files, profiles, spec, novel_mask, max_dets):
    predictions = []
    for index, path in enumerate(files):
        payload = load_tensor_file(path)
        query_ids = payload["candidate_query_ids"].long()
        class_ids = payload["candidate_class_ids"].long()
        sources = {
            "current": profile_log_scores(
                payload, profiles["current_power"], novel_mask, path
            ),
            "detector": profile_log_scores(
                payload, profiles["detector_scaled"], novel_mask, path
            ),
            "vlm": profile_log_scores(
                payload, profiles["vlm_scaled"], novel_mask, path
            ),
        }
        selected, gate_scores = selected_gate_pairs(
            payload, sources, spec, novel_mask, max_dets
        )
        selected_queries = query_ids[selected]
        selected_classes = class_ids[selected]
        selected_scores = gate_scores[selected].exp()
        selected_boxes = normalized_boxes_to_lvis(
            payload["query_boxes"][selected_queries].float(),
            int(payload["width"]),
            int(payload["height"]),
        )
        valid = (selected_boxes[:, 2] > 0) & (selected_boxes[:, 3] > 0)
        for box, score, class_id in zip(
            selected_boxes[valid].tolist(),
            selected_scores[valid].tolist(),
            selected_classes[valid].tolist(),
        ):
            predictions.append(
                {
                    "image_id": int(payload["image_id"]),
                    "category_id": int(class_id) + 1,
                    "bbox": box,
                    "score": float(score),
                }
            )
        if (index + 1) % 500 == 0:
            print(f"  [{spec['name']}] loaded {index + 1}/{len(files)}", flush=True)
    return predictions


def load_baseline_metrics(path):
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "current_power" in payload:
        payload = payload["current_power"]
    if not all(metric in payload for metric in METRICS):
        raise ValueError(f"baseline metrics in {path} do not contain LVIS metrics")
    return {metric: float(payload[metric]) for metric in METRICS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--max-dets", type=int, default=300)
    parser.add_argument(
        "--detector-multipliers", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0]
    )
    parser.add_argument(
        "--vlm-multipliers", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0]
    )
    parser.add_argument("--max-added-fp-per-net-tp", type=float, default=1.0)
    parser.add_argument("--max-lost-tp", type=int, default=5)
    parser.add_argument("--evaluate-top", type=int, default=3)
    parser.add_argument("--baseline-metrics", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.max_dets < 1 or args.evaluate_top < 0:
        raise ValueError("max detections must be positive and evaluate-top non-negative")
    if args.max_added_fp_per_net_tp < 0.0 or args.max_lost_tp < 0:
        raise ValueError("precision budgets must be non-negative")

    dump_dir = Path(args.dump_dir)
    manifest = json.loads((dump_dir / "manifest.json").read_text(encoding="utf-8"))
    extensions = set(manifest.get("candidate_pool_extensions", []))
    if not REQUIRED_EXTENSIONS <= extensions:
        raise ValueError(
            "cache lacks exact novel-only rescue pools; regenerate it with the "
            "current dump_ovd_raw_scores.py"
        )
    if args.max_dets > int(manifest["topk_per_profile"]):
        raise ValueError("--max-dets exceeds the exact cached top-k")
    profiles = {profile["name"]: profile for profile in manifest["profiles"]}
    required_profiles = {"current_power", "detector_scaled", "vlm_scaled"}
    if not required_profiles <= set(profiles):
        raise ValueError(f"cache lacks profiles: {sorted(required_profiles - set(profiles))}")

    files = sorted(dump_dir.glob("raw_*.pth"))
    if len(files) != int(manifest["num_dataset_images"]):
        raise RuntimeError("raw rescue cache is incomplete")
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
        if annotation["bbox"][2] <= 0 or annotation["bbox"][3] <= 0:
            continue
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    images_by_id = {int(image["id"]): image for image in annotation_data["images"]}

    num_classes = max(max(categories), max(manifest["novel_class_ids"]) + 1)
    novel_mask = torch.zeros(num_classes, dtype=torch.bool)
    novel_mask[torch.tensor(manifest["novel_class_ids"], dtype=torch.long)] = True
    specs = build_gate_grid(args.detector_multipliers, args.vlm_multipliers)
    screen = {
        spec["name"]: Counter(
            {
                "evaluated": 0,
                "tp": 0,
                "fp": 0,
                "rescued_tp": 0,
                "lost_tp": 0,
                "selected_r": 0,
                "selected_c": 0,
                "selected_f": 0,
                "added_candidates": 0,
                "added_evaluated": 0,
                "added_tp": 0,
                "added_fp": 0,
            }
        )
        for spec in specs
    }
    rescue_accumulator = make_rescue_accumulator()
    rescue_partition = Counter(
        {
            "current_miss": 0,
            "detector_only": 0,
            "vlm_only": 0,
            "both": 0,
            "neither": 0,
        }
    )

    for index, path in enumerate(files):
        payload = load_tensor_file(path)
        image_id = int(payload["image_id"])
        query_ids = payload["candidate_query_ids"].long()
        class_ids = payload["candidate_class_ids"].long()
        candidate_flat_ids = query_ids * num_classes + class_ids
        if not bool(torch.all(candidate_flat_ids[1:] >= candidate_flat_ids[:-1])):
            raise ValueError(f"candidate ids are not sorted in {path}")
        detector_logits = detector_logits_for_source(
            payload, "logmeanexp", path
        ).float()
        sources = {
            "current": profile_log_scores(
                payload, profiles["current_power"], novel_mask, path
            ),
            "detector": profile_log_scores(
                payload, profiles["detector_scaled"], novel_mask, path
            ),
            "vlm": profile_log_scores(
                payload, profiles["vlm_scaled"], novel_mask, path
            ),
        }
        query_boxes = normalized_cxcywh_to_xyxy(
            payload["query_boxes"], int(payload["width"]), int(payload["height"])
        )
        annotations = annotations_by_image.get(image_id, [])
        if annotations:
            gt_boxes = xywh_to_xyxy(
                torch.tensor([item["bbox"] for item in annotations], dtype=torch.float32)
            )
            gt_classes = torch.tensor(
                [int(item["category_id"]) - 1 for item in annotations],
                dtype=torch.long,
            )
        else:
            gt_boxes = query_boxes.new_empty((0, 4))
            gt_classes = class_ids.new_empty((0,))
        overlaps = pairwise_iou(gt_boxes, query_boxes)
        rare_gt = torch.tensor(
            [frequencies[int(class_id) + 1] == "r" for class_id in gt_classes],
            dtype=torch.bool,
        )

        component_selections = {}
        component_matches = {}
        for name, source in sources.items():
            selected = source.topk(min(args.max_dets, source.numel())).indices
            component_selections[name] = selected
            component_matches[name] = greedy_gt_topk_matches(
                overlaps,
                gt_classes,
                candidate_flat_ids[selected],
                num_classes,
                0.5,
            )
        current_hits, _ = component_matches["current"]
        detector_hits, detector_queries = component_matches["detector"]
        vlm_hits, vlm_queries = component_matches["vlm"]
        proposal_valid = (overlaps >= 0.5).any(dim=1)
        rare_current_miss = rare_gt & proposal_valid & ~current_hits
        rescue_partition["current_miss"] += int(rare_current_miss.sum())
        rescue_partition["detector_only"] += int(
            (rare_current_miss & detector_hits & ~vlm_hits).sum()
        )
        rescue_partition["vlm_only"] += int(
            (rare_current_miss & ~detector_hits & vlm_hits).sum()
        )
        rescue_partition["both"] += int(
            (rare_current_miss & detector_hits & vlm_hits).sum()
        )
        rescue_partition["neither"] += int(
            (rare_current_miss & ~detector_hits & ~vlm_hits).sum()
        )
        for branch, hits, other_hits, matched_queries, score_key in (
            (
                "detector_only",
                detector_hits,
                vlm_hits,
                detector_queries,
                "detector",
            ),
            ("vlm_only", vlm_hits, detector_hits, vlm_queries, "vlm"),
        ):
            rescued = rare_gt & ~current_hits & hits & ~other_hits
            if rescued.any():
                update_rescue_distribution(
                    rescue_accumulator,
                    branch,
                    torch.nonzero(rescued, as_tuple=False).flatten(),
                    matched_queries,
                    gt_classes,
                    overlaps,
                    candidate_flat_ids,
                    detector_logits,
                    payload["vlm_logits"].float(),
                    payload["vlm_log_normalizer"].float(),
                    sources["current"],
                    sources[score_key],
                    sources["current"][component_selections["current"][-1]],
                    sources[score_key][component_selections[score_key][-1]],
                    payload,
                    num_classes,
                )

        image_metadata = images_by_id[image_id]
        negative_classes = {
            int(value) - 1 for value in image_metadata.get("neg_category_ids", [])
        }
        not_exhaustive_classes = {
            int(value) - 1
            for value in image_metadata.get("not_exhaustive_category_ids", [])
        }
        for spec in specs:
            selected, gate_scores = selected_gate_pairs(
                payload, sources, spec, novel_mask, args.max_dets
            )
            selected_queries = query_ids[selected]
            selected_classes = class_ids[selected]
            selected_scores = gate_scores[selected].exp()
            row = screen[spec["name"]]
            for class_id in selected_classes.tolist():
                row[f"selected_{frequencies[int(class_id) + 1]}"] += 1
            rare_selected = torch.tensor(
                [
                    frequencies[int(class_id) + 1] == "r"
                    for class_id in selected_classes
                ],
                dtype=torch.bool,
            )
            outcomes = classify_rare_detections(
                query_boxes[selected_queries[rare_selected]],
                selected_scores[rare_selected],
                selected_classes[rare_selected],
                gt_boxes,
                gt_classes,
                negative_classes=negative_classes,
                not_exhaustive_classes=not_exhaustive_classes,
                iou_threshold=0.5,
                background_iou=0.1,
            )
            _, evaluated, tp, fp = _outcome_counts(outcomes)
            row["evaluated"] += evaluated
            row["tp"] += tp
            row["fp"] += fp
            current_flat_ids = candidate_flat_ids[component_selections["current"]]
            selected_flat_ids = candidate_flat_ids[selected]
            added = ~torch.isin(selected_flat_ids, current_flat_ids)
            added_rare = added & rare_selected
            added_outcomes = classify_rare_detections(
                query_boxes[selected_queries[added_rare]],
                selected_scores[added_rare],
                selected_classes[added_rare],
                gt_boxes,
                gt_classes,
                negative_classes=negative_classes,
                not_exhaustive_classes=not_exhaustive_classes,
                iou_threshold=0.5,
                background_iou=0.1,
                initially_matched_gt=current_hits,
            )
            added_counts, added_evaluated, added_tp, added_fp = _outcome_counts(
                added_outcomes
            )
            row["added_candidates"] += len(added_outcomes)
            row["added_evaluated"] += added_evaluated
            row["added_tp"] += added_tp
            row["added_fp"] += added_fp
            for outcome, count in added_counts.items():
                row[f"added_{outcome}"] += int(count)
            gate_hits, _ = greedy_gt_topk_matches(
                overlaps,
                gt_classes,
                candidate_flat_ids[selected],
                num_classes,
                0.5,
            )
            row["rescued_tp"] += int((rare_gt & ~current_hits & gate_hits).sum())
            row["lost_tp"] += int((rare_gt & current_hits & ~gate_hits).sum())
        if (index + 1) % 500 == 0:
            print(f"[screen] loaded {index + 1}/{len(files)}", flush=True)

    rescue_report = finalize_rescue_accumulator(rescue_accumulator)
    rows = finalize_screen(screen, specs)
    selected = choose_gates(
        rows,
        args.max_added_fp_per_net_tp,
        args.max_lost_tp,
        args.evaluate_top,
    )
    rescue_partition = {
        key: int(value) for key, value in rescue_partition.items()
    }
    if sum(
        rescue_partition[key]
        for key in ("detector_only", "vlm_only", "both", "neither")
    ) != rescue_partition["current_miss"]:
        raise RuntimeError("component rescue partition does not cover current misses")
    print_rescue_partition(rescue_partition)
    print_rescue_report(rescue_report)
    print_screen(
        rows, selected, args.max_added_fp_per_net_tp, args.max_lost_tp
    )

    baseline_metrics = load_baseline_metrics(args.baseline_metrics)
    formal_results = {}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def save_report():
        report = {
            "dump_dir": str(dump_dir),
            "screen_iou": 0.5,
            "precision_policy": {
                "max_added_fp_per_net_tp": args.max_added_fp_per_net_tp,
                "max_lost_tp": args.max_lost_tp,
                "evaluate_top": args.evaluate_top,
            },
            "rare_component_rescue_partition": rescue_partition,
            "rescue_distributions": rescue_report,
            "screen": rows,
            "selected_gates": [row["name"] for row in selected],
            "baseline_metrics": baseline_metrics,
            "official_lvis": formal_results,
        }
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[save] {output_path}")

    # Preserve the expensive full-validation screen even if an official LVIS
    # evaluation is later interrupted.
    save_report()
    if selected:
        from lvis import LVIS

        lvis_gt = LVIS(manifest["lvis_json"])
        for row in selected:
            predictions = predictions_for_gate(
                files, profiles, row, novel_mask, args.max_dets
            )
            formal_results[row["name"]] = evaluate_lvis(
                lvis_gt, predictions, args.max_dets
            )
            del predictions
            gc.collect()
            save_report()
        print("\n=== Official LVIS gate results ===")
        print(f"{'gate':>20} " + " ".join(f"{metric:>8}" for metric in METRICS))
        if baseline_metrics is not None:
            print(
                f"{'current_power':>20} "
                + " ".join(f"{baseline_metrics[metric]:8.4f}" for metric in METRICS)
            )
        for name, metrics in formal_results.items():
            print(
                f"{name:>20} "
                + " ".join(f"{metrics[metric]:8.4f}" for metric in METRICS)
            )
            if baseline_metrics is not None:
                print(
                    f"{'delta':>20} AP={metrics['AP'] - baseline_metrics['AP']:+.4f} "
                    f"APr={metrics['APr'] - baseline_metrics['APr']:+.4f}"
                )

    save_report()


if __name__ == "__main__":
    main()
