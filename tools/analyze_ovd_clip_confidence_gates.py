#!/usr/bin/env python
"""Screen CLIP top-1 confidence gates using an existing rescue cache.

Only novel candidates whose CLIP class is the query-level top-1 and whose
probability and top-1/top-2 margin pass a gate are allowed to receive the
isolated CLIP score.  The detector branch is never boosted.  The fixed-IoU
screen evaluates all confidence gates cheaply and official LVIS evaluation is
restricted to at most ``--evaluate-top`` Pareto survivors.

The cache stores CLIP top-2 summaries for every query.  Therefore a qualifying
top-1 pair can be recovered even when it is absent from the sparse candidate
union.  If such a pair's branch score is below the current top-k threshold it
cannot enter the gated top-k; otherwise the branch score is exact because its
unknown current score is bounded by that threshold.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_ovd_error_decomposition import (  # noqa: E402
    category_frequency,
    classify_rare_detections,
    normalized_cxcywh_to_xyxy,
    pairwise_iou,
    xywh_to_xyxy,
)
from tools.analyze_ovd_rescue_gates import (  # noqa: E402
    METRICS,
    _outcome_counts,
    choose_gates,
    finalize_screen,
    load_baseline_metrics,
    load_tensor_file,
    print_screen,
    profile_log_scores,
)
from tools.diagnose_tpa_usage import greedy_gt_topk_matches  # noqa: E402
from tools.evaluate_ovd_fusion import (  # noqa: E402
    evaluate_lvis,
    normalized_boxes_to_lvis,
)


def _token(value):
    return f"{float(value):g}".replace(".", "p")


def build_clip_gate_specs(probabilities, margins, vlm_multiplier):
    specs = [
        {
            "name": "gate_d0_v0",
            "detector_multiplier": 0.0,
            "vlm_multiplier": 0.0,
            "min_probability": None,
            "min_margin": None,
        }
    ]
    seen = set()
    for probability in probabilities:
        for margin in margins:
            # Since p_top2 >= 0, margin=p_top1-p_top2 can never exceed p_top1.
            # Canonicalizing avoids formally evaluating equivalent gates such
            # as (p>=0.8, margin>=0.9) and (p>=0.9, margin>=0.9).
            effective_probability = max(float(probability), float(margin))
            key = (effective_probability, float(margin))
            if key in seen:
                continue
            seen.add(key)
            specs.append(
                {
                    "name": (
                        f"clip_p{_token(effective_probability)}_m{_token(margin)}"
                    ),
                    "detector_multiplier": 0.0,
                    "vlm_multiplier": float(vlm_multiplier),
                    "min_probability": effective_probability,
                    "min_margin": float(margin),
                }
            )
    return specs


def current_selection(payload, current_scores, num_classes, max_dets):
    query_ids = payload["candidate_query_ids"].long()
    class_ids = payload["candidate_class_ids"].long()
    count = min(int(max_dets), current_scores.numel())
    selected = current_scores.topk(count).indices
    return {
        "query_ids": query_ids[selected],
        "class_ids": class_ids[selected],
        "flat_ids": query_ids[selected] * int(num_classes) + class_ids[selected],
        "scores": current_scores[selected],
        "recovered_outside_sparse_pool": 0,
    }


def clip_confidence_gate_selection(
    payload,
    current_scores,
    novel_mask,
    *,
    min_probability,
    min_margin,
    vlm_multiplier,
    novel_scale,
    max_dets,
):
    """Return the exact gated top-k, including missing query-level top-1 pairs."""
    if not 0.0 <= min_probability <= 1.0:
        raise ValueError("min_probability must be in [0,1]")
    if not -1.0 <= min_margin <= 1.0:
        raise ValueError("min_margin must be in [-1,1]")
    if vlm_multiplier <= 0.0 or novel_scale <= 0.0:
        raise ValueError("VLM multiplier and novel scale must be positive")

    summary = payload.get("component_query_summary")
    if summary is None:
        raise ValueError(
            "cache lacks component_query_summary; use the rescue-gate raw cache"
        )
    top_probabilities = summary["vlm_top_probabilities"].float()
    top_classes = summary["vlm_top_classes"].long()
    if (
        top_probabilities.ndim != 2
        or top_classes.shape != top_probabilities.shape
        or top_probabilities.shape[1] < 2
    ):
        raise ValueError("VLM query summary must contain aligned top-2 tensors")

    num_classes = novel_mask.numel()
    sparse_queries = payload["candidate_query_ids"].long()
    sparse_classes = payload["candidate_class_ids"].long()
    sparse_flat_ids = sparse_queries * num_classes + sparse_classes
    if sparse_flat_ids.numel() != current_scores.numel():
        raise ValueError("current scores do not match the sparse candidate pool")
    if sparse_flat_ids.numel() and not bool(
        torch.all(sparse_flat_ids[1:] >= sparse_flat_ids[:-1])
    ):
        raise ValueError("sparse candidate ids must be sorted")

    top_probability = top_probabilities[:, 0]
    top_margin = top_probability - top_probabilities[:, 1]
    top_class = top_classes[:, 0]
    qualified = (
        novel_mask[top_class]
        & (top_probability >= float(min_probability))
        & (top_margin >= float(min_margin))
    )
    query_range = torch.arange(top_class.numel(), dtype=torch.long)
    top_flat_ids = query_range * num_classes + top_class
    positions = torch.searchsorted(sparse_flat_ids, top_flat_ids)
    present = positions < sparse_flat_ids.numel()
    if present.any():
        present_indices = torch.nonzero(present, as_tuple=False).flatten()
        present[present_indices] &= (
            sparse_flat_ids[positions[present_indices]]
            == top_flat_ids[present_indices]
        )

    log_factor = math.log(float(vlm_multiplier) * float(novel_scale))
    branch_scores = top_probability.clamp_min(1e-30).log() + log_factor
    gate_scores = current_scores.clone()
    update_queries = torch.nonzero(qualified & present, as_tuple=False).flatten()
    if update_queries.numel():
        update_positions = positions[update_queries]
        gate_scores[update_positions] = torch.maximum(
            gate_scores[update_positions], branch_scores[update_queries]
        )

    # Only a missing pair above the unchanged current kth score can possibly
    # enter the final top-k.  For such a pair branch > current is guaranteed.
    baseline_count = min(int(max_dets), current_scores.numel())
    if baseline_count < 1:
        raise ValueError("candidate pool must be non-empty")
    current_threshold = current_scores.topk(baseline_count).values[-1]
    recovered_queries = torch.nonzero(
        qualified & ~present & (branch_scores > current_threshold),
        as_tuple=False,
    ).flatten()

    all_queries = torch.cat((sparse_queries, recovered_queries))
    all_classes = torch.cat((sparse_classes, top_class[recovered_queries]))
    all_flat_ids = torch.cat((sparse_flat_ids, top_flat_ids[recovered_queries]))
    all_scores = torch.cat((gate_scores, branch_scores[recovered_queries]))
    count = min(int(max_dets), all_scores.numel())
    selected = all_scores.topk(count).indices
    return {
        "query_ids": all_queries[selected],
        "class_ids": all_classes[selected],
        "flat_ids": all_flat_ids[selected],
        "scores": all_scores[selected],
        "recovered_outside_sparse_pool": int(recovered_queries.numel()),
    }


def predictions_for_gate(
    files,
    current_profile,
    spec,
    novel_mask,
    novel_scale,
    max_dets,
):
    predictions = []
    for index, path in enumerate(files):
        payload = load_tensor_file(path)
        current_scores = profile_log_scores(
            payload, current_profile, novel_mask, path
        )
        selected = clip_confidence_gate_selection(
            payload,
            current_scores,
            novel_mask,
            min_probability=spec["min_probability"],
            min_margin=spec["min_margin"],
            vlm_multiplier=spec["vlm_multiplier"],
            novel_scale=novel_scale,
            max_dets=max_dets,
        )
        boxes = normalized_boxes_to_lvis(
            payload["query_boxes"][selected["query_ids"]].float(),
            int(payload["width"]),
            int(payload["height"]),
        )
        valid = (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
        for box, score, class_id in zip(
            boxes[valid].tolist(),
            selected["scores"][valid].exp().tolist(),
            selected["class_ids"][valid].tolist(),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--max-dets", type=int, default=300)
    parser.add_argument(
        "--probabilities", type=float, nargs="+", default=[0.8, 0.9, 0.95]
    )
    parser.add_argument("--margins", type=float, nargs="+", default=[0.7, 0.8, 0.9])
    parser.add_argument("--vlm-multiplier", type=float, default=0.1)
    parser.add_argument("--max-added-fp-per-net-tp", type=float, default=1.0)
    parser.add_argument("--max-lost-tp", type=int, default=5)
    parser.add_argument("--evaluate-top", type=int, default=3)
    parser.add_argument("--baseline-metrics", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.max_dets < 1 or args.evaluate_top < 0:
        raise ValueError("max-dets must be positive and evaluate-top non-negative")
    if args.vlm_multiplier <= 0.0:
        raise ValueError("vlm-multiplier must be positive")
    if any(not 0.0 <= value <= 1.0 for value in args.probabilities):
        raise ValueError("probability thresholds must be in [0,1]")
    if any(not -1.0 <= value <= 1.0 for value in args.margins):
        raise ValueError("margin thresholds must be in [-1,1]")

    dump_dir = Path(args.dump_dir)
    manifest = json.loads((dump_dir / "manifest.json").read_text(encoding="utf-8"))
    profiles = {profile["name"]: profile for profile in manifest["profiles"]}
    if "current_power" not in profiles or "vlm_scaled" not in profiles:
        raise ValueError("cache lacks current_power or vlm_scaled profile")
    if args.max_dets > int(manifest["topk_per_profile"]):
        raise ValueError("max-dets exceeds the exact cached top-k")
    files = sorted(dump_dir.glob("raw_*.pth"))
    if len(files) != int(manifest["num_dataset_images"]):
        raise RuntimeError("raw rescue cache is incomplete")

    annotations = json.loads(
        Path(manifest["lvis_json"]).read_text(encoding="utf-8")
    )
    categories = {int(item["id"]): item for item in annotations["categories"]}
    frequencies = {
        category_id: category_frequency(category)
        for category_id, category in categories.items()
    }
    annotations_by_image = defaultdict(list)
    for annotation in annotations["annotations"]:
        if annotation.get("iscrowd", 0):
            continue
        if annotation["bbox"][2] <= 0 or annotation["bbox"][3] <= 0:
            continue
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    images_by_id = {int(image["id"]): image for image in annotations["images"]}

    num_classes = max(max(categories), max(manifest["novel_class_ids"]) + 1)
    novel_mask = torch.zeros(num_classes, dtype=torch.bool)
    novel_mask[torch.tensor(manifest["novel_class_ids"], dtype=torch.long)] = True
    novel_scale = float(profiles["vlm_scaled"]["novel_scale"])
    specs = build_clip_gate_specs(
        args.probabilities, args.margins, args.vlm_multiplier
    )
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
                "recovered_outside_sparse_pool": 0,
            }
        )
        for spec in specs
    }

    for index, path in enumerate(files):
        payload = load_tensor_file(path)
        current_scores = profile_log_scores(
            payload, profiles["current_power"], novel_mask, path
        )
        baseline = current_selection(
            payload, current_scores, num_classes, args.max_dets
        )
        query_boxes = normalized_cxcywh_to_xyxy(
            payload["query_boxes"], int(payload["width"]), int(payload["height"])
        )
        image_id = int(payload["image_id"])
        image_annotations = annotations_by_image.get(image_id, [])
        if image_annotations:
            gt_boxes = xywh_to_xyxy(
                torch.tensor(
                    [item["bbox"] for item in image_annotations],
                    dtype=torch.float32,
                )
            )
            gt_classes = torch.tensor(
                [int(item["category_id"]) - 1 for item in image_annotations],
                dtype=torch.long,
            )
        else:
            gt_boxes = query_boxes.new_empty((0, 4))
            gt_classes = torch.empty(0, dtype=torch.long)
        overlaps = pairwise_iou(gt_boxes, query_boxes)
        rare_gt = torch.tensor(
            [frequencies[int(class_id) + 1] == "r" for class_id in gt_classes],
            dtype=torch.bool,
        )
        current_hits, _ = greedy_gt_topk_matches(
            overlaps,
            gt_classes,
            baseline["flat_ids"],
            num_classes,
            0.5,
        )
        metadata = images_by_id[image_id]
        negative_classes = {
            int(value) - 1 for value in metadata.get("neg_category_ids", [])
        }
        not_exhaustive_classes = {
            int(value) - 1
            for value in metadata.get("not_exhaustive_category_ids", [])
        }

        for spec in specs:
            if spec["vlm_multiplier"] == 0.0:
                selected = baseline
            else:
                selected = clip_confidence_gate_selection(
                    payload,
                    current_scores,
                    novel_mask,
                    min_probability=spec["min_probability"],
                    min_margin=spec["min_margin"],
                    vlm_multiplier=spec["vlm_multiplier"],
                    novel_scale=novel_scale,
                    max_dets=args.max_dets,
                )
            selected_classes = selected["class_ids"]
            selected_queries = selected["query_ids"]
            selected_scores = selected["scores"].exp()
            row = screen[spec["name"]]
            row["recovered_outside_sparse_pool"] += selected[
                "recovered_outside_sparse_pool"
            ]
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

            added = ~torch.isin(selected["flat_ids"], baseline["flat_ids"])
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
                selected["flat_ids"],
                num_classes,
                0.5,
            )
            row["rescued_tp"] += int(
                (rare_gt & ~current_hits & gate_hits).sum()
            )
            row["lost_tp"] += int(
                (rare_gt & current_hits & ~gate_hits).sum()
            )
        if (index + 1) % 500 == 0:
            print(f"[confidence-screen] loaded {index + 1}/{len(files)}", flush=True)

    rows = finalize_screen(screen, specs)
    for row, spec in zip(rows, specs):
        row["min_probability"] = spec["min_probability"]
        row["min_margin"] = spec["min_margin"]
        row["recovered_outside_sparse_pool"] = int(
            screen[spec["name"]]["recovered_outside_sparse_pool"]
        )
    selected = choose_gates(
        rows,
        args.max_added_fp_per_net_tp,
        args.max_lost_tp,
        args.evaluate_top,
    )
    print("\n=== CLIP top-1 confidence policy ===")
    print(f"vlm_multiplier: {args.vlm_multiplier}")
    print(f"novel_scale: {novel_scale}")
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
            "vlm_multiplier": args.vlm_multiplier,
            "novel_scale": novel_scale,
            "precision_policy": {
                "max_added_fp_per_net_tp": args.max_added_fp_per_net_tp,
                "max_lost_tp": args.max_lost_tp,
                "evaluate_top": args.evaluate_top,
            },
            "screen": rows,
            "selected_gates": [row["name"] for row in selected],
            "baseline_metrics": baseline_metrics,
            "official_lvis": formal_results,
        }
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[save] {output_path}")

    save_report()
    if selected:
        from lvis import LVIS

        lvis_gt = LVIS(manifest["lvis_json"])
        for row in selected:
            predictions = predictions_for_gate(
                files,
                profiles["current_power"],
                row,
                novel_mask,
                novel_scale,
                args.max_dets,
            )
            formal_results[row["name"]] = evaluate_lvis(
                lvis_gt, predictions, args.max_dets
            )
            del predictions
            gc.collect()
            save_report()
        print("\n=== Official LVIS confidence-gate results ===")
        print(f"{'gate':>24} " + " ".join(f"{metric:>8}" for metric in METRICS))
        if baseline_metrics is not None:
            print(
                f"{'current_power':>24} "
                + " ".join(
                    f"{baseline_metrics[metric]:8.4f}" for metric in METRICS
                )
            )
        for name, metrics in formal_results.items():
            print(
                f"{name:>24} "
                + " ".join(f"{metrics[metric]:8.4f}" for metric in METRICS)
            )
            if baseline_metrics is not None:
                print(
                    f"{'delta':>24} "
                    f"AP={metrics['AP'] - baseline_metrics['AP']:+.4f} "
                    f"APr={metrics['APr'] - baseline_metrics['APr']:+.4f}"
                )
    save_report()


if __name__ == "__main__":
    main()
