#!/usr/bin/env python
"""Analyze rare misses from an existing GT-ROI diagnostic, without inference.

All numbers refer to the 322 validation images selected by the source
diagnostic, not to official LVIS AP. A rank is computed among 1203 classes;
the eligible predicted ROI is the CLIP-best or fused-best IoU-valid query.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


STATUSES = ("current_hit", "current_miss", "no_eligible_box")


def semantic_partition(row, topk=5):
    """Classify whether GT and CLIP-best predicted p3 ROIs rank true class."""
    gt_good = int(row["gt_rank"]) <= topk
    query_good = int(row["clip_best_query_rank"]) <= topk
    if gt_good and query_good:
        return "both_topk"
    if gt_good:
        return "gt_only_topk"
    if query_good:
        return "query_only_topk"
    return "neither_topk"


def inferred_detector_probability(row, *, beta, novel_scale):
    """Invert the actual rare-class power fusion on its fused-best query.

    s = novel_scale * p_det ** (1 - beta) * p_clip ** beta.
    This recovers the *true-class scalar score*, not a detector class rank.
    A zero rounded CLIP probability cannot be inverted reliably. This applies
    to rare classes because the configured LVIS train_norare split treats them
    as novel; the CLI verifies that relation against the class-order files.
    """
    if not 0.0 <= beta < 1.0 or novel_scale <= 0.0:
        raise ValueError("expected rare-class power fusion with 0 <= beta < 1")
    score = float(row["fused_true_score"])
    clip = float(row["fused_query_clip_true_probability"])
    if score <= 0.0 or clip <= 0.0:
        return None
    log_det = (
        math.log(score) - math.log(novel_scale) - beta * math.log(clip)
    ) / (1.0 - beta)
    probability = math.exp(log_det)
    if probability > 1.001:
        raise ValueError(
            "inferred detector probability exceeds 1; source fusion protocol "
            "or stored score is inconsistent"
        )
    return min(probability, 1.0)


def area_bin(row):
    x0, y0, x1, y1 = row["gt_xyxy"]
    area = max(float(x1) - float(x0), 0.0) * max(float(y1) - float(y0), 0.0)
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def iou_bin(row):
    iou = float(row["fused_query_iou"])
    if iou < 0.75:
        return "0.50-0.75"
    if iou < 0.90:
        return "0.75-0.90"
    return "0.90-1.00"


def distribution(values):
    finite = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "p10": float(np.percentile(finite, 10)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
    }


def summarize_group(rows):
    if not rows:
        return {"count": 0}
    eligible = [row for row in rows if row["status"] != "no_eligible_box"]
    misses = [row for row in rows if row["status"] == "current_miss"]
    result = {
        "count": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "miss_fraction": len(misses) / len(rows),
        "miss_clip_best_roi_top5": (
            sum(row["clip_best_query_rank"] <= 5 for row in misses) / len(misses)
            if misses else None
        ),
        "gt_roi_top1": sum(row["gt_rank"] == 1 for row in rows) / len(rows),
        "gt_roi_top5": sum(row["gt_rank"] <= 5 for row in rows) / len(rows),
        "gt_roi_rank": distribution(row["gt_rank"] for row in rows),
        "gt_roi_true_probability": distribution(
            row["gt_true_probability"] for row in rows
        ),
    }
    if eligible:
        result.update(
            {
                "clip_best_roi_top5": sum(
                    row["clip_best_query_rank"] <= 5 for row in eligible
                ) / len(eligible),
                "fused_best_roi_top5": sum(
                    row["fused_query_clip_rank"] <= 5 for row in eligible
                ) / len(eligible),
                "clip_best_roi_rank": distribution(
                    row["clip_best_query_rank"] for row in eligible
                ),
                "fused_best_roi_clip_probability": distribution(
                    row["fused_query_clip_true_probability"] for row in eligible
                ),
                "fused_best_roi_detector_probability": distribution(
                    row["inferred_detector_probability"] for row in eligible
                ),
                "fused_best_roi_iou": distribution(
                    row["fused_query_iou"] for row in eligible
                ),
                "score_over_top300_threshold": distribution(
                    row["score_over_top300_threshold"] for row in eligible
                ),
            }
        )
    return result


def analyze(rows, protocol, *, class_names=None, seen_class_ids=None):
    if not rows or len({(r["image_id"], r["gt_index"]) for r in rows}) != len(rows):
        raise ValueError("expected nonempty, uniquely identified per-GT rows")
    beta = float(protocol["beta"])
    scale = float(protocol["novel_scale"])
    groups = defaultdict(list)
    enriched = []
    for original in rows:
        row = dict(original)
        status = row["status"]
        if status not in STATUSES:
            raise ValueError(f"unexpected status: {status}")
        row["area_bin"] = area_bin(row)
        if status != "no_eligible_box":
            threshold = float(row["image_topk_threshold"])
            if threshold <= 0.0:
                raise ValueError("top-300 threshold must be positive")
            row["score_over_top300_threshold"] = (
                float(row["fused_true_score"]) / threshold
            )
            row["inferred_detector_probability"] = inferred_detector_probability(
                row, beta=beta, novel_scale=scale
            )
            row["semantic_partition"] = semantic_partition(row)
            row["iou_bin"] = iou_bin(row)
        enriched.append(row)
        groups[status].append(row)

    misses = groups["current_miss"]
    if not misses:
        raise ValueError("source contains no localization-valid rare misses")
    # The previous diagnostic observed no one-to-one-only misses. The score
    # ratio should therefore be below 1 for every current_miss.
    above_threshold = sum(
        row["score_over_top300_threshold"] >= 1.0001 for row in misses
    )
    if above_threshold:
        raise ValueError(
            f"{above_threshold} current misses exceed the stored top-300 threshold; "
            "this source requires separate one-to-one matching analysis"
        )

    partitions = Counter(row["semantic_partition"] for row in misses)
    roi_top5_misses = [
        row for row in misses if row["clip_best_query_rank"] <= 5
    ]
    near_threshold = [
        row for row in misses if row["score_over_top300_threshold"] >= 0.8
    ]
    by_area = defaultdict(list)
    by_iou = defaultdict(list)
    by_category = defaultdict(list)
    for row in enriched:
        by_area[row["area_bin"]].append(row)
        by_category[int(row["category_id"])].append(row)
        if row["status"] != "no_eligible_box":
            by_iou[row["iou_bin"]].append(row)

    category_table = []
    for category_id, category_rows in by_category.items():
        category_misses = [r for r in category_rows if r["status"] == "current_miss"]
        if not category_misses:
            continue
        category_table.append(
            {
                "category_id": category_id,
                "name": (
                    class_names[category_id]
                    if class_names is not None and category_id < len(class_names)
                    else None
                ),
                "rare_gt_in_selected_images": len(category_rows),
                "miss_count": len(category_misses),
                "miss_fraction": len(category_misses) / len(category_rows),
                "miss_gt_roi_top5": sum(r["gt_rank"] <= 5 for r in category_misses)
                / len(category_misses),
                "miss_clip_best_roi_top5": sum(
                    r["clip_best_query_rank"] <= 5 for r in category_misses
                ) / len(category_misses),
            }
        )
    category_table.sort(key=lambda x: (-x["miss_count"], x["category_id"]))
    wrong_clip_top1 = Counter(
        int(row["fused_query_clip_top1_category_id"])
        for row in misses
        if int(row["fused_query_clip_top1_category_id"]) != int(row["category_id"])
    )
    wrong_clip_count = sum(wrong_clip_top1.values())

    return {
        "scope": "rare GT in source diagnostic images; not official LVIS AP",
        "fusion_protocol": protocol,
        "status_counts": {status: len(groups[status]) for status in STATUSES},
        "status_summary": {
            status: summarize_group(groups[status]) for status in STATUSES
        },
        "miss_semantic_partition": dict(partitions),
        "miss_semantic_partition_fraction": {
            key: partitions[key] / len(misses)
            for key in ("both_topk", "gt_only_topk", "query_only_topk", "neither_topk")
        },
        "miss_clip_best_roi_top5": summarize_group(roi_top5_misses),
        "miss_near_top300_threshold_ratio_ge_0p8": summarize_group(near_threshold),
        "miss_near_threshold_and_clip_top5": sum(
            row["clip_best_query_rank"] <= 5 for row in near_threshold
        ),
        "by_area": {key: summarize_group(value) for key, value in by_area.items()},
        "by_iou": {key: summarize_group(value) for key, value in by_iou.items()},
        "categories_with_misses": category_table,
        "miss_instances": misses,
        "wrong_clip_top1_on_fused_best_query": [
            {
                "category_id": category_id,
                "name": (
                    class_names[category_id]
                    if class_names is not None and category_id < len(class_names)
                    else None
                ),
                "seen": (
                    category_id in seen_class_ids
                    if seen_class_ids is not None else None
                ),
                "count": count,
            }
            for category_id, count in wrong_clip_top1.most_common(20)
        ],
        "wrong_clip_top1_seen_fraction": (
            sum(
                count for category_id, count in wrong_clip_top1.items()
                if category_id in seen_class_ids
            ) / wrong_clip_count
            if seen_class_ids is not None and wrong_clip_count else None
        ),
        "detector_inversion_unavailable": sum(
            row["inferred_detector_probability"] is None
            for row in enriched
            if row["status"] != "no_eligible_box"
        ),
    }


def print_report(report):
    print("=== Rare ROI-path failure analysis (source images only) ===")
    print("GT counts:", report["status_counts"])
    print("Exact power-fusion inversion recovers true-class detector score, not rank.")
    print(
        "status              N  GT-ROI top5%  query-ROI top5%  "
        "CLIP p-med  det p-med  score/threshold-med"
    )
    for status in STATUSES:
        row = report["status_summary"][status]
        if not row["count"]:
            continue
        if status == "no_eligible_box":
            print(f"{status:>18} {row['count']:4d} {row['gt_roi_top5']*100:13.2f}")
            continue
        print(
            f"{status:>18} {row['count']:4d} {row['gt_roi_top5']*100:13.2f} "
            f"{row['clip_best_roi_top5']*100:15.2f} "
            f"{row['fused_best_roi_clip_probability']['median']:11.5f} "
            f"{row['fused_best_roi_detector_probability'].get('median', float('nan')):10.5f} "
            f"{row['score_over_top300_threshold']['median']:19.3f}"
        )
    print("\nMiss semantic partition at true-class top-5:")
    for name in ("both_topk", "gt_only_topk", "query_only_topk", "neither_topk"):
        count = report["miss_semantic_partition"].get(name, 0)
        fraction = report["miss_semantic_partition_fraction"][name]
        print(f"  {name:20} {count:4d} ({fraction*100:5.1f}%)")
    print(
        "Near top-300 threshold (ratio>=0.8): "
        f"{report['miss_near_top300_threshold_ratio_ge_0p8']['count']} misses, "
        f"{report['miss_near_threshold_and_clip_top5']} with CLIP-best ROI top-5"
    )
    print("\nArea / IoU strata: N, miss%, miss CLIP-best ROI top-5%")
    for axis in ("by_area", "by_iou"):
        for name, row in report[axis].items():
            miss_top5 = row["miss_clip_best_roi_top5"]
            miss_top5_percent = (
                100 * miss_top5 if miss_top5 is not None else float("nan")
            )
            print(
                f"  {axis:7} {name:10} N={row['count']:4d} "
                f"miss={row['miss_fraction']*100:5.1f}% "
                f"miss query-ROI top5={miss_top5_percent:5.1f}%"
            )
    print("\nMost frequent rare classes among misses (counts, not AP):")
    for row in report["categories_with_misses"][:10]:
        label = row["name"] or str(row["category_id"])
        print(
            f"  {label}: miss {row['miss_count']}/{row['rare_gt_in_selected_images']}, "
            f"GT-ROI top5={row['miss_gt_roi_top5']*100:.1f}%, "
            f"query-ROI top5={row['miss_clip_best_roi_top5']*100:.1f}%"
        )
    seen_fraction = report["wrong_clip_top1_seen_fraction"]
    if seen_fraction is not None:
        print(
            "\nAmong wrong CLIP top-1 predictions on fused-best ROIs, "
            f"seen-class fraction={seen_fraction*100:.1f}%"
        )
    print("Most frequent wrong CLIP top-1 categories:")
    for row in report["wrong_clip_top1_on_fused_best_query"][:10]:
        label = row["name"] or str(row["category_id"])
        print(f"  {label}: {row['count']} (seen={row['seen']})")
    print(
        "\nCaveat: rank/top-300 counts are diagnostic coverage, not LVIS precision/AP; "
        "small per-class groups are unstable."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--class-names", default="dataset/lvis/lvis_v1_all_classes.json"
    )
    parser.add_argument(
        "--seen-classes", default="dataset/lvis/lvis_v1_seen_classes.json"
    )
    args = parser.parse_args()
    source = json.loads(Path(args.source_json).read_text(encoding="utf-8"))
    if not source.get("gt_roi_rare_instances"):
        raise ValueError("source JSON must include gt_roi_rare_instances")
    if not math.isclose(float(source.get("min_iou", -1)), 0.5):
        raise ValueError("this report expects the source IoU threshold to be 0.5")
    if int(source.get("rare_recall", {}).get("image_topk", -1)) != 300:
        raise ValueError("this report expects image top-300 selection")
    if len(source["gt_roi_rare_instances"]) != int(
        source["rare_recall"].get("rare_gt", -1)
    ):
        raise ValueError("source per-GT rows disagree with rare-recall count")
    class_names = None
    if args.class_names and Path(args.class_names).is_file():
        class_names = json.loads(Path(args.class_names).read_text(encoding="utf-8"))
        if not isinstance(class_names, list):
            raise ValueError("class-names JSON must be a list")
    if class_names is None or not Path(args.seen_classes).is_file():
        raise ValueError(
            "all/seen LVIS class-order files are required to verify that "
            "rare rows use the novel-class fusion coefficients"
        )
    seen_classes = set(json.loads(Path(args.seen_classes).read_text(encoding="utf-8")))
    seen_class_ids = {
        category_id for category_id, name in enumerate(class_names)
        if name in seen_classes
    }
    rare_ids = {int(row["category_id"]) for row in source["gt_roi_rare_instances"]}
    if any(category_id >= len(class_names) for category_id in rare_ids):
        raise ValueError("source rare category ID exceeds all-class order")
    unexpected_seen = [
        category_id for category_id in rare_ids
        if class_names[category_id] in seen_classes
    ]
    if unexpected_seen:
        raise ValueError(
            f"source rare rows contain seen categories: {unexpected_seen[:10]}"
        )
    report = analyze(
        source["gt_roi_rare_instances"],
        source["fusion_protocol"],
        class_names=class_names,
        seen_class_ids=seen_class_ids,
    )
    report["source_json"] = args.source_json
    report["checkpoint"] = source.get("checkpoint")
    print_report(report)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[save] {output_path}")


if __name__ == "__main__":
    main()
