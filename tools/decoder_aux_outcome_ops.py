"""CPU-only decomposition of a completed decoder auxiliary-gradient A/B trial."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from statistics import mean, median


IOU_KEYS = ("0.50", "0.75")


def box_iou_xywh(a, b):
    if len(a) != 4 or len(b) != 4:
        raise ValueError("bbox must contain four xywh values")
    ax1, ay1, aw, ah = map(float, a)
    bx1, by1, bw, bh = map(float, b)
    if min(aw, ah, bw, bh) < 0 or not all(math.isfinite(v) for v in (*a, *b)):
        raise ValueError("bbox must be finite with nonnegative width/height")
    ax2, ay2, bx2, by2 = ax1+aw, ay1+ah, bx1+bw, by1+bh
    overlap = max(0., min(ax2, bx2)-max(ax1, bx1))*max(0., min(ay2, by2)-max(ay1, by1))
    union = aw*ah+bw*bh-overlap
    return overlap/union if union > 0 else 0.


def _index_gt(rows, label):
    result = {}
    for row in rows:
        key = row["gt_id"]
        if key in result:
            raise ValueError(f"Duplicate {label} GT ID")
        result[key] = row
    return result


def _missing_reason(row, gt, predictions, threshold):
    if row["selected_iou_eligible_count"]:
        return "correct_class_iou_candidate_unmatched"
    if any(box_iou_xywh(gt["bbox"], prediction["bbox"]) >= threshold
           for prediction in predictions.get(gt["image_id"], ())):
        return "geometry_present_true_class_absent_from_final_top300"
    return "no_iou_eligible_detection_in_final_top300"


def _full_validation_pr_summary(row, key):
    """Compact one-class PR evidence without duplicating full saved curves."""
    change = row["recall_ranking"][key]
    sides = row["curves"][key]
    if change is None or sides["old"] is None or sides["new"] is None:
        return None
    old, new = sides["old"], sides["new"]
    parts = change["ap_partition_points"]
    return {
        "delta_AP_points": change["delta_AP_points"],
        "delta_max_recall_points": change["delta_max_recall_points"],
        "shared_recall_precision_points": parts["shared_recall_precision"],
        "lost_recall_support_points": parts["lost_recall_support"],
        "gained_recall_support_points": parts["gained_recall_support"],
        "same_recall_mean_delta_fp_before": change["same_recall_mean_delta_fp_before"],
        "A_tp": old["true_positives"], "B_tp": new["true_positives"],
        "delta_tp": new["true_positives"] - old["true_positives"],
        "A_fp": old["false_positives"], "B_fp": new["false_positives"],
        "delta_fp": new["false_positives"] - old["false_positives"],
    }


def build_top300_transitions(old_matches, new_matches, dataset, old_predictions, new_predictions,
                             ap_comparison):
    """Official rare-GT transitions; reasons are limited to final top-300 evidence."""
    annotations = {row["id"]: row for row in dataset["annotations"]}
    categories = {row["id"]: row for row in dataset["categories"]}
    prediction_index = {}
    for label, rows in (("A", old_predictions), ("B", new_predictions)):
        by_image = defaultdict(list)
        for row in rows:
            by_image[row["image_id"]].append(row)
        if any(len(rows) > 300 for rows in by_image.values()):
            raise ValueError(f"{label} final prediction JSON exceeds top-300 per image")
        prediction_index[label] = by_image
    ap_rows = {row["category_id"]: row for row in ap_comparison["macro_attribution"]["per_class"]}
    curve_rows = {row["category_id"]: row for row in ap_comparison["per_class"]}
    result = {
        "scope": ("Official LVIS one-to-one rare-GT transitions on every rare-GT image. "
                  "FP totals here exclude negative-control images; full-validation FP/ranking effects are in PR comparison."),
        "reason_scope": ("Final global top-300 detections only; cannot distinguish missing proposals "
                         "from true-class pairs discarded before top-300."),
        "summary_by_iou": {}, "declining_class_summary_by_iou": {},
        "per_class_by_iou": {}, "gt_by_iou": {},
    }
    previous_universe = None
    for key in IOU_KEYS:
        threshold = float(key)
        old = _index_gt(old_matches["gt_by_iou"][key], "A")
        new = _index_gt(new_matches["gt_by_iou"][key], "B")
        if old.keys() != new.keys():
            raise ValueError("A/B official GT universes differ")
        universe = {(row["gt_id"], row["category_id"], row["gt_ignore"]) for row in old.values()}
        if previous_universe is not None and universe != previous_universe:
            raise ValueError("Official GT universe changes across IoUs")
        previous_universe = universe
        transitions, loss_reasons, gain_reasons = Counter(), Counter(), Counter()
        grouped, details = defaultdict(list), []
        for gt_id in sorted(old):
            before, after = old[gt_id], new[gt_id]
            if before["category_id"] != after["category_id"] or before["gt_ignore"] != after["gt_ignore"]:
                raise ValueError("A/B category/ignore flags differ")
            ignored = before["gt_ignore"]
            transition = "ignored" if ignored else (
                "both_hit" if before["matched"] and after["matched"] else
                "A_hit_B_miss" if before["matched"] else
                "A_miss_B_hit" if after["matched"] else "both_miss")
            reason = None
            gt = annotations[gt_id]
            if transition == "A_hit_B_miss":
                reason = _missing_reason(after, gt, prediction_index["B"], threshold)
                loss_reasons[reason] += 1
            elif transition == "A_miss_B_hit":
                reason = _missing_reason(before, gt, prediction_index["A"], threshold)
                gain_reasons[reason] += 1
            if not ignored:
                transitions[transition] += 1
                grouped[before["category_id"]].append((transition, reason))
            details.append({"gt_id": gt_id, "image_id": before["image_id"],
                            "category_id": before["category_id"],
                            "name": categories[before["category_id"]]["name"],
                            "gt_ignore": ignored, "transition": transition, "reason": reason,
                            "A": before, "B": after})
        per_class = []
        for category_id, rows in sorted(grouped.items()):
            counts = Counter(t for t, _ in rows)
            losses = Counter(r for t, r in rows if t == "A_hit_B_miss")
            gains = Counter(r for t, r in rows if t == "A_miss_B_hit")
            if category_id not in curve_rows:
                raise ValueError("Rare GT category absent from all-class PR comparison")
            per_class.append({"category_id": category_id, "name": categories[category_id]["name"],
                              "gt": len(rows), "A_tp": counts["both_hit"]+counts["A_hit_B_miss"],
                              "B_tp": counts["both_hit"]+counts["A_miss_B_hit"],
                              "lost": counts["A_hit_B_miss"], "gained": counts["A_miss_B_hit"],
                              "loss_reasons": dict(losses), "gain_reasons": dict(gains),
                              "full_validation_pr": _full_validation_pr_summary(
                                  curve_rows[category_id], key),
                              **ap_rows.get(category_id, {})})
        a_summary, b_summary = old_matches["summary_by_iou"][key], new_matches["summary_by_iou"][key]
        if b_summary["true_positives"]-a_summary["true_positives"] != (
                transitions["A_miss_B_hit"]-transitions["A_hit_B_miss"]):
            raise ValueError("GT transition TP closure failed")
        result["summary_by_iou"][key] = {
            "valid_gt": sum(transitions.values()), "transitions": dict(transitions),
            "loss_reasons": dict(loss_reasons), "gain_reasons": dict(gain_reasons),
            "A_tp": a_summary["true_positives"], "B_tp": b_summary["true_positives"],
            "delta_tp": b_summary["true_positives"]-a_summary["true_positives"],
            "A_fp": a_summary["false_positives"], "B_fp": b_summary["false_positives"],
            "delta_fp": b_summary["false_positives"]-a_summary["false_positives"],
        }
        declining = [row for row in per_class if row.get("delta_AP", 0) < 0]
        pr_rows = [row["full_validation_pr"] for row in declining
                   if row["full_validation_pr"] is not None]
        reason_gt = Counter()
        reason_classes = Counter()
        for row in declining:
            reason_gt.update(row["loss_reasons"])
            reason_classes.update(row["loss_reasons"].keys())
        result["declining_class_summary_by_iou"][key] = {
            "scope": ("Non-exclusive diagnostics for classes with official IoU-averaged delta AP < 0. "
                      "Recall, shared-support precision and FP-ordering effects may overlap."),
            "classes": len(declining),
            "A_hit_B_miss_gt": sum(row["lost"] for row in declining),
            "A_miss_B_hit_gt": sum(row["gained"] for row in declining),
            "lost_gt_reasons": dict(reason_gt),
            "classes_with_lost_gt_reason": dict(reason_classes),
            "classes_with_lower_max_recall": sum(
                row["delta_max_recall_points"] < 0 for row in pr_rows),
            "classes_with_shared_recall_precision_loss": sum(
                row["shared_recall_precision_points"] < 0 for row in pr_rows),
            "classes_with_more_fp_before_same_recall_tp": sum(
                row["same_recall_mean_delta_fp_before"] is not None
                and row["same_recall_mean_delta_fp_before"] > 0 for row in pr_rows),
            "classes_with_more_total_full_validation_fp": sum(
                row["delta_fp"] > 0 for row in pr_rows),
        }
        result["per_class_by_iou"][key] = per_class
        result["gt_by_iou"][key] = details
    return result


def summarize_pr(comparison):
    result = {}
    for key in IOU_KEYS:
        partition = Counter()
        more = fewer = equal = valid = 0
        for row in comparison["per_class"]:
            change = row["recall_ranking"][key]
            if change is None:
                continue
            valid += 1
            for name, value in change["ap_partition_points"].items():
                partition[name] += value
            value = change["same_recall_mean_delta_fp_before"]
            if value is not None:
                more += value > 0
                fewer += value < 0
                equal += value == 0
        denominator = comparison["macro_attribution"]["valid_category_count"]
        macro_partition = {k: v/denominator for k, v in partition.items()}
        metric = "delta_AP50" if key == "0.50" else "delta_AP75"
        expected = math.fsum(row[metric] for row in comparison["macro_attribution"]["per_class"])/denominator
        if not math.isclose(math.fsum(macro_partition.values()), expected, rel_tol=0, abs_tol=2e-5):
            raise ValueError(f"IoU={key} recall/ranking AP partition does not close")
        result[key] = {
            "valid_curve_comparisons": valid,
            "macro_delta_AP_points": expected,
            "macro_delta_partition_points": macro_partition,
            "closure_error_points": math.fsum(macro_partition.values())-expected,
            "classes_with_more_fps_before_same_recall_tp": more,
            "classes_with_fewer_fps_before_same_recall_tp": fewer,
            "classes_with_equal_fps_before_same_recall_tp": equal,
            "scope": "IoU-specific AP50/AP75 decomposition; not the 10-IoU official APr decomposition",
        }
    return result


def distribution(values):
    if not values or any(not math.isfinite(float(v)) for v in values):
        raise ValueError("Distribution needs finite values")
    ordered = sorted(map(float, values))
    def percentile(q):
        position = q*(len(ordered)-1)
        lo, hi = math.floor(position), math.ceil(position)
        return ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo)
    return {"n": len(ordered), "min": ordered[0], "p05": percentile(.05),
            "median": median(ordered), "mean": mean(ordered), "p95": percentile(.95),
            "max": ordered[-1]}


def summarize_gradient_logs(logs):
    result = {}
    for arm, ranks in logs.items():
        if not ranks or len({len(rows) for rows in ranks}) != 1:
            raise ValueError("Gradient logs have incomplete rank coverage")
        if any(float(row["full_detector_norm"]) <= 0 or float(row["aux_norm"]) < 0
               or float(row["core_post_norm"]) < 0 or not 0 < float(row["clip_coefficient"]) <= 1
               for rank in ranks for row in rank):
            raise ValueError("Gradient logs contain invalid norms or clipping coefficients")
        updates = len(ranks[0])
        rows = []
        for index in range(updates):
            step = [rank[index] for rank in ranks]
            if len({row["iteration"] for row in step}) != 1:
                raise ValueError("Gradient logs disagree on iteration")
            rows.append({key: mean(float(row[key]) for row in step)
                         for key in ("full_detector_norm", "clip_coefficient", "aux_norm", "core_post_norm")})
        result[arm] = {key: distribution([row[key] for row in rows]) for key in rows[0]}
        result[arm]["aux_over_full_norm"] = distribution([
            row["aux_norm"]/row["full_detector_norm"] for row in rows])
    return {"by_arm": result,
            "correlation_scope": "Global per-step norms have no category axis; per-class delta-AP correlation is not identifiable.",
            "causal_scope": "Norms describe intervention magnitude only; they do not attribute category outcomes to a loss source."}
