"""Validate and compare saved ``report_lvis_rare_pr.py`` reports.

AP values and AP deltas use percentage points; raw precision and recall use
fractions. Ranks count only non-ignored detections. Equal-score detections keep
the source evaluator's stable order: ``fp_before`` is an ordering statistic,
not a count of detections with a strictly greater score.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping
import math
from statistics import median


AP_TOLERANCE = 1e-5
CURVE_TOLERANCE = 1e-9
IOU_METRICS = {"0.50": "AP50", "0.75": "AP75"}


def _required(value, keys, context):
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    missing = set(keys) - value.keys()
    if missing:
        raise ValueError(f"{context} misses required keys: {sorted(missing)}")


def _integer(value, context):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a nonnegative integer")
    return value


def _number(value, context):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{context} must be a finite number")
    return float(value)


def _ap(value, context):
    if value is None:
        return None
    value = _number(value, context)
    if not 0 <= value <= 100:
        raise ValueError(f"{context} must be in [0, 100] AP points or null")
    return value


def _same_number(actual, expected, context, tolerance=AP_TOLERANCE):
    if actual is None or expected is None:
        equal = actual is expected
    else:
        equal = math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance)
    if not equal:
        raise ValueError(f"{context} disagrees: {actual!r} versus {expected!r}")


def _interpolated_precision(curve, num_gt):
    if not num_gt:
        return [None] * 101
    envelope = [point["precision"] for point in curve]
    for index in range(len(envelope) - 2, -1, -1):
        envelope[index] = max(envelope[index], envelope[index + 1])
    recall = [point["recall"] for point in curve]
    # Multiplication reproduces np.linspace(0, 1, 101), used by LVIS.
    indices = [bisect_left(recall, index * 0.01) for index in range(101)]
    return [envelope[index] if index < len(envelope) else 0.0 for index in indices]


def summarize_ranked_curve(curve_report):
    """Return every TP's 1-based valid-detection rank and preceding FP count.

    An absent first TP and an undefined AP (no valid GT) are represented by
    ``None``. Present, empty streams with valid GT have AP=0 and max_recall=0.
    The saved cumulative stream is validated without sorting or changing ties.
    """
    count_keys = ("num_gt", "valid_detections", "ignored_detections",
                  "true_positives", "false_positives")
    _required(curve_report, (*count_keys, "curve"), "ranked curve")
    counts = {key: _integer(curve_report[key], key) for key in count_keys}
    curve = curve_report["curve"]
    if not isinstance(curve, list):
        raise ValueError("ranked curve must contain a curve list")
    if counts["valid_detections"] != len(curve):
        raise ValueError("valid_detections disagrees with non-ignored curve length")
    previous_tp = previous_fp = 0
    previous_score = math.inf
    every_tp = []
    has_score_ties = False
    for rank, point in enumerate(curve, 1):
        context = f"ranked curve rank {rank}"
        _required(point, ("score", "tp", "fp", "precision", "recall"), context)
        if any(point.get(key, False) for key in ("ignored", "ignore", "dt_ignore")):
            raise ValueError(f"{context}: ignored detections must be excluded")
        score = _number(point["score"], f"{context} score")
        tp = _integer(point["tp"], f"{context} tp")
        fp = _integer(point["fp"], f"{context} fp")
        if score > previous_score:
            raise ValueError(f"{context}: scores must be descending")
        has_score_ties |= score == previous_score
        if (tp - previous_tp, fp - previous_fp) not in ((1, 0), (0, 1)):
            raise ValueError(f"{context}: cumulative TP/FP must advance exactly one")
        if tp > counts["num_gt"]:
            raise ValueError(f"{context}: TP exceeds num_gt")
        precision = _number(point["precision"], f"{context} precision")
        recall = _number(point["recall"], f"{context} recall")
        _same_number(precision, tp / rank, f"{context} precision denominator",
                     CURVE_TOLERANCE)
        expected_recall = tp / counts["num_gt"] if counts["num_gt"] else 0.0
        _same_number(recall, expected_recall, f"{context} recall denominator",
                     CURVE_TOLERANCE)
        if tp > previous_tp:
            every_tp.append({"tp_index": tp, "rank": rank, "score": score,
                             "fp_before": fp, "precision": precision, "recall": recall})
        previous_tp, previous_fp, previous_score = tp, fp, score
    if (previous_tp, previous_fp) != (counts["true_positives"], counts["false_positives"]):
        raise ValueError("curve terminal TP/FP disagrees with reported totals")

    interpolated = _interpolated_precision(curve, counts["num_gt"])
    if "official_interpolated_pr" in curve_report:
        official = curve_report["official_interpolated_pr"]
        if not isinstance(official, list) or len(official) != 101:
            raise ValueError("official_interpolated_pr must have 101 recall points")
        for index, point in enumerate(official):
            context = f"official_interpolated_pr index {index}"
            _required(point, ("recall", "precision"), context)
            _same_number(_number(point["recall"], f"{context} recall"), index * 0.01,
                         f"{context} recall grid", CURVE_TOLERANCE)
            precision = point["precision"]
            if precision is not None:
                precision = _number(precision, f"{context} precision")
            _same_number(precision, interpolated[index], f"{context} raw/official precision",
                         CURVE_TOLERANCE)
    first = every_tp[0] if every_tp else None
    return {
        **counts,
        "tp": previous_tp,
        "fp": previous_fp,
        "first_tp_rank": first["rank"] if first else None,
        "fp_before_first_tp": first["fp_before"] if first else None,
        "first_tp_score": first["score"] if first else None,
        "every_tp": every_tp,
        "fp_before_tp_median": median(point["fp_before"] for point in every_tp) if every_tp else None,
        "max_recall": previous_tp / counts["num_gt"] if counts["num_gt"] else None,
        "interpolated_AP_points": 100 * sum(interpolated) / 101 if counts["num_gt"] else None,
        "has_score_ties": has_score_ties,
    }


def _validated_report(report, side):
    _required(report, ("max_dets", "official_apr", "rare_category_count", "per_class"), side)
    max_dets = _integer(report["max_dets"], f"{side} max_dets")
    if max_dets == 0:
        raise ValueError(f"{side} max_dets must be positive")
    rows = report["per_class"]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{side} per_class must be a nonempty list")
    count = _integer(report["rare_category_count"], f"{side} rare_category_count")
    if count != len(rows):
        raise ValueError(f"{side} rare_category_count disagrees with per_class count")
    by_name, by_id = {}, {}
    for row in rows:
        context = f"{side} per_class"
        _required(row, ("category_id", "name", "gt_annotations", "AP", "AP50", "AP75"), context)
        category_id = _integer(row["category_id"], f"{context} category_id")
        name = row["name"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"{context} name must be a nonempty string")
        if category_id in by_id or name in by_name:
            raise ValueError(f"{side} has duplicate category ID or name: {name}")
        _integer(row["gt_annotations"], f"{side} {name} gt_annotations")
        for metric in ("AP", "AP50", "AP75"):
            _ap(row[metric], f"{side} {name} {metric}")
        by_name[name] = row
        by_id[category_id] = row
    valid_ap = [row["AP"] for row in rows if row["AP"] is not None]
    if not valid_ap:
        raise ValueError(f"{side} has no valid rare-category AP")
    official_apr = _ap(report["official_apr"], f"{side} official_apr")
    _same_number(official_apr, sum(valid_ap) / len(valid_ap), f"{side} per-class macro APr")
    focus = report.get("focus", {})
    if not isinstance(focus, Mapping):
        raise ValueError(f"{side} focus must be a mapping")
    for name, entry in focus.items():
        if name not in by_name:
            raise ValueError(f"{side} focus category {name!r} absent from per_class")
        if entry is None:
            continue
        _required(entry, ("category_id", "AP", "AP50", "AP75"), f"{side} focus {name}")
        row = by_name[name]
        if entry["category_id"] != row["category_id"]:
            raise ValueError(f"{side} focus {name} category_id disagrees with per_class")
        for key in ("name", "gt_annotations"):
            if key in entry and entry[key] != row[key]:
                raise ValueError(f"{side} focus {name} {key} disagrees with per_class")
        for metric in ("AP", "AP50", "AP75"):
            value = _ap(entry[metric], f"{side} focus {name} {metric}")
            _same_number(value, row[metric], f"{side} focus {name} {metric}")
        if not isinstance(entry.get("iou_curves", {}), Mapping):
            raise ValueError(f"{side} focus {name} iou_curves must be a mapping")
    return by_name, by_id


def compare_reports(old_report, new_report, focus_names=None):
    """Compare requested rare classes after checking the full rare AP macro.

    Missing focus records/IoUs stay ``None`` and make ``complete`` false;
    missing category-level AP metadata is an error. ``focus_names=None``
    selects every rare category in the old report's source order.
    """
    old_names, old_ids = _validated_report(old_report, "old")
    new_names, new_ids = _validated_report(new_report, "new")
    if old_ids.keys() != new_ids.keys():
        raise ValueError("old/new category ID sets disagree")
    if old_report["max_dets"] != new_report["max_dets"]:
        raise ValueError("old/new max_dets disagree")
    for category_id, old_row in old_ids.items():
        new_row = new_ids[category_id]
        for key in ("name", "gt_annotations"):
            if old_row[key] != new_row[key]:
                raise ValueError(f"old/new category {category_id} {key} disagree")
        for metric in ("AP", "AP50", "AP75"):
            if (old_row[metric] is None) != (new_row[metric] is None):
                raise ValueError(f"old/new {old_row['name']} {metric} valid-category masks disagree")
    if focus_names is None:
        focus_names = list(old_names)
    elif isinstance(focus_names, str):
        raise ValueError("focus_names must be a sequence of names, not a string")
    else:
        focus_names = list(focus_names)
    if any(not isinstance(name, str) for name in focus_names):
        raise ValueError("focus_names must contain strings")
    if len(set(focus_names)) != len(focus_names):
        raise ValueError("focus_names must not contain duplicates")
    unknown = [name for name in focus_names if name not in old_names]
    if unknown:
        raise ValueError(f"focus classes are not rare report categories: {unknown}")
    per_class, missing = [], []
    for name in focus_names:
        old_row, new_row = old_names[name], new_names[name]
        row = {key: old_row[key] for key in ("category_id", "name", "gt_annotations")}
        for metric in ("AP", "AP50", "AP75"):
            old_ap, new_ap = old_row[metric], new_row[metric]
            row.update({f"old_{metric}": old_ap, f"new_{metric}": new_ap,
                        f"delta_{metric}": new_ap - old_ap if old_ap is not None and new_ap is not None else None})
        row["curves"] = {}
        gt_counts = []
        for iou, metric in IOU_METRICS.items():
            pair = {}
            for side, report, ap_row in (("old", old_report, old_row), ("new", new_report, new_row)):
                entry = report.get("focus", {}).get(name)
                curve = entry.get("iou_curves", {}).get(iou) if entry else None
                if curve is None:
                    pair[side] = None
                    missing.append({"side": side, "name": name, "iou": iou})
                    continue
                try:
                    summary = summarize_ranked_curve(curve)
                except ValueError as error:
                    raise ValueError(f"{side} {name} IoU {iou}: {error}") from error
                if summary["num_gt"] > row["gt_annotations"]:
                    raise ValueError(f"{side} {name} IoU {iou} num_gt exceeds gt_annotations")
                gt_counts.append(summary["num_gt"])
                _same_number(summary["interpolated_AP_points"], ap_row[metric],
                             f"{side} {name} raw 101-point {metric}")
                pair[side] = summary
            row["curves"][iou] = pair
        if len(set(gt_counts)) > 1:
            raise ValueError(f"old/new or IoU curves for {name} num_gt disagree")
        per_class.append(row)
    return {
        "scope": {
            "evaluation": "LVIS rare categories, bbox, all area",
            "max_dets": old_report["max_dets"],
            "rare_category_count": len(old_names),
            "valid_rare_category_count": sum(row["AP"] is not None for row in old_names.values()),
            "focus_names": focus_names,
            "ap_units": "points (0-100); deltas are new minus old",
            "precision_recall_units": "fractions (0-1)",
            "rank_policy": "1-based non-ignored detections; preserve stable source order for score ties",
            "fp_before_policy": "preceding FP in source order, including tied scores; not strictly higher-score FP",
        },
        "old_apr": old_report["official_apr"],
        "new_apr": new_report["official_apr"],
        "delta_apr": new_report["official_apr"] - old_report["official_apr"],
        "per_class": per_class,
        "missing_curves": missing,
        "complete": not missing,
    }
