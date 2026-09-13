"""Join official LVIS matches to saved query coverage without running a model.

Per-instance weights explain *macro recall*, never AP. Optional full-validation
AP differences are joined only at category level as descriptive context.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import math


REASONS = (
    "no_eligible_query",
    "correct_pair_below_topk",
    "matching_competition",
    "candidate_membership_disagreement",
    "candidate_matching_or_postprocess",
)
TRANSITIONS = ("both_hit", "both_miss", "old_hit_new_miss", "old_miss_new_hit")


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _flag(value, name):
    if not isinstance(value, (bool, int)) or value not in (False, True):
        raise ValueError(f"{name} must be boolean")
    return bool(value)


def _required(row, keys, context):
    if not isinstance(row, Mapping):
        raise ValueError(f"{context} must be a mapping")
    missing = set(keys) - row.keys()
    if missing:
        raise ValueError(f"{context} misses required keys: {sorted(missing)}")


def _threshold(value):
    result = _number(value, "IoU threshold")
    if not 0 < result <= 1:
        raise ValueError("IoU threshold must be within (0,1]")
    return result


def _iou_key(value):
    return f"{value:.2f}" if value == round(value, 2) else repr(value)


def _threshold_map(mapping, context):
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{context} must be a mapping")
    result = {}
    for key, value in mapping.items():
        threshold = _threshold(key)
        if threshold in result:
            raise ValueError(f"{context} contains duplicate numeric IoU thresholds")
        result[threshold] = value
    return result


def _categories(category_meta):
    items = list(category_meta.items()) if isinstance(category_meta, Mapping) else [(None, row) for row in category_meta]
    result = {}
    for supplied_id, row in items:
        _required(row, ("name", "frequency"), "category metadata")
        category_id = row.get("id", row.get("category_id", supplied_id))
        category_id = _integer(category_id, "category ID")
        if supplied_id is not None and str(supplied_id) != str(category_id):
            raise ValueError("category metadata key and category ID disagree")
        if "category_id" in row and row["category_id"] != category_id:
            raise ValueError("category metadata id and category_id disagree")
        if category_id in result:
            raise ValueError(f"Duplicate category ID {category_id}")
        if row["frequency"] not in ("r", "c", "f"):
            raise ValueError(f"Invalid frequency for category {category_id}")
        result[category_id] = dict(row, id=category_id)
    if not result:
        raise ValueError("category metadata must not be empty")
    for class_index, category_id in enumerate(sorted(result)):
        row = result[category_id]
        if "class_index" in row and row["class_index"] != class_index:
            raise ValueError(f"Category {category_id} class_index disagrees with sorted global category IDs")
        row["class_index"] = class_index
        for key in ("raw_gt_annotations", "gt_annotations"):
            if key in row and _integer(row[key], f"category {category_id} {key}") < 0:
                raise ValueError("GT annotation counts must be nonnegative")
    return result


def _official_rows(rows, threshold, categories, context):
    result = {}
    seen_ids = set()
    for row in rows:
        _required(row, ("gt_id", "image_id", "category_id", "gt_ignore", "matched",
                        "matched_detection_id", "selected_iou_eligible_count", "selected_candidates"), context)
        gt_id = _integer(row["gt_id"], f"{context} gt_id")
        image_id = _integer(row["image_id"], f"{context} image_id")
        category_id = _integer(row["category_id"], f"{context} category_id")
        identity = (image_id, gt_id)
        detail = f"{context} image={image_id} GT={gt_id}"
        if gt_id in seen_ids:
            raise ValueError(f"Duplicate official GT ID: {detail}")
        seen_ids.add(gt_id)
        if category_id not in categories or categories[category_id]["frequency"] != "r":
            raise ValueError(f"{detail} is not a known rare category")
        ignored = _flag(row["gt_ignore"], f"{detail} gt_ignore")
        matched = _flag(row["matched"], f"{detail} matched")
        matched_id = row["matched_detection_id"]
        if "matched_detection_score" not in row and "matched_score" not in row:
            raise ValueError(f"{detail} misses matched_detection_score")
        matched_score = row.get("matched_detection_score", row.get("matched_score"))
        if matched:
            _integer(matched_id, f"{detail} matched_detection_id")
            matched_score = _number(matched_score, f"{detail} matched_detection_score")
        elif matched_id is not None or matched_score is not None:
            raise ValueError(f"{detail} is unmatched but has a matched detection")
        candidates = row["selected_candidates"]
        count = _integer(row["selected_iou_eligible_count"], f"{detail} selected_iou_eligible_count")
        if not isinstance(candidates, list) or count != len(candidates) or count < 0:
            raise ValueError(f"{detail} selected candidate count disagrees with candidates")
        candidate_ids = set()
        for candidate in candidates:
            _required(candidate, ("detection_id", "score", "iou", "matched_gt_id", "ignored"), detail)
            detection_id = _integer(candidate["detection_id"], f"{detail} candidate detection_id")
            if detection_id in candidate_ids:
                raise ValueError(f"{detail} has duplicate candidate detection IDs")
            candidate_ids.add(detection_id)
            _number(candidate["score"], f"{detail} candidate score")
            iou = _number(candidate["iou"], f"{detail} candidate IoU")
            if not min(threshold, 1 - 1e-10) <= iou <= 1:
                raise ValueError(f"{detail} retained candidate does not meet official IoU threshold")
            _flag(candidate["ignored"], f"{detail} candidate ignored")
            assignment = candidate["matched_gt_id"]
            if assignment is not None:
                _integer(assignment, f"{detail} candidate matched_gt_id")
            if assignment == gt_id and (not matched or detection_id != matched_id):
                raise ValueError(f"{detail} candidate/GT matching is not reciprocal")
        if matched:
            matching = [c for c in candidates if c["detection_id"] == matched_id and c["matched_gt_id"] == gt_id]
            if len(matching) != 1:
                raise ValueError(f"{detail} official match is absent from its eligible candidates; inspect IoU boundary or prediction mismatch")
            if not math.isclose(float(matching[0]["score"]), matched_score, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"{detail} matched score and candidate score disagree")
            if not ignored and matching[0]["ignored"]:
                raise ValueError(f"{detail} nonignored GT match has an ignored detection")
        result[identity] = {
            **row, "gt_ignore": ignored, "matched": matched,
            "matched_detection_score": matched_score,
        }
    return result


def _check_official_summary(summary, rows, context):
    _required(summary, ("num_gt", "true_positives", "false_positives", "macro_recall"), context)
    counts, hits = Counter(), Counter()
    for row in rows.values():
        if not row["gt_ignore"]:
            counts[row["category_id"]] += 1
            hits[row["category_id"]] += int(row["matched"])
    for field, expected in (("num_gt", sum(counts.values())), ("true_positives", sum(hits.values()))):
        if _integer(summary[field], f"{context} {field}") != expected:
            raise ValueError(f"{context} {field}={summary[field]} disagrees with official GT rows ({expected})")
    if _integer(summary["false_positives"], f"{context} false_positives") < 0:
        raise ValueError(f"{context} false_positives must be nonnegative")
    expected_macro = math.fsum(hits[c] / counts[c] for c in counts) / len(counts) if counts else None
    if expected_macro is None:
        if summary["macro_recall"] is not None:
            raise ValueError(f"{context} macro_recall must be null without valid GT")
    elif not math.isclose(_number(summary["macro_recall"], f"{context} macro_recall"), expected_macro, rel_tol=0, abs_tol=1e-10):
        raise ValueError(f"{context} macro_recall disagrees with equal-class official GT rows")
    if "macro_recall_valid_categories" in summary and summary["macro_recall_valid_categories"] != len(counts):
        raise ValueError(f"{context} macro recall category count disagrees with GT rows")
    if "valid_detections" in summary and summary["valid_detections"] != summary["true_positives"] + summary["false_positives"]:
        raise ValueError(f"{context} valid_detections does not equal TP + FP")
    return counts


def _query_rows(rows, thresholds, categories):
    category_ids = sorted(categories)
    result = {}
    for row in rows:
        _required(row, ("variant",), "pairing row")
        if row["variant"] not in ("old_old", "new_new"):
            continue
        _required(row, ("gt_id", "image_id", "class_index", "iou_threshold", "eligible", "pair_topk"), "pairing row")
        threshold = _threshold(row["iou_threshold"])
        if threshold not in thresholds:
            raise ValueError(f"Pairing IoU={threshold} is absent from official matches")
        class_index = _integer(row["class_index"], "pairing class_index")
        if not 0 <= class_index < len(category_ids):
            raise ValueError("Pairing class_index is outside the global vocabulary")
        category_id = category_ids[class_index]
        key = (threshold, _integer(row["image_id"], "pairing image_id"),
               _integer(row["gt_id"], "pairing gt_id"), row["variant"])
        if key in result:
            raise ValueError(f"Duplicate pairing row: {key}")
        eligible = _flag(row["eligible"], "pairing eligible")
        pair_topk = _flag(row["pair_topk"], "pairing pair_topk")
        if pair_topk and not eligible:
            raise ValueError(f"Pairing row has pair_topk but no eligible query: {key}")
        if row.get("best_iou") is not None:
            best_iou = _number(row["best_iou"], "pairing best_iou")
            if not 0 <= best_iou <= 1:
                raise ValueError("Pairing best_iou must be within [0,1]")
        result[key] = {**row, "category_id": category_id, "eligible": eligible, "pair_topk": pair_topk}
    return result


def _missing_reason(official, query, context):
    if official["matched"]:
        if not query["eligible"] or not query["pair_topk"]:
            raise ValueError(f"{context}: official hit contradicts saved eligible/pair_topk; inspect IoU rounding, postprocess coordinates, or mismatched predictions/cache")
        return None
    candidates = official["selected_candidates"]
    if candidates and (not query["eligible"] or not query["pair_topk"]):
        raise ValueError(f"{context}: retained IoU-eligible candidates contradict saved query/top-k coverage; inspect IoU boundary or cache identity")
    if not query["eligible"]:
        return "no_eligible_query"
    if not query["pair_topk"]:
        return "correct_pair_below_topk"
    if not candidates:
        return "candidate_membership_disagreement"
    if any(c["matched_gt_id"] not in (None, 0, official["gt_id"]) for c in candidates):
        return "matching_competition"
    return "candidate_matching_or_postprocess"


def _ap_comparison(ap_reports, categories, raw_counts, valid_categories):
    if ap_reports is None:
        return None
    _required(ap_reports, ("old", "new"), "ap_reports")
    expected_ids = {c for c, row in categories.items() if row["frequency"] == "r"}
    reports = {}
    for label in ("old", "new"):
        report = ap_reports[label]
        _required(report, ("per_class", "official_apr"), f"{label} AP report")
        indexed = {}
        for row in report["per_class"]:
            _required(row, ("category_id", "gt_annotations", "AP"), f"{label} AP category")
            category_id = _integer(row["category_id"], "AP category_id")
            if category_id in indexed:
                raise ValueError(f"Duplicate category {category_id} in {label} AP report")
            if category_id not in expected_ids:
                raise ValueError(f"Unknown/nonrare category {category_id} in {label} AP report")
            meta = categories[category_id]
            expected_count = meta.get("raw_gt_annotations", meta.get("gt_annotations", raw_counts[category_id]))
            if _integer(row["gt_annotations"], "AP gt_annotations") != expected_count:
                raise ValueError(f"{label} AP category {category_id} raw GT count disagrees with annotations")
            if expected_count != raw_counts[category_id]:
                raise ValueError(f"Official panel lacks full rare GT coverage for AP join: category {category_id}")
            if "name" in row and row["name"] != meta["name"]:
                raise ValueError(f"{label} AP category {category_id} name disagrees with metadata")
            if row["AP"] is not None:
                ap = _number(row["AP"], f"{label} AP category {category_id}")
                if not 0 <= ap <= 100:
                    raise ValueError("AP reports must use AP points within [0,100]")
                indexed[category_id] = {**row, "AP": ap}
            else:
                indexed[category_id] = dict(row)
        if set(indexed) != expected_ids:
            raise ValueError(f"{label} AP report does not cover every rare category")
        valid_ids = {c for c, row in indexed.items() if row["AP"] is not None}
        if valid_ids != set(valid_categories):
            raise ValueError(f"{label} valid AP categories disagree with the nonignored rare GT universe")
        if not valid_ids:
            raise ValueError("AP comparison needs at least one valid rare category")
        macro = math.fsum(indexed[c]["AP"] for c in sorted(valid_ids)) / len(valid_ids)
        official = _number(report["official_apr"], f"{label} official_apr")
        if not math.isclose(macro, official, rel_tol=0, abs_tol=1e-5):
            raise ValueError(f"{label} per-class mean AP does not close to official_apr")
        reports[label] = {"classes": indexed, "official_apr": official}
    per_class = []
    for category_id in sorted(expected_ids):
        old_ap, new_ap = (reports[label]["classes"][category_id]["AP"] for label in ("old", "new"))
        delta = new_ap - old_ap if old_ap is not None else None
        per_class.append({
            "category_id": category_id, "name": categories[category_id]["name"],
            "gt_annotations": raw_counts[category_id],
            "old_AP": old_ap, "new_AP": new_ap, "delta_AP": delta,
            "apr_contribution_points": delta / len(valid_categories) if delta is not None else None,
        })
    delta_apr = reports["new"]["official_apr"] - reports["old"]["official_apr"]
    total = math.fsum(row["apr_contribution_points"] for row in per_class if row["apr_contribution_points"] is not None)
    if not math.isclose(total, delta_apr, rel_tol=0, abs_tol=2e-5):
        raise ValueError("Summed AP contributions do not close to reported APr difference")
    return {
        "scope": "Full-validation category AP difference; descriptive join, not an instance/stage allocation of AP",
        "official_apr_old": reports["old"]["official_apr"],
        "official_apr_new": reports["new"]["official_apr"],
        "delta_apr": delta_apr, "valid_category_ids": sorted(valid_categories),
        "closure_error_points": total - delta_apr, "per_class": per_class,
    }


def build_transition_report(old_matches, new_matches, pairing_rows, category_meta, *, ap_reports=None):
    """Attribute native old/new GT hit changes to saved detection stages.

    ``category_meta`` accepts the raw dataset category list or a dict keyed by
    category ID. Optional ``class_index`` must match globally sorted IDs;
    ``gt_annotations``/``raw_gt_annotations`` may supply full raw GT counts.
    Official match inputs contain ``summary_by_iou`` and ``gt_by_iou``. Query
    rows come from the saved pairing report; nonnative variants are ignored.

    Every valid rare class, including classes with zero hits, receives equal
    macro-recall weight. A lost/gained GT contributes -/+100/(Nclasses*GT_c)
    percentage points. Ignored GT remain explicit, with zero contribution, and
    need not have pairing rows. Detection IDs are local to each checkpoint.

    Returns ``summary_by_iou``, ``per_class_by_iou``, ``gt_by_iou`` and optional
    ``ap_comparison``. Contradictory coverage/matching fails explicitly instead
    of assigning an unsupported failure reason, including IoU-boundary errors.
    """
    categories = _categories(category_meta)
    sources, summaries = {}, {}
    for label, matches in (("old", old_matches), ("new", new_matches)):
        _required(matches, ("gt_by_iou", "summary_by_iou"), f"{label} official matches")
        sources[label] = _threshold_map(matches["gt_by_iou"], f"{label} gt_by_iou")
        summaries[label] = _threshold_map(matches["summary_by_iou"], f"{label} summary_by_iou")
        if set(sources[label]) != set(summaries[label]):
            raise ValueError(f"{label} summary and GT IoU thresholds differ")
    thresholds = sorted(sources["old"])
    if not thresholds or set(thresholds) != set(sources["new"]):
        raise ValueError("Old/new official IoU threshold sets must be equal and nonempty")
    queries = _query_rows(pairing_rows, thresholds, categories)
    result = {
        "scope": "Official rare-GT transitions on saved panel predictions; stage contributions explain macro recall, not AP or training causes",
        "weighting": "Each gained/lost nonignored GT contributes +/-100/(valid rare classes * class GT count) macro-recall percentage points",
        "summary_by_iou": {}, "per_class_by_iou": {}, "gt_by_iou": {},
        "ap_comparison": None,
    }
    baseline = None
    used_query_keys = set()
    for threshold in thresholds:
        key = _iou_key(threshold)
        indexed = {label: _official_rows(sources[label][threshold], threshold, categories, f"{label} IoU={key}") for label in ("old", "new")}
        if set(indexed["old"]) != set(indexed["new"]):
            raise ValueError(f"Old/new official GT universes differ at IoU={key}")
        identities = {}
        for identity, old in indexed["old"].items():
            new = indexed["new"][identity]
            if old["category_id"] != new["category_id"] or old["gt_ignore"] != new["gt_ignore"]:
                raise ValueError(f"Old/new category or gt_ignore differs: IoU={key}, GT={identity}")
            identities[identity] = (old["category_id"], old["gt_ignore"])
        if baseline is not None and identities != baseline:
            raise ValueError("Official GT identities/categories/ignore flags change across IoUs")
        baseline = identities
        counts = _check_official_summary(summaries["old"][threshold], indexed["old"], f"old IoU={key}")
        _check_official_summary(summaries["new"][threshold], indexed["new"], f"new IoU={key}")
        class_count = len(counts)
        transitions, losses, gains = Counter(), Counter(), Counter()
        rows, grouped = [], {category_id: [] for category_id in counts}
        for identity in sorted(indexed["old"]):
            old, new = indexed["old"][identity], indexed["new"][identity]
            category_id = old["category_id"]
            states, reasons = {}, {}
            for label, official in (("old", old), ("new", new)):
                query_key = (threshold, *identity, f"{label}_{label}")
                query = queries.get(query_key)
                if query is not None:
                    used_query_keys.add(query_key)
                    if query["category_id"] != category_id:
                        raise ValueError(f"Pairing class_index disagrees with official category: {query_key}")
                if query is None and not official["gt_ignore"]:
                    raise ValueError(f"Missing pairing row for nonignored GT: {query_key}")
                reasons[label] = None if official["gt_ignore"] else _missing_reason(official, query, str(query_key))
                states[label] = {
                    "matched": official["matched"],
                    "matched_detection_id": official["matched_detection_id"],
                    "matched_detection_score": official["matched_detection_score"],
                    "eligible": query["eligible"] if query is not None else None,
                    "pair_topk": query["pair_topk"] if query is not None else None,
                    "best_iou": query.get("best_iou") if query is not None else None,
                    "selected_iou_eligible_count": official["selected_iou_eligible_count"],
                    "selected_candidates": official["selected_candidates"],
                    "miss_reason": reasons[label],
                }
            ignored = old["gt_ignore"]
            transition = "ignored" if ignored else (
                "both_hit" if old["matched"] and new["matched"] else
                "old_hit_new_miss" if old["matched"] else
                "old_miss_new_hit" if new["matched"] else "both_miss"
            )
            sign = 0 if ignored else int(new["matched"]) - int(old["matched"])
            contribution = sign * 100.0 / (class_count * counts[category_id]) if sign else 0.0
            reason = reasons["new"] if sign == -1 else reasons["old"] if sign == 1 else None
            row = {
                "gt_id": identity[1], "image_id": identity[0], "category_id": category_id,
                "name": categories[category_id]["name"], "iou_threshold": threshold,
                "gt_ignore": ignored, "transition": transition, "reason": reason,
                "old_state": states["old"], "new_state": states["new"],
                "macro_recall_contribution_points": contribution,
            }
            rows.append(row)
            if not ignored:
                transitions[transition] += 1
                grouped[category_id].append(row)
                if sign < 0:
                    losses[reason] += 1
                elif sign > 0:
                    gains[reason] += 1
        per_class = []
        for category_id, items in sorted(grouped.items()):
            category_transitions = Counter(row["transition"] for row in items)
            category_losses = Counter(row["reason"] for row in items if row["transition"] == "old_hit_new_miss")
            category_gains = Counter(row["reason"] for row in items if row["transition"] == "old_miss_new_hit")
            old_tp = sum(row["old_state"]["matched"] for row in items)
            new_tp = sum(row["new_state"]["matched"] for row in items)
            per_class.append({
                "category_id": category_id, "name": categories[category_id]["name"],
                "class_index": categories[category_id]["class_index"], "num_gt": len(items),
                "old_tp": old_tp, "new_tp": new_tp, "net_tp_change": new_tp - old_tp,
                "old_recall": old_tp / len(items), "new_recall": new_tp / len(items),
                "losses": category_transitions["old_hit_new_miss"], "gains": category_transitions["old_miss_new_hit"],
                "transitions": {name: category_transitions[name] for name in TRANSITIONS},
                "loss_reason_counts": {name: category_losses[name] for name in REASONS},
                "gain_reason_counts": {name: category_gains[name] for name in REASONS},
                "macro_recall_contribution_points": math.fsum(row["macro_recall_contribution_points"] for row in items),
                "old_AP": None, "new_AP": None, "delta_AP": None, "apr_contribution_points": None,
            })
        old_summary, new_summary = summaries["old"][threshold], summaries["new"][threshold]
        delta = 100.0 * (new_summary["macro_recall"] - old_summary["macro_recall"]) if class_count else None
        loss_weight = math.fsum(row["macro_recall_contribution_points"] for row in rows if row["transition"] == "old_hit_new_miss")
        gain_weight = math.fsum(row["macro_recall_contribution_points"] for row in rows if row["transition"] == "old_miss_new_hit")
        net_tp = new_summary["true_positives"] - old_summary["true_positives"]
        if net_tp != sum(gains.values()) - sum(losses.values()):
            raise ValueError(f"TP transition counts fail to close at IoU={key}")
        closure_error = loss_weight + gain_weight - delta if delta is not None else None
        if closure_error is not None and abs(closure_error) > 1e-8:
            raise ValueError(f"Macro recall contributions fail to close at IoU={key}")
        result["summary_by_iou"][key] = {
            "num_gt": sum(counts.values()), "num_ignored_gt": len(rows) - sum(counts.values()),
            "num_valid_classes": class_count, "valid_category_ids": sorted(counts),
            "old_summary": dict(old_summary), "new_summary": dict(new_summary),
            "transitions": {name: transitions[name] for name in TRANSITIONS},
            "loss_reason_counts": {name: losses[name] for name in REASONS},
            "gain_reason_counts": {name: gains[name] for name in REASONS},
            "net_tp_change": net_tp,
            "net_fp_change": new_summary["false_positives"] - old_summary["false_positives"],
            "macro_recall_change_points": delta,
            "loss_macro_recall_points": loss_weight, "gain_macro_recall_points": gain_weight,
            "loss_reason_macro_recall_points": {name: math.fsum(row["macro_recall_contribution_points"] for row in rows if row["transition"] == "old_hit_new_miss" and row["reason"] == name) for name in REASONS},
            "gain_reason_macro_recall_points": {name: math.fsum(row["macro_recall_contribution_points"] for row in rows if row["transition"] == "old_miss_new_hit" and row["reason"] == name) for name in REASONS},
            "invariants": {"tp_change_equals_gains_minus_losses": True, "macro_recall_closure_error_points": closure_error},
        }
        result["per_class_by_iou"][key] = per_class
        result["gt_by_iou"][key] = rows
    extra = set(queries) - used_query_keys
    if extra:
        raise ValueError(f"Pairing rows contain GT absent from official matching: {sorted(extra)[:3]}")
    raw_counts = Counter(category_id for category_id, _ in baseline.values())
    valid_categories = {category_id for category_id, ignored in baseline.values() if not ignored}
    result["ap_comparison"] = _ap_comparison(ap_reports, categories, raw_counts, valid_categories)
    if result["ap_comparison"] is not None:
        ap_rows = {row["category_id"]: row for row in result["ap_comparison"]["per_class"]}
        for per_class in result["per_class_by_iou"].values():
            for row in per_class:
                row.update({key: ap_rows[row["category_id"]][key] for key in ("old_AP", "new_AP", "delta_AP", "apr_contribution_points")})
    return result
