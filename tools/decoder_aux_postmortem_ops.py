"""Full-validation A/B PR analysis; stdlib-only, not a training-cause attribution."""
from __future__ import annotations

import math
from statistics import median

from tools.rare_pr_comparison_ops import compare_reports, compare_recall_and_ranking, summarize_ranked_curve

IOUS = tuple(f"{i/100:.2f}" for i in range(50, 100, 5))
FOCUS = ("bass_horn", "keg", "lasagna")  # Fixed BEFORE observing the paired outcome.
PARTS = ("lost_recall_support", "gained_recall_support", "shared_recall_precision")
EPS = 1e-5
STRATA = (("1", 1, 1), ("2-4", 2, 4), ("5-9", 5, 9), ("10-19", 10, 19), ("20+", 20, math.inf))


def close(a, b, label):
    if not math.isfinite(a) or not math.isfinite(b) or abs(a-b) > 2*EPS:
        raise ValueError(f"{label} closure failed: {a} vs {b}")


def sign(value):
    return "gain" if value > EPS else "decline" if value < -EPS else "unchanged"


def cohort(rows, denominator):
    deltas = [r["delta_AP"] for r in rows]
    return {"classes": len(rows), "mean_delta_AP": math.fsum(deltas)/len(rows) if rows else None,
            "median_delta_AP": median(deltas) if rows else None,
            "apr_contribution": math.fsum(deltas)/denominator,
            "gains": sum(sign(v) == "gain" for v in deltas),
            "declines": sum(sign(v) == "decline" for v in deltas),
            "unchanged": sum(sign(v) == "unchanged" for v in deltas),
            "partition_contribution": {k: math.fsum(r["ap_partition_points"][k] for r in rows)/denominator
                                       for k in PARTS}}


def representatives(rows):
    """One gain/loss/unchanged example per GT stratum, with deterministic ties.

    These are labeled outcome-selected examples, NOT an unbiased sample. Every
    valid rare category was analyzed before this display-only selection.
    """
    selected = []
    for label, lower, upper in STRATA:
        group = [r for r in rows if lower <= r["gt_annotations"] <= upper]
        for outcome in ("gain", "decline", "unchanged"):
            candidates = [r for r in group if sign(r["delta_AP"]) == outcome]
            candidates.sort(key=lambda r: ((-r["delta_AP"] if outcome == "gain" else r["delta_AP"])
                                           if outcome != "unchanged" else 0., r["category_id"]))
            if candidates:
                selected.append({"gt_range": label, "outcome": outcome, "name": candidates[0]["name"]})
    return selected


def analyze(a, b, *, focus=FOCUS, labels=None):
    # The PR arithmetic is also used for time-separated checkpoints; callers
    # must not inherit the A/B intervention's treatment labels or focus classes.
    labels = dict(labels or {"A": "normal updates", "B": "auxiliary decoder gradient blocked"})
    if set(labels) != {"A", "B"} or any(not isinstance(v, str) or not v.strip() for v in labels.values()):
        raise ValueError("Expected explicit nonempty A/B endpoint labels")
    # Validate complete rare taxonomy, AP mask and official macro; curves follow below.
    base = compare_reports(a, b, [])
    if a["max_dets"] != 300:
        raise ValueError("Expected original image-level max_dets=300")
    for source in (a, b):
        if source.get("curve_scope") != "all_rare_categories":
            raise ValueError("Need all rare curves, not a selected focus-only report")
    rows = []
    n = base["macro_attribution"]["valid_category_count"]
    for ap in base["macro_attribution"]["per_class"]:
        name = ap["name"]
        row = {**ap, "iou": {}}
        raw_gts = []
        for iou in IOUS:
            summaries = {}
            for arm, source in (("A", a), ("B", b)):
                entry = source.get("focus", {}).get(name)
                if entry is None or entry.get("iou_curves", {}).get(iou) is None:
                    raise ValueError(f"Missing {arm} {name} IoU={iou}; missing is not zero FP")
                summary = summarize_ranked_curve(entry["iou_curves"][iou])
                if not 0 < summary["num_gt"] <= ap["gt_annotations"]:
                    raise ValueError(f"Valid AP class has invalid evaluated GT count: {name}")
                raw_gts.append(summary["num_gt"])
                summaries[arm] = summary
            change = compare_recall_and_ranking(summaries["A"], summaries["B"])
            fields = ("num_gt", "tp", "fp", "max_recall", "interpolated_AP_points",
                      "first_tp_rank", "fp_before_first_tp", "fp_before_tp_median", "has_score_ties")
            row["iou"][iou] = {"A": {k: summaries["A"][k] for k in fields},
                                "B": {k: summaries["B"][k] for k in fields}, "change": change}
        if len(set(raw_gts)) != 1:
            raise ValueError(f"A/B or IoU evaluated GT masks differ: {name}")
        row["evaluated_gt"] = raw_gts[0]
        for arm, prefix in (("A", "old"), ("B", "new")):
            mean_ap = math.fsum(row["iou"][x][arm]["interpolated_AP_points"] for x in IOUS)/len(IOUS)
            close(mean_ap, ap[prefix+"_AP"], f"{arm} {name} all-IoU AP")
            for metric, iou in (("AP50", "0.50"), ("AP75", "0.75")):
                close(row["iou"][iou][arm]["interpolated_AP_points"], ap[prefix+"_"+metric], f"{arm} {name} {metric}")
        row["ap_partition_points"] = {k: math.fsum(row["iou"][x]["change"]["ap_partition_points"][k]
                                                   for x in IOUS)/len(IOUS) for k in PARTS}
        close(math.fsum(row["ap_partition_points"].values()), row["delta_AP"], f"{name} partition")
        row["outcome"] = sign(row["delta_AP"])
        rows.append(row)
    total = cohort(rows, n)
    close(total["apr_contribution"], base["delta_apr"], "official delta APr")
    close(math.fsum(total["partition_contribution"].values()), base["delta_apr"], "full APr partition")
    original = [r for r in rows if r["name"] in focus]
    by_name = {r["name"]: r for r in rows}
    original_rows = [{"name": name, "available": name in by_name,
                      "gt_annotations": by_name[name]["gt_annotations"] if name in by_name else None,
                      "A_AP": by_name[name]["old_AP"] if name in by_name else None,
                      "B_AP": by_name[name]["new_AP"] if name in by_name else None,
                      "delta_AP": by_name[name]["delta_AP"] if name in by_name else None,
                      "outcome": by_name[name]["outcome"] if name in by_name else "no_valid_AP"} for name in focus]
    by_iou = {}
    for iou in IOUS:
        changes = [r["iou"][iou]["change"] for r in rows]
        by_iou[iou] = {
            "macro_delta_AP": math.fsum(c["delta_AP_points"] for c in changes)/n,
            "macro_partition": {k: math.fsum(c["ap_partition_points"][k] for c in changes)/n for k in PARTS},
            "classes_with_recall_loss": sum(c["delta_max_recall_points"] < -EPS for c in changes),
            "classes_with_recall_gain": sum(c["delta_max_recall_points"] > EPS for c in changes),
            "classes_with_shared_precision_loss": sum(c["ap_partition_points"]["shared_recall_precision"] < -EPS for c in changes),
            "classes_with_more_fp_before_same_recall_mean": sum(
                c["same_recall_mean_delta_fp_before"] is not None and c["same_recall_mean_delta_fp_before"] > EPS for c in changes),
            "classes_without_shared_tp_ordinal": sum(c["same_recall_mean_delta_fp_before"] is None for c in changes),
        }
    return {"complete": True, "endpoint_labels": labels,
            "A_apr": base["old_apr"], "B_apr": base["new_apr"], "delta_apr": base["delta_apr"],
            "valid_classes": n, "all_rare_categories": a["rare_category_count"],
            "global": total, "positive_apr_contribution": base["macro_attribution"]["positive_contribution"],
            "negative_apr_contribution": base["macro_attribution"]["negative_contribution"],
            "gt_strata": [{"gt_range": label, **cohort([r for r in rows if lo <= r["gt_annotations"] <= hi], n)}
                          for label, lo, hi in STRATA],
            "original_focus": {"classes": original_rows, "cohort": cohort(original, n),
                               "rest": cohort([r for r in rows if r["name"] not in focus], n)},
            "by_iou": by_iou, "representatives": representatives(rows),
            "per_class": rows,
            "scope": [
                "All valid rare classes, all official ten IoUs and original all-image top-300 predictions.",
                f"old=A ({labels['A']}); new=B ({labels['B']}). AP values are percentage points.",
                "Lost/new recall support and shared-recall precision sum EXACTLY to official B-A APr.",
                "Shared precision is not an FP-only causal effect: FP can move up OR TP move down.",
                "FP-before compares TP ordinals at equal recall, not paired GT identities; stable score ties retained.",
                "Recall loss in saved detections does not establish missing raw box proposals.",
                "Equal weight per valid class; GT strata are sensitivity diagnostics, not replacement APr.",
                "Representative classes are outcome-selected for explanation; all classes remain in the analysis.",
                "No automatic loss/layer tuning, training, GPU inference, or global mechanism claim.",
            ]}


def display(result):
    print("\n=== Full-validation A/B rare postmortem (B - A) ===", flush=True)
    print(f"valid={result['valid_classes']} A={result['A_apr']:.4f} B={result['B_apr']:.4f} "
          f"delta_APr={result['delta_apr']:+.4f}")
    g = result["global"]
    print(f"gain/decline/unchanged={g['gains']}/{g['declines']}/{g['unchanged']}; "
          f"positive={result['positive_apr_contribution']:+.4f} negative={result['negative_apr_contribution']:+.4f}")
    print("\n=== Exact all-IoU APr partition ===")
    for k, v in g["partition_contribution"].items():
        print(f"{k}: {v:+.4f}")
    print("Sum equals delta_APr; shared precision is NOT an FP-only causal attribution.")
    print("\n=== GT strata (all valid classes remain in the denominator) ===")
    for s in result["gt_strata"]:
        print(s)
    print("\n=== Original focus vs remaining classes ===")
    for s in result["original_focus"]["classes"]:
        print(s)
    print("original:", result["original_focus"]["cohort"])
    print("rest:", result["original_focus"]["rest"])
    for title, candidates in (("Largest declines", sorted(result["per_class"], key=lambda r: r["delta_AP"])),
                              ("Largest gains", sorted(result["per_class"], key=lambda r: -r["delta_AP"]))):
        print(f"\n=== {title}: class GT AP_A AP_B delta_AP APr_contribution ===")
        desired = "decline" if title == "Largest declines" else "gain"
        for r in [r for r in candidates if r["outcome"] == desired][:15]:
            print(f"{r['name']:25} {r['gt_annotations']:4} {r['old_AP']:.3f} {r['new_AP']:.3f} "
                  f"{r['delta_AP']:+.3f} {r['apr_contribution']:+.4f}")
    by_name = {r["name"]: r for r in result["per_class"]}
    print("\n=== Outcome/GT-stratified examples; not a tuning set ===")
    print("class GT AP_A AP_B delta_AP delta_R50 mean_delta_FP_before50")
    for choice in result["representatives"]:
        r = by_name[choice["name"]]
        c = r["iou"]["0.50"]["change"]
        print(f"{r['name']:25} {r['gt_annotations']:4} {r['old_AP']:.3f} {r['new_AP']:.3f} "
              f"{r['delta_AP']:+.3f} {c['delta_max_recall_points']:+.3f} "
              f"{c['same_recall_mean_delta_fp_before']} ({choice['outcome']}, GT={choice['gt_range']})")
    print("All classes and ten-IoU PR changes are in report.json. No new experiment is auto-selected.", flush=True)
