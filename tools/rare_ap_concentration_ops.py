"""All-valid-class AP attribution; no detector, training, or threshold tuning."""
from __future__ import annotations

import math
from statistics import median

from tools.rare_pr_comparison_ops import compare_reports

GT_BINS = (("1", 1, 1), ("2-4", 2, 4), ("5-9", 5, 9),
           ("10-19", 10, 19), ("20+", 20, math.inf))
SENSITIVITY_K = (1, 2, 5)


def cohort(rows, denominator):
    deltas = [r["delta_AP"] for r in rows]
    positive = math.fsum(x for x in deltas if x > 0) / denominator
    negative = math.fsum(x for x in deltas if x < 0) / denominator
    return {
        "classes": len(rows), "gt_annotations": sum(r["gt_annotations"] for r in rows),
        "mean_delta_AP": math.fsum(deltas) / len(rows) if rows else None,
        "median_delta_AP": median(deltas) if rows else None,
        "positive_classes": sum(x > 0 for x in deltas),
        "negative_classes": sum(x < 0 for x in deltas),
        "zero_classes": sum(x == 0 for x in deltas),
        "positive_apr_contribution": positive, "negative_apr_contribution": negative,
        "apr_contribution": math.fsum(deltas) / denominator,
    }


def analyze(a_report, p_report):
    # Empty PR focus deliberately skips local PR mechanisms, NOT the macro AP
    # validation or any rare category. Undefined AP stays excluded, never zero.
    comparison = compare_reports(a_report, p_report, focus_names=[])
    if comparison["scope"]["max_dets"] != 300:
        raise ValueError("Expected official all-class image top-300 protocol")
    rows = []
    for source in comparison["macro_attribution"]["per_class"]:
        row = dict(source)
        if row["gt_annotations"] <= 0:
            raise ValueError("Valid rare AP with zero validation GT annotations")
        for metric in ("AP", "AP50", "AP75"):
            row["A_"+metric] = row.pop("old_"+metric)
            row["P_"+metric] = row.pop("new_"+metric)
        rows.append(row)
    rows.sort(key=lambda r: r["category_id"])
    n = len(rows)
    total = cohort(rows, n)
    strata = [{"gt_range": label, **cohort([r for r in rows if lo <= r["gt_annotations"] <= hi], n)}
              for label, lo, hi in GT_BINS]
    gains = sorted((r for r in rows if r["delta_AP"] > 0), key=lambda r: (-r["delta_AP"], r["category_id"]))
    losses = sorted((r for r in rows if r["delta_AP"] < 0), key=lambda r: (r["delta_AP"], r["category_id"]))
    single_gains = [r for r in gains if r["gt_annotations"] == 1]
    single_losses = [r for r in losses if r["gt_annotations"] == 1]
    sensitivities = []
    for k in SENSITIVITY_K:
        for policy, removed in (("largest_gains", gains[:k]), ("largest_losses", losses[:k]),
                                ("balanced_gains_and_losses", gains[:k]+losses[:k]),
                                ("single_GT_largest_gains", single_gains[:k]),
                                ("single_GT_balanced_tails", single_gains[:k]+single_losses[:k])):
            ids = {r["category_id"] for r in removed}
            sensitivities.append({
                "policy": policy, "requested_k_per_tail": k,
                "removed": [{key: r[key] for key in ("category_id", "name", "gt_annotations", "delta_AP", "apr_contribution")}
                            for r in removed],
                "removed_apr_contribution": math.fsum(r["delta_AP"] for r in removed)/n,
                "retained": cohort([r for r in rows if r["category_id"] not in ids], n),
            })
    single = strata[0]
    positive_total = total["positive_apr_contribution"]
    summary = {
        "A_apr": comparison["old_apr"], "P_apr": comparison["new_apr"],
        "delta_apr": comparison["delta_apr"],
        "rare_categories": comparison["scope"]["rare_category_count"],
        "valid_rare_categories": n, "excluded_undefined_categories": comparison["scope"]["rare_category_count"]-n,
        "global": total, "gt_strata": strata,
        "gt_at_least": {str(k): cohort([r for r in rows if r["gt_annotations"] >= k], n) for k in (2, 5, 10)},
        "singleton": {
            **single,
            "share_of_all_positive_contribution": single["positive_apr_contribution"]/positive_total if positive_total else None,
            "non_singleton": cohort([r for r in rows if r["gt_annotations"] != 1], n),
            "top_positive_classes": single_gains[:5],
        },
        "top_gains": gains[:20], "top_losses": losses[:20],
        "sensitivity": sensitivities, "per_class": rows,
        "scope": [
            "Official full-validation bbox AP over IoU .50:.05:.95, all area, maxDets=300; delta=P-A in AP points.",
            "Every valid rare class has equal weight. Undefined AP is excluded, not replaced by zero.",
            "GT strata use validation annotation counts, NOT training frequency, and do not replace official APr.",
            "Every contribution uses the original valid-class denominator; retained mean uses only retained classes.",
            "Top-tail removals (k=1,2,5; both signs shown) are post-hoc concentration diagnostics, not corrected metrics.",
            "Exact sign counts; no rounding before analysis. No significance test or independent-seed replication.",
            "No causal loss/module attribution, per-class tuning, automatic adoption or new training.",
        ],
    }
    if not math.isclose(sum(s["apr_contribution"] for s in strata), summary["delta_apr"], abs_tol=2e-5, rel_tol=0):
        raise ValueError("GT strata do not reconstruct official delta APr")
    return summary


def format_results(report):
    def number(x):
        return "n/a" if x is None else f"{x:+.4f}"

    total = report["global"]
    lines = ["=== All-valid rare AP concentration: P - A ===",
             f"A=APR+projection, P=same APR without projection; valid={report['valid_rare_categories']} / rare={report['rare_categories']}",
             f"A APr={report['A_apr']:.4f} P APr={report['P_apr']:.4f} delta={report['delta_apr']:+.4f}",
             f"positive contribution={total['positive_apr_contribution']:+.4f} negative={total['negative_apr_contribution']:+.4f}",
             "\nGT count  classes  mean dAP  median dAP  +/-/zero  APr contribution (+ / - / net)"]
    for s in report["gt_strata"]:
        lines.append(f"{s['gt_range']:>8} {s['classes']:8d} {number(s['mean_delta_AP']):>9} {number(s['median_delta_AP']):>11} "
                     f"{s['positive_classes']}/{s['negative_classes']}/{s['zero_classes']}  "
                     f"{s['positive_apr_contribution']:+.4f} / {s['negative_apr_contribution']:+.4f} / {s['apr_contribution']:+.4f}")
    s = report["singleton"]
    lines += ["\n=== Single-GT concentration ===",
              f"single-GT net contribution={s['apr_contribution']:+.4f}; other classes net contribution={s['non_singleton']['apr_contribution']:+.4f}",
              f"single-GT share of ALL POSITIVE contribution={s['share_of_all_positive_contribution']}",
              f"without ALL single-GT classes: mean dAP={number(s['non_singleton']['mean_delta_AP'])}, classes={s['non_singleton']['classes']}"]
    for k, s in report["gt_at_least"].items():
        lines.append(f"GT>={k}: classes={s['classes']} mean dAP={number(s['mean_delta_AP'])} median={number(s['median_delta_AP'])} contribution={s['apr_contribution']:+.4f}")
    for title, key in (("Largest gains", "top_gains"), ("Largest declines", "top_losses")):
        lines.append(f"\n=== {title}: class / GT / A AP / P AP / delta / contribution ===")
        for r in report[key]:
            lines.append(f"{r['name']:26} {r['gt_annotations']:4d} {r['A_AP']:8.3f} {r['P_AP']:8.3f} {r['delta_AP']:+9.3f} {r['apr_contribution']:+9.4f}")
    lines.append("\n=== Post-hoc tail-removal sensitivity; NOT official APr ===")
    for s in report["sensitivity"]:
        names = ", ".join(f"{r['name']}(GT={r['gt_annotations']})" for r in s["removed"]) or "none"
        lines.append(f"{s['policy']} k={s['requested_k_per_tail']} removed=[{names}] "
                     f"retained classes={s['retained']['classes']} mean dAP={number(s['retained']['mean_delta_AP'])} "
                     f"retained contribution={s['retained']['apr_contribution']:+.4f}")
    lines += ["\nScope:", *report["scope"]]
    return "\n".join(lines)+"\n"
