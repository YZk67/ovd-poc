"""Frozen-region endpoint arithmetic and local TPA gradient probes, not AdamW replay."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tools.tpa_geometry_audit_ops import WEIGHTS
from tools.tpa_gradient_audit_ops import direction_jvp, gradient_directions


def select_ranking_classes(comparison, count=3):
    if not comparison.get("complete") or not 1 <= count <= 5:
        raise ValueError("Need a complete PR comparison and 1..5 classes")
    candidates = []
    for row in comparison["per_class"]:
        pair = row["curves"].get("0.50", {})
        old, new = pair.get("old"), pair.get("new")
        if (old and new and row["delta_AP"] is not None and row["delta_AP"] < -1e-5
                and row["delta_AP50"] is not None and row["delta_AP50"] < -1e-5
                and old["num_gt"] == new["num_gt"]
                and old["true_positives"] == new["true_positives"] > 0):
            candidates.append(row)
    candidates.sort(key=lambda row: (row["delta_AP"], row["category_id"]))
    names = [row["name"] for row in candidates[:count]]
    if not names:
        raise ValueError("No recall-preserving AP50 regressions in the supplied PR focus")
    return names


def flat_delta(old, new):
    for name in WEIGHTS:
        if old[name].shape != new[name].shape:
            raise ValueError(f"TPA shape changed: {name}")
    delta = torch.cat([(new[name].double() - old[name].double()).flatten() for name in WEIGHTS])
    if not torch.isfinite(delta).all():
        raise ValueError("Nonfinite checkpoint delta")
    return delta


def selected_logits(features, prototypes, indices, bank):
    if (not math.isfinite(bank["temperature"]) or bank["temperature"] <= 0
            or not math.isfinite(bank["logit_scale"]) or bank["logit_scale"] <= 0
            or not math.isfinite(bank["cls_bias"])):
        raise ValueError("Classifier temperature/scale must be positive; bias finite")
    if not torch.isfinite(features).all() or not torch.isfinite(prototypes).all():
        raise ValueError("Nonfinite features/prototypes")
    slots = F.normalize(prototypes.double()[indices], dim=-1)
    cosines = (slots * F.normalize(features.double(), dim=-1)[:, None]).sum(-1)
    tau = bank["temperature"]
    return bank["logit_scale"] * tau * (torch.logsumexp(cosines / tau, -1) - math.log(slots.shape[1])) + bank["cls_bias"]


def grouped_margins(values, records):
    """Mean of every selected TP minus every selected FP; equal region weight.

    The labels are fixed 10ep/source-new diagnostic labels, NOT rematched under
    a counterfactual. Reused counterpart queries are not independent samples.
    """
    if values.ndim != 1 or len(values) != len(records) or not torch.isfinite(values).all():
        raise ValueError("One finite scalar per diagnostic region is required")
    if any(r["kind"] not in ("tp", "fp") for r in records):
        raise ValueError("Only fixed TP/FP labels are accepted")
    result = {}
    for name in sorted({r["category"] for r in records}):
        tp = [i for i, r in enumerate(records) if r["category"] == name and r["kind"] == "tp"]
        fp = [i for i, r in enumerate(records) if r["category"] == name and r["kind"] == "fp"]
        result[name] = {"tp_regions": len(tp), "fp_regions": len(fp), "pairs": len(tp) * len(fp),
                        "mean_tp_minus_fp": float(values[tp].mean() - values[fp].mean()) if tp and fp else None}
    return result


def endpoint_effects(features, indices, banks, records, clip_logp, beta):
    """Two-order-average terminal-bank swap; native boxes need not coincide.

    Query/bias residual includes upstream TPA/query fusion, NOT an isolated
    causal effect of detector-only training. The CLIP term includes ROI change.
    """
    if not 0 <= beta <= 1 or not len(records):
        raise ValueError("Nonempty diagnostic panel and beta in [0,1] required")
    for key in ("category_ids", "temperature", "logit_scale", "tpa_tau", "prompt_sha256"):
        if banks["old"][key] != banks["new"][key]:
            raise ValueError(f"Endpoint protocol differs: {key}")
    native, logdet = {}, {}
    for q_side in ("old", "new"):
        for p_side in ("old", "new"):
            logits = selected_logits(features[q_side], banks[p_side]["prototypes"], indices, banks[q_side])
            logdet[q_side, p_side] = (1 - beta) * F.logsigmoid(logits)
            if q_side == p_side:
                native[q_side] = logits
    a, b, c, d = (logdet[key] for key in (("old", "old"), ("old", "new"), ("new", "old"), ("new", "new")))
    effects = {"terminal_tpa_bank": ((b - a) + (d - c)) / 2,
               "query_and_bias_path": ((c - a) + (d - b)) / 2,
               "clip_roi_path": beta * (clip_logp["new"] - clip_logp["old"])}
    effects["total"] = (d + beta * clip_logp["new"]) - (a + beta * clip_logp["old"])
    error = float((effects["terminal_tpa_bank"] + effects["query_and_bias_path"]
                   + effects["clip_roi_path"] - effects["total"]).abs().max())
    if error > 1e-8:
        raise ValueError("Endpoint log-score decomposition does not close")
    return {"closure_max_abs_error": error,
            "native_margin": {side: grouped_margins(logdet[side, side] + beta * clip_logp[side], records)
                              for side in ("old", "new")},
            "margin_change": {key: grouped_margins(value, records) for key, value in effects.items()},
            "native_logits": {key: value.tolist() for key, value in native.items()},
            "per_region_effects": {key: value.tolist() for key, value in effects.items()}}


def local_margin_audit(captured, state, prompts, bank, features, indices, records, beta, actual_delta):
    windows = []
    for window in captured["windows"]:
        directions, stats = gradient_directions(window["gradients"], captured["clip_max_norm"])
        clip = stats["directions"]["routed_total"]["clip_coefficient_if_used_alone"]
        effects, report = {}, {"window": window["window"], "mean_losses": window["mean_losses"],
                               "microbatches": window["microbatches"], "routing": stats["routing"],
                               "common_clip_coefficient": clip, "directions": {}}
        for name, gradient in directions.items():
            print(f"[JVP] window={window['window']} direction={name}", flush=True)
            jvp = direction_jvp(state, prompts, bank, features, indices, gradient, query_only=True)
            logits = torch.tensor(jvp["query_logits"], dtype=torch.float64)[:, -1]
            cached = torch.tensor([r["native_cache_logit"] for r in records], dtype=torch.float64)
            if float((logits - cached).abs().max()) > 2e-3:
                raise ValueError("Local gradient forward disagrees with native cached logits")
            unit_dlogit = torch.tensor(jvp["query_logit_derivatives"], dtype=torch.float64)[:, -1]
            norm = stats["directions"][name]["norm"]
            effects[name] = (1 - beta) * torch.sigmoid(-logits) * unit_dlogit * norm * clip
            denominator = float(gradient.double().norm() * actual_delta.norm())
            alignment = (float(torch.dot(-gradient.double(), actual_delta) / denominator)
                         if denominator > 1e-12 else None)
            report["directions"][name] = {
                "gradient_norm": norm, "cos_descent_with_8ep_to_10ep_tpa_delta": alignment,
                "margin_derivative": grouped_margins(effects[name], records),
            }
        error = float((effects["detector"] + effects["rpsa"] + effects["apr"]
                       + effects["projection_added"] - effects["routed_total"]).abs().max())
        if error > 1e-6:
            raise ValueError("Common-clipped component margin derivatives do not add")
        report["additive_closure_max_abs_error"] = error
        windows.append(report)
    return windows


def probe_pairing(old_capture, new_capture):
    """Identical seed is not evidence that mapped images or FedLoss subsets agree."""
    checked, paired, missing = 0, 0, 0
    old, new = old_capture["windows"], new_capture["windows"]
    if len(old) != len(new):
        raise ValueError("Endpoint probe window counts differ")
    for a, b in zip(old, new):
        if len(a["microbatches"]) != len(b["microbatches"]):
            raise ValueError("Endpoint microbatch counts differ")
        for x, y in zip(a["microbatches"], b["microbatches"]):
            checked += 1
            keys = ("image_ids", "global_gt_classes_before_remap", "fedloss_category_indices", "mapped_inputs")
            verified = all(key in x and key in y for key in keys)
            for row in (x, y):
                mapped = row.get("mapped_inputs", [])
                verified = (verified and bool(mapped) and len(mapped) == len(row.get("image_ids", []))
                            and all(r.get("gt_boxes_sha256") and r.get("gt_classes_sha256")
                                    and r.get("image_sha256") for r in mapped))
            if not verified:
                missing += 1
            elif all(x[key] == y[key] for key in keys):
                paired += 1
    return {"microbatches": checked, "matched_inputs_and_fedloss": paired, "unverifiable": missing,
            "all_verified_equal": bool(checked) and checked == paired,
            "note": "Does not verify identical internal dropout/denoising draws, DDP sampling, or historical training batches."}
