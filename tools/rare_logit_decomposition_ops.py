"""Fixed-query calibrated Eq. (2) decomposition, without model forward passes."""

from __future__ import annotations

from collections import defaultdict
import math
from statistics import mean, median

import torch
import torch.nn.functional as F

from lami_dino.pairing_diagnostic_ops import replay_classifier


@torch.no_grad()
def decompose_logit(feature, prototypes, *, temperature, logit_scale, cls_bias):
    """Return exact additive LOGIT terms, not additive probabilities.

    Center = normalize(mean(raw prototypes)), matching the classifier-only
    centroid control. Net mode delta is signed. Its two subterms separate the
    mean-of-unit-slots response from the nonnegative log-mean-exp uplift.
    This does NOT rerun decoder query fusion or reproduce full mode_scale=0.
    """
    if feature.ndim != 1 or prototypes.ndim != 2 or prototypes.shape[1] != feature.numel():
        raise ValueError("Expected feature [D] and prototypes [K,D]")
    if not feature.numel() or not prototypes.shape[0]:
        raise ValueError("Feature dimension and prototype count must be positive")
    if not feature.is_floating_point() or not prototypes.is_floating_point():
        raise ValueError("Features and prototypes must be floating point")
    if not torch.isfinite(feature).all() or not torch.isfinite(prototypes).all():
        raise ValueError("Features and prototypes must be finite")
    tau, scale, bias = map(float, (temperature, logit_scale, cls_bias))
    if not all(math.isfinite(v) for v in (tau, scale, bias)) or tau <= 0 or scale <= 0:
        raise ValueError("Temperature/scale must be finite positive; bias must be finite")
    raw = prototypes.detach().cpu().double()
    x = F.normalize(feature.detach().cpu().double(), dim=0)
    slots = F.normalize(raw, dim=-1)
    cosine = slots @ x
    raw_center = raw.mean(0)
    center = scale * torch.dot(x, F.normalize(raw_center, dim=0))
    average = scale * cosine.mean()
    uplift = scale * tau * (
        torch.logsumexp((cosine - cosine.mean()) / tau, dim=0) - math.log(len(raw))
    )
    correction = average - center
    mode_delta = correction + uplift
    native = scale * tau * (torch.logsumexp(cosine / tau, dim=0) - math.log(len(raw))) + bias
    closure = abs(float(center + mode_delta + bias - native))
    if closure > 1e-8 or float(uplift) < -1e-8:
        raise ValueError("Logit decomposition failed additive/Jensen checks")
    replay = replay_classifier(
        feature.detach().cpu().float()[None], raw.float()[None], temperature=tau,
        logit_scale=scale, cls_bias=bias,
    )[0, 0]
    replay_error = abs(float(replay) - float(native))
    if replay_error > 1e-4:
        raise ValueError("Float64 decomposition disagrees with native float32 Eq. (2)")
    posterior = (cosine / tau).softmax(0)
    return {
        "center_response": float(center), "mode_delta": float(mode_delta), "bias": bias,
        "native_logit": float(native), "centroid_logit": float(center) + bias,
        "mean_slot_response": float(average), "center_to_mean_correction": float(correction),
        "dispersion_uplift": float(uplift), "mean_slot_logit": float(average) + bias,
        "closure_abs_error": closure, "float32_replay_abs_error": replay_error,
        "prototype_count": len(raw), "raw_center_norm": float(raw_center.norm()),
        "center_direction_defined": bool(raw_center.norm() > 1e-12),
        "unit_slot_center_norm": float(slots.mean(0).norm()),
        "feature_norm": float(feature.detach().double().norm()),
        "prototype_norms": raw.norm(dim=-1).tolist(),
        "slot_cosines": cosine.tolist(), "slot_posterior": posterior.tolist(),
        "winning_slot": int(cosine.argmax()), "posterior_max": float(posterior.max()),
        "temperature": tau, "logit_scale": scale,
    }


@torch.no_grad()
def decompose_entry(entry, sample, bank, class_index, protocol):
    """Validate a saved native query identity; keep its CLIP branch fixed."""
    q = entry["query_id"]
    if not isinstance(q, int) or isinstance(q, bool) or not 0 <= q < len(sample["features"]):
        raise ValueError("Invalid saved query ID")
    box = sample["query_boxes"][q].detach().cpu().double()
    saved_box = box.new_tensor(entry["box_xyxy"])
    if (box.shape != (4,) or saved_box.shape != (4,) or not torch.isfinite(box).all()
            or not torch.isfinite(saved_box).all() or (box - saved_box).abs().max() > .05):
        raise ValueError("Saved query box does not match cache")
    terms = decompose_logit(
        sample["features"][q], bank["prototypes"][class_index],
        temperature=bank["temperature"], logit_scale=bank["logit_scale"], cls_bias=bank["cls_bias"],
    )
    roi = sample["roi_features"][q:q + 1].cpu().float()
    text = bank["vlm_text"].cpu().float()
    clip_logp = float(F.log_softmax(roi @ text.t() * bank["vlm_temperature"], dim=-1)[0, class_index])
    det_logp = float(F.logsigmoid(torch.tensor(terms["native_logit"], dtype=torch.float64)))
    weight = float(protocol["beta"] if bank["novel_mask"][class_index] else protocol["alpha"])
    offset = math.log(protocol["novel_scale"]) if bank["novel_mask"][class_index] else 0.
    controls = {}
    for name, key in (("native", "native_logit"), ("centroid", "centroid_logit"), ("mean_slot", "mean_slot_logit")):
        logp = float(F.logsigmoid(torch.tensor(terms[key], dtype=torch.float64)))
        fused_log = (1 - weight) * logp + weight * clip_logp + offset
        controls[name] = {"detector_probability": math.exp(logp),
                          "fused_log_score": fused_log, "fused_score": math.exp(fused_log)}
    checks = {
        "saved_detector_logp_abs_error": abs(det_logp - entry["detector_log_probability"]),
        "saved_clip_logp_abs_error": abs(clip_logp - entry["clip_log_probability"]),
        "saved_fused_score_abs_error": abs(controls["native"]["fused_score"] - entry["fused_score"]),
    }
    if (not all(math.isfinite(v) for v in checks.values())
            or checks["saved_detector_logp_abs_error"] > 2e-4
            or checks["saved_clip_logp_abs_error"] > 2e-4
            or checks["saved_fused_score_abs_error"] > 5e-5):
        raise ValueError(f"Saved native scores disagree with cache replay: {checks}")
    return {**entry, "logit_terms": terms, "fixed_query_controls": controls,
            "score_replay_checks": checks}


TERMS = ("center_response", "mode_delta", "bias", "native_logit",
         "center_to_mean_correction", "dispersion_uplift")


def paired_delta(source, other, source_side):
    old, new = (other, source) if source_side == "new" else (source, other)
    result = {key: new["logit_terms"][key] - old["logit_terms"][key] for key in TERMS}
    result["closure_abs_error"] = abs(
        result["center_response"] + result["mode_delta"] + result["bias"] - result["native_logit"]
    )
    if result["closure_abs_error"] > 1e-8:
        raise ValueError("Paired logit delta failed additive check")
    return result


def summarize(rows):
    """Primary summaries use new-source regions only, avoiding doubled TP controls.

    The old counterpart inherits the REGION label, not an official old TP/FP
    status. Overlapping regions and repeated old queries remain non-independent.
    """
    groups, sources = defaultdict(list), defaultdict(dict)
    for row in rows:
        if row["source_side"] != "new" or len(row["source_candidates"]) != 1:
            continue
        source = row["source_candidates"][0]
        identity = (row["image_id"], source["query_id"])
        sources[row["category"], row["kind"]][identity] = source
        if row["new_minus_old_logit_terms"] is not None:
            groups[row["category"], row["kind"]].append(row)
    summaries = []
    for (category, kind), items in sorted(groups.items()):
        sides = {"new": [r["source_candidates"][0] for r in items],
                 "old": [r["other_best_score_box"] for r in items]}
        summaries.append({
            "category": category, "region_kind": "new_" + kind, "pairs": len(items),
            "unique_old_queries": len({(r["image_id"], r["other_best_score_box"]["query_id"]) for r in items}),
            "unique_new_queries": len({(r["image_id"], r["source_candidates"][0]["query_id"]) for r in items}),
            "old_mean": {k: mean(e["logit_terms"][k] for e in sides["old"]) for k in TERMS},
            "new_mean": {k: mean(e["logit_terms"][k] for e in sides["new"]) for k in TERMS},
            "delta_mean": {k: mean(r["new_minus_old_logit_terms"][k] for r in items) for k in TERMS},
            "delta_median": {k: median(r["new_minus_old_logit_terms"][k] for r in items) for k in TERMS},
        })
    ordering = []
    for category in sorted({key[0] for key in sources}):
        fps, tps = list(sources[category, "fp"].values()), list(sources[category, "tp"].values())
        counts = {"fp_above_tp_native": 0, "fp_above_tp_centroid": 0,
                  "native_only_inversions": 0, "centroid_only_inversions": 0,
                  "native_ties": 0, "centroid_ties": 0}
        for fp in fps:
            for tp in tps:
                native = (fp["fixed_query_controls"]["native"]["fused_log_score"]
                          - tp["fixed_query_controls"]["native"]["fused_log_score"])
                centroid = (fp["fixed_query_controls"]["centroid"]["fused_log_score"]
                            - tp["fixed_query_controls"]["centroid"]["fused_log_score"])
                n_above, c_above = native > 1e-6, centroid > 1e-6
                counts["fp_above_tp_native"] += n_above
                counts["fp_above_tp_centroid"] += c_above
                counts["native_only_inversions"] += n_above and centroid < -1e-6
                counts["centroid_only_inversions"] += c_above and native < -1e-6
                counts["native_ties"] += abs(native) <= 1e-6
                counts["centroid_ties"] += abs(centroid) <= 1e-6
        ordering.append({"category": category, "fp_queries": len(fps), "tp_queries": len(tps),
                         "selected_pairs": len(fps) * len(tps), **counts})
    return summaries, ordering
