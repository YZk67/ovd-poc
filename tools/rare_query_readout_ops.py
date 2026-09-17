"""CPU terminal-classifier counterfactuals, NOT model/decoder interventions."""

from __future__ import annotations

from itertools import permutations, product
import math

import torch
import torch.nn.functional as F

from lami_dino.pairing_diagnostic_ops import replay_classifier


SIDES = ("old", "new")


def check_banks(banks):
    for key in ("category_ids", "temperature", "logit_scale", "tpa_tau", "vlm_temperature", "prompt_sha256"):
        if banks["old"][key] != banks["new"][key]:
            raise ValueError(f"Cannot isolate query/bank/bias: endpoint {key} differs")
    for key in ("vlm_text", "novel_mask"):
        if not torch.equal(banks["old"][key], banks["new"][key]):
            raise ValueError(f"Frozen CLIP/class mapping differs: {key}")
    if banks["old"]["prototypes"].shape != banks["new"]["prototypes"].shape:
        raise ValueError("Endpoint prototype shapes differ")
    for side in SIDES:
        for key in ("temperature", "logit_scale", "cls_bias", "vlm_temperature"):
            value = float(banks[side][key])
            if not math.isfinite(value) or (key != "cls_bias" and value <= 0):
                raise ValueError(f"Invalid {side} {key}")


@torch.no_grad()
def readout_grid(samples, banks, entries, class_index, protocol):
    """Read each FIXED native candidate with both banks and both scalar biases.

    Only C-way terminal logits are recomputed, not the whole image's selection.
    CLIP logp, box and comparison cutoff stay native to each candidate. Query IDs
    from different endpoints have no correspondence. Float64 keeps factor sums
    stable; native corners must also agree with the saved float32 readout.
    """
    check_banks(banks)
    if not banks["old"]["novel_mask"][class_index]:
        raise ValueError("Expected a novel/rare target")
    beta, scale = float(protocol["beta"]), float(protocol["novel_scale"])
    if not 0 <= beta <= 1 or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid fusion settings")
    rows = []
    for side in SIDES:
        ids = [e["query_id"] for e in entries[side]]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Need nonempty, distinct fixed candidates at each endpoint")
        features = samples[side]["features"][ids].detach().cpu().double()
        for p_side in SIDES:
            base_logits = replay_classifier(features, banks[p_side]["prototypes"].cpu().double(),
                temperature=banks[side]["temperature"], logit_scale=banks[side]["logit_scale"], cls_bias=0.)
            for b_side in SIDES:
                bias = float(banks[b_side]["cls_bias"])
                logits = base_logits + bias
                for index, entry in enumerate(entries[side]):
                    values, true = logits[index], logits[index, class_index]
                    wrong = values.clone()
                    wrong[class_index] = -torch.inf
                    logp = float(F.logsigmoid(true))
                    clip_logp = float(entry["clip_log_probability"])
                    cutoff = float(entry["image_topk_threshold"])
                    if not math.isfinite(clip_logp) or clip_logp > 0 or not math.isfinite(cutoff) or cutoff <= 0:
                        raise ValueError("Invalid fixed CLIP/cutoff values")
                    fused_log = (1-beta)*logp + beta*clip_logp + math.log(scale)
                    rows.append({
                        "feature_side": side, "query_id": entry["query_id"],
                        "bank_side": p_side, "bias_side": b_side,
                        "native_corner": side == p_side == b_side,
                        "box_xyxy": entry["box_xyxy"], "iou_to_gt": entry["region_iou"],
                        "pre_bias_logit": float(base_logits[index, class_index]),
                        "bias": bias, "detector_logit": float(true), "detector_log_probability": logp,
                        "detector_probability": math.exp(logp),
                        "detector_class_rank_interval": [1+int((values > true).sum()), int((values >= true).sum())],
                        "detector_true_minus_best_other": float(true-wrong.max()),
                        "detector_top1_category_id": banks[side]["category_ids"][int(values.argmax())],
                        "fixed_clip_log_probability": clip_logp, "fused_log_score": fused_log,
                        "fused_score": math.exp(fused_log), "frozen_native_cutoff": cutoff,
                        "frozen_cutoff_ratio": math.exp(fused_log)/cutoff,
                        "frozen_cutoff_log_margin": fused_log-math.log(cutoff),
                        "counterfactual_image_rank": None, "counterfactual_tp": None,
                    })
    return rows


def shapley_three(values):
    """Exact six-order average for three binary factors; not causal attribution."""
    if set(values) != set(product((0, 1), repeat=3)) or not all(math.isfinite(v) for v in values.values()):
        raise ValueError("Need eight finite factor corners")
    result = [0., 0., 0.]
    for order in permutations(range(3)):
        state = [0, 0, 0]
        for factor in order:
            previous = tuple(state)
            state[factor] = 1
            result[factor] += (values[tuple(state)]-values[previous])/6.
    closure = abs(sum(result)-(values[1, 1, 1]-values[0, 0, 0]))
    if closure > 1e-8:
        raise ValueError("Factor decomposition does not close")
    return dict(zip(("projected_query", "terminal_bank", "scalar_bias"), result)), closure


def primary_attribution(rows, primary_ids, beta):
    """Primary GT anchors were selected BEFORE any swap, independently by side."""
    selected = [r for r in rows if r["query_id"] == primary_ids[r["feature_side"]]]
    corners = {tuple(SIDES.index(r[k]) for k in ("feature_side", "bank_side", "bias_side")): r
               for r in selected}
    if len(selected) != 8 or len(corners) != 8:
        raise ValueError("Expected exactly eight primary-candidate corners")
    logit_values = {key: r["detector_logit"] for key, r in corners.items()}
    logdet_values = {key: (1-beta)*r["detector_log_probability"] for key, r in corners.items()}
    logit, error_l = shapley_three(logit_values)
    logdet, error_p = shapley_three(logdet_values)
    old, new = corners[0, 0, 0], corners[1, 1, 1]
    clip = beta*(new["fixed_clip_log_probability"]-old["fixed_clip_log_probability"])
    competition = -math.log(new["frozen_native_cutoff"]/old["frozen_native_cutoff"])
    observed = new["fused_log_score"]-old["fused_log_score"]
    margin_delta = new["frozen_cutoff_log_margin"]-old["frozen_cutoff_log_margin"]
    error_f = abs(sum(logdet.values())+clip-observed)
    error_m = abs(sum(logdet.values())+clip+competition-margin_delta)
    if max(error_f, error_m) > 1e-8:
        raise ValueError("Frozen-CLIP score decomposition does not close")
    # Pre-bias readout: show each swap order as well as the symmetric average.
    a, b, c, d = [corners[k]["pre_bias_logit"] for k in ((0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0))]
    return {"primary_query_ids": primary_ids, "corners": selected,
            "detector_logit_delta": logit_values[1, 1, 1]-logit_values[0, 0, 0],
            "logit_contributions": logit,
            "pre_bias_swap_effects": {"query_at_old_bank": c-a, "query_at_new_bank": d-b,
                                       "bank_at_old_query": b-a, "bank_at_new_query": d-c,
                                       "query_bank_interaction": d-c-b+a},
            "weighted_log_detector_contributions": logdet, "weighted_clip_log_delta": clip,
            "fused_log_score_delta": observed, "competition_log_margin_term": competition,
            "native_log_score_to_cutoff_delta": margin_delta,
            "closure_errors": {"logit": error_l, "weighted_log_detector": error_p,
                               "fused_log_score": error_f, "log_margin": error_m},
            "method": "Six-order-average binary-factor allocation. Descriptive frozen-readout "
                      "arithmetic, NOT a historical training-gradient or decoder-only effect."}
