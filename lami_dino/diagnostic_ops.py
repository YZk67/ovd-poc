"""Tensor-only helpers for OVD inference diagnostics.

These functions deliberately have no Detectron2/LVIS dependency so score
fusion and prototype ablations can be unit-tested on CPU.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple, Union

import torch
import torch.nn.functional as F

from .prototype_ops import calibrated_logmeanexp_similarity


def fuse_detector_vlm_scores(
    detector_logits: torch.Tensor,
    vlm_logits: torch.Tensor,
    novel_mask: torch.Tensor,
    *,
    fusion: str = "power",
    base_weight: float = 0.0,
    novel_weight: float = 0.3,
    novel_scale: float = 3.0,
    detector_temperature: float = 1.0,
    vlm_temperature: float = 1.0,
) -> torch.Tensor:
    """Fuse detector sigmoid scores and full-vocabulary VLM logits.

    ``power`` exactly reproduces DINO's current score ensemble when both
    temperatures are one, ``base_weight=alpha`` and
    ``novel_weight=beta``. ``logprob_add`` keeps the detector coefficient at
    one and adds a separately weighted VLM log-probability; it is a calibrated
    alternative that can be replayed from the same cached logits.
    """
    if detector_logits.shape != vlm_logits.shape:
        raise ValueError(
            "detector_logits and vlm_logits must have identical shapes, got "
            f"{detector_logits.shape} and {vlm_logits.shape}"
        )
    if detector_logits.ndim < 2:
        raise ValueError("score tensors must have at least query and class dimensions")
    if novel_mask.ndim != 1 or novel_mask.numel() != detector_logits.shape[-1]:
        raise ValueError(
            "novel_mask must be [C] and match the class dimension, got "
            f"{novel_mask.shape} for C={detector_logits.shape[-1]}"
        )
    if detector_temperature <= 0.0 or vlm_temperature <= 0.0:
        raise ValueError("detector and VLM temperatures must be positive")
    if novel_scale <= 0.0:
        raise ValueError("novel_scale must be positive")
    for name, value in (("base_weight", base_weight), ("novel_weight", novel_weight)):
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
        if fusion == "power" and value > 1.0:
            raise ValueError(f"{name} must be within [0, 1] for power fusion")

    detector_logprob = F.logsigmoid(detector_logits / detector_temperature)
    vlm_logprob = F.log_softmax(vlm_logits / vlm_temperature, dim=-1)
    weights = torch.full(
        (detector_logits.shape[-1],),
        float(base_weight),
        dtype=detector_logits.dtype,
        device=detector_logits.device,
    )
    weights[novel_mask.to(device=weights.device)] = float(novel_weight)

    if fusion == "power":
        log_scores = (1.0 - weights) * detector_logprob + weights * vlm_logprob
    elif fusion == "logprob_add":
        log_scores = detector_logprob + weights * vlm_logprob
    else:
        raise ValueError(f"unknown fusion {fusion!r}; expected 'power' or 'logprob_add'")

    novel_bias = torch.zeros_like(weights)
    novel_bias[novel_mask.to(device=weights.device)] = math.log(float(novel_scale))
    # Returning log-scores avoids underflow. Ranking and AP are invariant to a
    # shared monotonic exp, while the current power formula can be recovered by
    # exponentiating when an exact probability comparison is needed.
    return log_scores + novel_bias


def sparse_fusion_candidate_pairs(
    detector_logits: torch.Tensor,
    vlm_logits: torch.Tensor,
    novel_mask: torch.Tensor,
    *,
    topk: int,
    profiles: Iterable[Dict[str, Union[float, str]]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the union of top query/class pairs required by profiles.

    Caching only this union makes full-LVIS offline replay practical. Every
    supplied profile is exact for its top-``topk`` predictions because those
    pairs are explicitly included in the cache.
    """
    if detector_logits.ndim != 2 or vlm_logits.ndim != 2:
        raise ValueError("candidate construction expects [Q, C] logits")
    if detector_logits.shape != vlm_logits.shape:
        raise ValueError("detector and VLM logits must have identical shapes")
    if topk < 1:
        raise ValueError("topk must be positive")

    flat_indices = []
    for profile in profiles:
        scores = fuse_detector_vlm_scores(
            detector_logits,
            vlm_logits,
            novel_mask,
            fusion=str(profile.get("fusion", "power")),
            base_weight=float(profile.get("base_weight", 0.0)),
            novel_weight=float(profile.get("novel_weight", 0.3)),
            novel_scale=float(profile.get("novel_scale", 3.0)),
            detector_temperature=float(profile.get("detector_temperature", 1.0)),
            vlm_temperature=float(profile.get("vlm_temperature", 1.0)),
        )
        count = min(int(topk), scores.numel())
        flat_indices.append(scores.reshape(-1).topk(count).indices)

    if not flat_indices:
        raise ValueError("at least one fusion profile is required")
    flat = torch.unique(torch.cat(flat_indices), sorted=True)
    num_classes = detector_logits.shape[-1]
    return torch.div(flat, num_classes, rounding_mode="floor"), flat % num_classes


def fuse_sparse_detector_vlm_scores(
    detector_logits: torch.Tensor,
    vlm_logits: torch.Tensor,
    vlm_log_normalizer: torch.Tensor,
    candidate_query_ids: torch.Tensor,
    candidate_class_ids: torch.Tensor,
    novel_mask: torch.Tensor,
    *,
    fusion: str = "power",
    base_weight: float = 0.0,
    novel_weight: float = 0.3,
    novel_scale: float = 3.0,
    detector_temperature: float = 1.0,
    vlm_temperature: float = 1.0,
) -> torch.Tensor:
    """Replay fusion for a sparse union of query/category candidates.

    The cache stores the dense softmax log-normalizer for each query. Exact
    sparse replay therefore remains possible without storing Q x C logits.
    Temperature changes would require a different normalizer and are rejected.
    """
    if detector_temperature != 1.0 or vlm_temperature != 1.0:
        raise ValueError(
            "sparse replay currently supports detector_temperature=vlm_temperature=1; "
            "temperature sweeps require a dump with matching normalizers"
        )
    tensors = (detector_logits, vlm_logits, candidate_query_ids, candidate_class_ids)
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("sparse candidate tensors must all be one-dimensional")
    if len({tensor.numel() for tensor in tensors}) != 1:
        raise ValueError("sparse candidate tensors must have identical lengths")
    if novel_mask.ndim != 1:
        raise ValueError("novel_mask must be one-dimensional")
    query_ids = candidate_query_ids.long()
    class_ids = candidate_class_ids.long()
    if query_ids.numel() and int(query_ids.max()) >= vlm_log_normalizer.numel():
        raise ValueError("candidate query id exceeds VLM normalizer length")
    if class_ids.numel() and int(class_ids.max()) >= novel_mask.numel():
        raise ValueError("candidate class id exceeds novel mask length")

    detector_logprob = F.logsigmoid(detector_logits.float())
    vlm_logprob = vlm_logits.float() - vlm_log_normalizer.float()[query_ids]
    is_novel = novel_mask[class_ids].to(device=detector_logprob.device)
    weights = torch.where(
        is_novel,
        detector_logprob.new_tensor(float(novel_weight)),
        detector_logprob.new_tensor(float(base_weight)),
    )
    if fusion == "power":
        if not 0.0 <= base_weight <= 1.0 or not 0.0 <= novel_weight <= 1.0:
            raise ValueError("power fusion weights must be within [0, 1]")
        log_scores = (1.0 - weights) * detector_logprob + weights * vlm_logprob
    elif fusion == "logprob_add":
        if base_weight < 0.0 or novel_weight < 0.0:
            raise ValueError("logprob_add weights must be non-negative")
        log_scores = detector_logprob + weights * vlm_logprob
    else:
        raise ValueError(f"unknown fusion {fusion!r}")
    if novel_scale <= 0.0:
        raise ValueError("novel_scale must be positive")
    return log_scores + is_novel.to(log_scores.dtype) * math.log(float(novel_scale))


def prototype_variant_logits(
    features: torch.Tensor,
    prototypes: torch.Tensor,
    prompt_features: torch.Tensor,
    *,
    temperature: float,
    logit_scale: float,
) -> Dict[str, torch.Tensor]:
    """Compute matched-query logits for three prototype aggregations.

    ``features`` may contain only the handful of queries matched to GT boxes;
    this avoids a costly Q x C x K diagnostic over every decoder query.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be [N, D], got {features.shape}")
    if prototypes.ndim != 3:
        raise ValueError(f"prototypes must be [C, K, D], got {prototypes.shape}")
    if prompt_features.ndim != 3:
        raise ValueError(
            f"prompt_features must be [C, P, D], got {prompt_features.shape}"
        )
    if prototypes.shape[0] != prompt_features.shape[0]:
        raise ValueError("prototype and prompt class counts differ")
    if features.shape[-1] != prototypes.shape[-1] or features.shape[-1] != prompt_features.shape[-1]:
        raise ValueError("feature dimensions differ")

    features = F.normalize(features.float(), p=2, dim=-1)
    prototypes = F.normalize(prototypes.float(), p=2, dim=-1)
    prototype_mean = F.normalize(prototypes.mean(dim=1), p=2, dim=-1)
    prompt_mean = F.normalize(prompt_features.float().mean(dim=1), p=2, dim=-1)
    return {
        "logmeanexp": calibrated_logmeanexp_similarity(
            features,
            prototypes,
            temperature=temperature,
            logit_scale=logit_scale,
        ),
        "prototype_mean": float(logit_scale) * features @ prototype_mean.t(),
        "prompt_mean": float(logit_scale) * features @ prompt_mean.t(),
    }


def true_class_mode_weights(
    features: torch.Tensor,
    prototypes: torch.Tensor,
    class_ids: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Return per-instance posterior weights over its true class's K modes."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if features.ndim != 2 or prototypes.ndim != 3 or class_ids.ndim != 1:
        raise ValueError("expected features [N,D], prototypes [C,K,D], class_ids [N]")
    if features.shape[0] != class_ids.shape[0]:
        raise ValueError("features and class_ids must contain the same number of samples")
    features = F.normalize(features.float(), p=2, dim=-1)
    selected = F.normalize(prototypes[class_ids].float(), p=2, dim=-1)
    similarities = torch.einsum("nd,nkd->nk", features, selected)
    return F.softmax(similarities / float(temperature), dim=-1)


def pairwise_box_iou_xyxy(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """Pairwise IoU for absolute-coordinate ``xyxy`` boxes."""
    if boxes1.ndim != 2 or boxes2.ndim != 2:
        raise ValueError("boxes must be rank-2 tensors")
    if boxes1.shape[-1] != 4 or boxes2.shape[-1] != 4:
        raise ValueError("boxes must have shape [N,4] and [M,4]")
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0).prod(dim=-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0).prod(dim=-1)
    left_top = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (right_bottom - left_top).clamp_min(0).prod(dim=-1)
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp_min(1e-12)


def detection_stage_hits(
    query_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    selected_query_ids: torch.Tensor,
    selected_class_ids: torch.Tensor,
    *,
    thresholds: Iterable[float],
) -> Tuple[torch.Tensor, Dict[float, Dict[str, torch.Tensor]]]:
    """Decompose per-GT coverage into proposal and top-k classification stages.

    ``query_boxes`` contains every class-agnostic decoder proposal, while
    ``selected_*`` contains the exact image-level top-k query/category pairs.
    The returned booleans are per GT and intentionally measure coverage upper
    bounds rather than one-to-one LVIS matching.
    """
    if query_boxes.ndim != 2 or query_boxes.shape[-1] != 4:
        raise ValueError("query_boxes must have shape [Q,4]")
    if gt_boxes.ndim != 2 or gt_boxes.shape[-1] != 4:
        raise ValueError("gt_boxes must have shape [G,4]")
    if gt_classes.ndim != 1 or gt_classes.numel() != gt_boxes.shape[0]:
        raise ValueError("gt_classes must have shape [G]")
    if selected_query_ids.ndim != 1 or selected_class_ids.ndim != 1:
        raise ValueError("selected query and class ids must be one-dimensional")
    if selected_query_ids.numel() != selected_class_ids.numel():
        raise ValueError("selected query and class ids must have equal lengths")
    threshold_values: List[float] = [float(value) for value in thresholds]
    if not threshold_values or any(value <= 0.0 or value > 1.0 for value in threshold_values):
        raise ValueError("IoU thresholds must be non-empty and within (0,1]")
    if gt_boxes.shape[0] == 0:
        empty = torch.empty(0, dtype=torch.float32, device=query_boxes.device)
        return empty, {
            value: {
                key: torch.empty(0, dtype=torch.bool, device=query_boxes.device)
                for key in ("proposal", "best_query_pair", "class_aware_topk")
            }
            for value in threshold_values
        }
    if query_boxes.shape[0] == 0:
        raise ValueError("at least one query box is required when GT boxes exist")

    selected_query_ids = selected_query_ids.long()
    selected_class_ids = selected_class_ids.long()
    if selected_query_ids.numel():
        if int(selected_query_ids.min()) < 0 or int(selected_query_ids.max()) >= query_boxes.shape[0]:
            raise ValueError("selected query id is outside the query-box range")

    ious = pairwise_box_iou_xyxy(gt_boxes, query_boxes)
    best_iou, best_query = ious.max(dim=1)
    if selected_query_ids.numel():
        same_class = gt_classes[:, None] == selected_class_ids[None, :]
        selected_ious = ious[:, selected_query_ids]
        best_pair_selected = (
            (best_query[:, None] == selected_query_ids[None, :]) & same_class
        ).any(dim=1)
    else:
        same_class = torch.empty(
            (gt_boxes.shape[0], 0), dtype=torch.bool, device=gt_boxes.device
        )
        selected_ious = torch.empty(
            (gt_boxes.shape[0], 0), dtype=ious.dtype, device=ious.device
        )
        best_pair_selected = torch.zeros(
            gt_boxes.shape[0], dtype=torch.bool, device=gt_boxes.device
        )

    stages = {}
    for threshold in threshold_values:
        proposal = best_iou >= threshold
        class_aware = (
            ((selected_ious >= threshold) & same_class).any(dim=1)
            if selected_query_ids.numel()
            else torch.zeros_like(proposal)
        )
        stages[threshold] = {
            "proposal": proposal,
            "best_query_pair": proposal & best_pair_selected,
            "class_aware_topk": class_aware,
        }
    return best_iou, stages
