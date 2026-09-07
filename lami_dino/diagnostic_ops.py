"""Tensor-only helpers for OVD inference diagnostics.

These functions deliberately have no Detectron2/LVIS dependency so score
fusion and prototype ablations can be unit-tested on CPU.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple, Union

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
