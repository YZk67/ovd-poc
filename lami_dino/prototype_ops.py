"""Pure tensor operations for InstructDet's multi-prototype interface.

This module intentionally depends only on PyTorch so the equations shared by
the classifier and query-initialisation paths can be tested without importing
Detectron2/Detrex.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def calibrated_logmeanexp_similarity(
    features: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    temperature: float,
    logit_scale: float,
) -> torch.Tensor:
    """Equation (2): smooth-max similarity without a prototype-count bias.

    Args:
        features: Normalized visual features with shape ``[..., D]``.
        prototypes: Normalized text prototypes with shape ``[C, K, D]``.
        temperature: Positive smooth-max temperature.
        logit_scale: Shared category-logit scale.

    Returns:
        Category logits with shape ``[..., C]``.

    Subtracting ``log(K)`` makes the operation a log-*mean*-exp. Therefore a
    category represented by K identical prototypes has exactly the same logit
    as the corresponding single-prototype category.
    """
    if features.ndim < 2:
        raise ValueError(f"features must have shape [..., D], got {features.shape}")
    if prototypes.ndim != 3:
        raise ValueError(f"prototypes must have shape [C, K, D], got {prototypes.shape}")
    if features.shape[-1] != prototypes.shape[-1]:
        raise ValueError(
            "feature/prototype dimensions differ: "
            f"{features.shape[-1]} vs {prototypes.shape[-1]}"
        )
    if prototypes.shape[1] < 1:
        raise ValueError("at least one prototype per category is required")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    similarities = torch.einsum("...d,ckd->...ck", features, prototypes)
    log_mean_exp = torch.logsumexp(similarities / temperature, dim=-1)
    log_mean_exp = log_mean_exp - math.log(prototypes.shape[1])
    return float(logit_scale) * float(temperature) * log_mean_exp


def soft_category_prototype_fusion(
    region_features: torch.Tensor,
    category_logits: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    category_topk: int,
    category_temperature: float,
    prototype_temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Equations (3)-(4): top-R category and within-category prototype fusion.

    Args:
        region_features: Selected encoder features ``[B, Q, D]``.
        category_logits: Their category logits ``[B, Q, C]``.
        prototypes: Projected text prototypes ``[C, K, D]``.

    Returns:
        ``(fused, category_ids, category_weights, prototype_weights)`` where
        ``fused`` has shape ``[B, Q, D]`` and the remaining tensors expose the
        routing decisions for monitoring.
    """
    if region_features.ndim != 3:
        raise ValueError(
            f"region_features must have shape [B, Q, D], got {region_features.shape}"
        )
    if category_logits.ndim != 3:
        raise ValueError(
            f"category_logits must have shape [B, Q, C], got {category_logits.shape}"
        )
    if prototypes.ndim != 3:
        raise ValueError(f"prototypes must have shape [C, K, D], got {prototypes.shape}")
    if region_features.shape[:2] != category_logits.shape[:2]:
        raise ValueError("region_features and category_logits must share [B, Q]")
    if category_logits.shape[-1] != prototypes.shape[0]:
        raise ValueError(
            "category count differs between logits and prototypes: "
            f"{category_logits.shape[-1]} vs {prototypes.shape[0]}"
        )
    if region_features.shape[-1] != prototypes.shape[-1]:
        raise ValueError(
            "feature/prototype dimensions differ: "
            f"{region_features.shape[-1]} vs {prototypes.shape[-1]}"
        )
    if category_topk < 1:
        raise ValueError(f"category_topk must be >= 1, got {category_topk}")
    if category_temperature <= 0 or prototype_temperature <= 0:
        raise ValueError("category and prototype temperatures must be positive")

    topk = min(int(category_topk), category_logits.shape[-1])
    top_scores, category_ids = category_logits.topk(topk, dim=-1)
    category_weights = F.softmax(top_scores / category_temperature, dim=-1)

    # Advanced indexing maps [B,Q,R] category ids to [B,Q,R,K,D].  Use the
    # normalized projected prototypes both for cosine routing and for the query
    # mixture, matching U_{c,k}=norm(W_q P_{c,k}) in the paper.
    selected = F.normalize(prototypes[category_ids], p=2, dim=-1)
    regions = F.normalize(region_features, p=2, dim=-1)
    similarities = torch.einsum("bqd,bqrkd->bqrk", regions, selected)
    prototype_weights = F.softmax(similarities / prototype_temperature, dim=-1)

    category_mixtures = torch.einsum("bqrk,bqrkd->bqrd", prototype_weights, selected)
    fused = torch.einsum("bqr,bqrd->bqd", category_weights, category_mixtures)
    return fused, category_ids, category_weights, prototype_weights
