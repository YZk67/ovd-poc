"""Pure tensor operations for InstructDet's multi-prototype interface.

This module intentionally depends only on PyTorch so the equations shared by
the classifier and query-initialisation paths can be tested without importing
Detectron2/Detrex.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def route_conflicting_task_gradient(
    total_gradient: torch.Tensor,
    apr_gradient: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Remove only the task-gradient component that opposes APR.

    ``total_gradient`` is the gradient of ``L_task + L_APR`` and
    ``apr_gradient`` is the gradient of ``L_APR`` alone.  When the recovered
    task gradient has a negative inner product with the APR gradient, following
    it would increase APR to first order.  Project that single conflicting
    component away and preserve every orthogonal/aligned task component.
    """
    if total_gradient.shape != apr_gradient.shape:
        raise ValueError(
            "total_gradient and apr_gradient must have the same shape, got "
            f"{total_gradient.shape} and {apr_gradient.shape}"
        )
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    task_gradient = total_gradient - apr_gradient
    task_norm = task_gradient.norm(2)
    apr_norm = apr_gradient.norm(2)
    inner_product = torch.dot(task_gradient.reshape(-1), apr_gradient.reshape(-1))
    cosine = inner_product / (task_norm * apr_norm).clamp_min(eps)

    should_project = bool(
        inner_product.detach().item() < 0.0
        and apr_norm.detach().item() > eps
    )
    routed_task_gradient = task_gradient
    if should_project:
        routed_task_gradient = task_gradient - (
            inner_product / apr_gradient.square().sum().clamp_min(eps)
        ) * apr_gradient

    routed_gradient = apr_gradient + routed_task_gradient
    stats = {
        "task_grad_norm": task_norm.detach(),
        "apr_grad_norm": apr_norm.detach(),
        "task_apr_cosine": cosine.detach(),
        "conflict_projected": total_gradient.new_tensor(float(should_project)),
        "routed_grad_norm": routed_gradient.norm(2).detach(),
    }
    return routed_gradient, stats


def prototype_task_view(
    prototypes: torch.Tensor,
    *,
    iteration: int,
    stabilization_steps: int,
    task_gradient_scale: float = 1.0,
    training: bool,
) -> torch.Tensor:
    """Detach task consumers during the APR-only prototype formation phase.

    APR is computed from the original ``prototypes`` tensor before this view is
    created, so TPA continues to receive its diversity/balance gradient. Only
    detector classification and query-fusion gradients are blocked until the
    prototype set has formed distinct directions. Afterwards their gradients
    are multiplied by ``task_gradient_scale`` without changing forward values;
    APR continues to use the unscaled prototype tensor.
    """
    if stabilization_steps < 0:
        raise ValueError("stabilization_steps must be non-negative")
    if not 0.0 <= task_gradient_scale <= 1.0:
        raise ValueError("task_gradient_scale must be within [0, 1]")
    if training and int(iteration) < int(stabilization_steps):
        return prototypes.detach()
    if training and task_gradient_scale < 1.0:
        # Straight-through gradient scaling: numerically this is exactly
        # ``prototypes`` in the forward pass, while its task gradient is scaled.
        detached = prototypes.detach()
        return detached + float(task_gradient_scale) * (prototypes - detached)
    return prototypes


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
