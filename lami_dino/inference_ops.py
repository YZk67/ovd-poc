"""Pure tensor helpers for DINO inference candidate selection."""

from __future__ import annotations

from typing import Tuple

import torch


def select_query_class_topk(
    scores: torch.Tensor,
    *,
    max_detections: int,
    per_query_class_topk: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select image-level detections with an optional per-query class cap.

    Args:
        scores: Classification scores with shape ``[B, Q, C]``.
        max_detections: Maximum number of query/category pairs per image.
        per_query_class_topk: ``0`` preserves the original global ``Q*C``
            top-k. A positive value first retains at most that many categories
            for each query and then applies the image-level top-k.

    Returns:
        ``(selected_scores, query_ids, class_ids)``, each shaped ``[B, M]``.
    """
    if scores.ndim != 3:
        raise ValueError(f"scores must have shape [B,Q,C], got {scores.shape}")
    if max_detections < 1:
        raise ValueError("max_detections must be positive")
    if per_query_class_topk < 0:
        raise ValueError("per_query_class_topk must be non-negative")
    batch_size, num_queries, num_classes = scores.shape
    if num_queries < 1 or num_classes < 1:
        raise ValueError("scores must contain at least one query and class")

    if per_query_class_topk == 0:
        count = min(int(max_detections), num_queries * num_classes)
        selected_scores, flat_ids = scores.reshape(batch_size, -1).topk(
            count, dim=1
        )
        query_ids = torch.div(flat_ids, num_classes, rounding_mode="floor")
        class_ids = flat_ids % num_classes
        return selected_scores, query_ids, class_ids

    class_cap = min(int(per_query_class_topk), num_classes)
    query_scores, query_classes = scores.topk(class_cap, dim=-1)
    candidate_scores = query_scores.reshape(batch_size, -1)
    candidate_classes = query_classes.reshape(batch_size, -1)
    count = min(int(max_detections), candidate_scores.shape[1])
    selected_scores, candidate_ids = candidate_scores.topk(count, dim=1)
    query_ids = torch.div(candidate_ids, class_cap, rounding_mode="floor")
    class_ids = torch.gather(candidate_classes, 1, candidate_ids)
    return selected_scores, query_ids, class_ids
