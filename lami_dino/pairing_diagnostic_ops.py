"""Pure-PyTorch classifier replay and GT-paired classification diagnostics.

Anchors are selected independently for each GT, so coverage here is deliberately
not one-to-one detection matching or LVIS AP. Pair checkpoints by ``gt_id``;
their query indices have no correspondence.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from .diagnostic_ops import fuse_detector_vlm_scores, pairwise_box_iou_xyxy
from .prototype_ops import calibrated_logmeanexp_similarity


def _check_float_tensor(name: str, value: torch.Tensor, ndim: int) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise ValueError(f"{name} must be a rank-{ndim} tensor")
    if not value.is_floating_point():
        raise ValueError(f"{name} must be floating point")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _finite_scalar(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@torch.no_grad()
def replay_classifier(
    features: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    temperature: float = 0.07,
    logit_scale: float = 50.0,
    cls_bias: float = 0.0,
    query_chunk_size: int = 128,
) -> torch.Tensor:
    """Replay calibrated Eq. (2) on raw post-linear ``[Q,D]`` features.

    Raw ``[C,K,D]`` prototypes and features are normalized internally, matching
    the classifier's ``norm_weight=True`` path. The output retains the features'
    dtype/device. For a centroid control, pass ``prototypes.mean(1, keepdim=True)``;
    averaging the raw bank must precede normalization.
    """
    _check_float_tensor("features", features, 2)
    _check_float_tensor("prototypes", prototypes, 3)
    if features.shape[-1] != prototypes.shape[-1] or features.shape[-1] < 1:
        raise ValueError("features and prototypes need the same nonempty D dimension")
    if prototypes.shape[0] < 1 or prototypes.shape[1] < 1:
        raise ValueError("prototypes need at least one class and one slot")
    temperature = _finite_scalar("temperature", temperature)
    logit_scale = _finite_scalar("logit_scale", logit_scale)
    cls_bias = _finite_scalar("cls_bias", cls_bias)
    if temperature <= 0 or logit_scale <= 0:
        raise ValueError("temperature and logit_scale must be positive")
    if isinstance(query_chunk_size, bool) or not isinstance(query_chunk_size, int) or query_chunk_size < 1:
        raise ValueError("query_chunk_size must be a positive integer")
    if features.shape[0] == 0:
        return features.new_empty((0, prototypes.shape[0]))
    bank = F.normalize(prototypes.to(device=features.device, dtype=features.dtype), dim=-1)
    chunks = [
        calibrated_logmeanexp_similarity(
            F.normalize(chunk, dim=-1),
            bank,
            temperature=temperature,
            logit_scale=logit_scale,
        ) + cls_bias
        for chunk in features.split(query_chunk_size, dim=0)
    ]
    logits = torch.cat(chunks, dim=0)
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("classifier replay produced nonfinite logits")
    return logits


def _class_metrics(logits: torch.Tensor, query_id: int, class_index: int) -> dict:
    values = logits[query_id]
    true_logit = float(values[class_index])
    wrong = torch.cat((values[:class_index], values[class_index + 1:]))
    return {
        # Strictly larger values give tied classes their optimistic rank.
        "detector_rank": 1 + int((values > values[class_index]).sum()),
        "true_logit": true_logit,
        "margin": true_logit - float(wrong.max()) if wrong.numel() else None,
    }


@torch.no_grad()
def analyze_image(
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    gt_ids: List[int],
    query_boxes: torch.Tensor,
    variant_logits: Dict[str, torch.Tensor],
    vlm_logits: torch.Tensor,
    novel_mask: torch.Tensor,
    *,
    native_variant: str,
    alpha: float,
    beta: float,
    novel_scale: float,
    max_dets: int,
    iou_thresholds: Sequence[float],
) -> dict:
    """Return JSON-safe rows and global top-k predictions for one image.

    Boxes are absolute ``xyxy``; classes index the full shared vocabulary.
    ``vlm_logits`` must include the native CLIP temperature, before softmax.
    Each flat row identifies one GT, IoU threshold, and variant. Its ``query_id``
    maximizes the *native fused true-class score* among IoU-eligible queries and
    stays fixed across all variants. ``geometry_anchor`` uses highest IoU alone,
    even below the threshold; ``oracle`` reselects the highest fused true-class
    score for each variant and is an explicitly optimistic diagnostic.

    ``in_topk`` tests the fixed anchor pair; ``pair_topk`` tests coverage by any
    eligible true-class pair. Both use exact global top-k indices over ALL
    classes, without rare filtering. ``threshold_ratio`` divides anchor score
    by the variant's last retained score (null for absent/zero thresholds), and
    does not resolve ties. Missing eligible queries remain rows with null anchor
    metrics. A single-class vocabulary has null true-minus-wrong margin.

    This helper detaches inputs and computes on CPU. It neither matches boxes
    one-to-one nor applies LVIS ignore rules; use ``top_predictions`` with the
    evaluation annotations for false-positive attribution.
    """
    _check_float_tensor("gt_boxes", gt_boxes, 2)
    _check_float_tensor("query_boxes", query_boxes, 2)
    _check_float_tensor("vlm_logits", vlm_logits, 2)
    if gt_boxes.shape[1] != 4 or query_boxes.shape[1] != 4:
        raise ValueError("boxes must have shape [N,4]")
    for name, boxes in (("gt_boxes", gt_boxes), ("query_boxes", query_boxes)):
        if bool((boxes[:, 2:] < boxes[:, :2]).any()):
            raise ValueError(f"{name} must satisfy x2 >= x1 and y2 >= y1")
    num_gt, num_queries = gt_boxes.shape[0], query_boxes.shape[0]
    num_classes = vlm_logits.shape[1]
    if num_classes < 1 or vlm_logits.shape[0] != num_queries:
        raise ValueError("vlm_logits must be [Q,C] with C >= 1")
    if not isinstance(gt_classes, torch.Tensor) or gt_classes.shape != (num_gt,):
        raise ValueError("gt_classes must have shape [G]")
    if gt_classes.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("gt_classes must have an integer dtype")
    if bool(((gt_classes < 0) | (gt_classes >= num_classes)).any()):
        raise ValueError("gt_classes must index the full class vocabulary")
    if len(gt_ids) != num_gt or any(isinstance(x, bool) or not isinstance(x, int) for x in gt_ids):
        raise ValueError("gt_ids must contain one integer per GT")
    if len(set(gt_ids)) != num_gt:
        raise ValueError("gt_ids must be unique within the image")
    if not isinstance(novel_mask, torch.Tensor) or novel_mask.shape != (num_classes,) or novel_mask.dtype != torch.bool:
        raise ValueError("novel_mask must be a boolean [C] tensor")
    if not variant_logits or native_variant not in variant_logits:
        raise ValueError("native_variant must identify an available classifier variant")
    for name, logits in variant_logits.items():
        if not isinstance(name, str):
            raise ValueError("variant names must be strings")
        _check_float_tensor(f"variant_logits[{name!r}]", logits, 2)
        if logits.shape != vlm_logits.shape:
            raise ValueError("all variants and vlm_logits must share shape [Q,C]")
    alpha = _finite_scalar("alpha", alpha)
    beta = _finite_scalar("beta", beta)
    novel_scale = _finite_scalar("novel_scale", novel_scale)
    if not 0 <= alpha <= 1 or not 0 <= beta <= 1 or novel_scale <= 0:
        raise ValueError("alpha/beta must be within [0,1] and novel_scale positive")
    if isinstance(max_dets, bool) or not isinstance(max_dets, int) or max_dets < 0:
        raise ValueError("max_dets must be a non-negative integer")
    thresholds = [_finite_scalar("iou_threshold", value) for value in iou_thresholds]
    if not thresholds or any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("iou_thresholds must be a nonempty sequence within [0,1]")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("iou_thresholds must not contain duplicates")

    # At least float32 avoids unsupported/underflow-prone CPU half operations.
    def cpu_float(value: torch.Tensor) -> torch.Tensor:
        dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
        return value.detach().to(device="cpu", dtype=dtype)

    gt_boxes, query_boxes = cpu_float(gt_boxes), cpu_float(query_boxes)
    gt_classes = gt_classes.detach().cpu()
    vlm_logits = cpu_float(vlm_logits)
    variant_logits = {name: cpu_float(value) for name, value in variant_logits.items()}
    novel_mask = novel_mask.detach().cpu()
    ious = pairwise_box_iou_xyxy(gt_boxes, query_boxes)
    clip_probabilities = vlm_logits.softmax(dim=-1)
    fused, selected, top_predictions, topk_thresholds = {}, {}, {}, {}
    for name, logits in variant_logits.items():
        scores = fuse_detector_vlm_scores(
            logits, vlm_logits, novel_mask,
            base_weight=alpha, novel_weight=beta, novel_scale=novel_scale,
        ).exp()
        if not bool(torch.isfinite(scores).all()):
            raise ValueError(f"fusion for {name!r} produced nonfinite scores")
        fused[name] = scores
        count = min(max_dets, scores.numel())
        top_scores, flat_indices = scores.flatten().topk(count)
        selected[name] = torch.zeros(scores.numel(), dtype=torch.bool)
        selected[name][flat_indices] = True
        selected[name] = selected[name].reshape(num_queries, num_classes)
        topk_thresholds[name] = float(top_scores[-1]) if count else None
        top_predictions[name] = [
            {
                "query_id": int(index) // num_classes,
                "class_index": int(index) % num_classes,
                "score": float(score),
                "box_xyxy": query_boxes[int(index) // num_classes].tolist(),
            }
            for index, score in zip(flat_indices, top_scores)
        ]

    rows = []
    for gt_index, gt_id in enumerate(gt_ids):
        class_index = int(gt_classes[gt_index])
        gt_ious = ious[gt_index]
        geometry_id = int(gt_ious.argmax()) if num_queries else None
        best_iou = float(gt_ious[geometry_id]) if geometry_id is not None else None
        for iou_threshold in thresholds:
            eligible_ids = torch.where(gt_ious >= iou_threshold)[0]
            num_eligible = int(eligible_ids.numel())
            anchor_id = (
                int(eligible_ids[fused[native_variant][eligible_ids, class_index].argmax()])
                if num_eligible else None
            )
            for name, logits in variant_logits.items():
                row = {
                    "gt_id": gt_id,
                    "class_index": class_index,
                    "iou_threshold": iou_threshold,
                    "variant": name,
                    "eligible": bool(num_eligible),
                    "num_eligible": num_eligible,
                    "best_iou": best_iou,
                    "query_id": anchor_id,
                    "iou": None,
                    "detector_rank": None,
                    "true_logit": None,
                    "margin": None,
                    "clip_rank": None,
                    "clip_probability": None,
                    "fused_score": None,
                    "threshold_ratio": None,
                    "in_topk": False,
                    "pair_topk": bool(selected[name][eligible_ids, class_index].any()),
                    "geometry_anchor": None,
                    "oracle": None,
                }
                if geometry_id is not None:
                    row["geometry_anchor"] = {
                        "query_id": geometry_id,
                        "iou": best_iou,
                        "eligible": bool(best_iou >= iou_threshold),
                        **_class_metrics(logits, geometry_id, class_index),
                    }
                if anchor_id is not None:
                    score = float(fused[name][anchor_id, class_index])
                    cutoff = topk_thresholds[name]
                    ratio = score / cutoff if cutoff is not None and cutoff > 0 else None
                    row.update({
                        "iou": float(gt_ious[anchor_id]),
                        **_class_metrics(logits, anchor_id, class_index),
                        "clip_rank": 1 + int((vlm_logits[anchor_id] > vlm_logits[anchor_id, class_index]).sum()),
                        "clip_probability": float(clip_probabilities[anchor_id, class_index]),
                        "fused_score": score,
                        "threshold_ratio": ratio if ratio is None or math.isfinite(ratio) else None,
                        "in_topk": bool(selected[name][anchor_id, class_index]),
                    })
                    oracle_id = int(eligible_ids[fused[name][eligible_ids, class_index].argmax()])
                    row["oracle"] = {
                        "query_id": oracle_id,
                        "iou": float(gt_ious[oracle_id]),
                        **_class_metrics(logits, oracle_id, class_index),
                        "fused_score": float(fused[name][oracle_id, class_index]),
                    }
                rows.append(row)
    return {"rows": rows, "top_predictions": top_predictions, "topk_thresholds": topk_thresholds}
