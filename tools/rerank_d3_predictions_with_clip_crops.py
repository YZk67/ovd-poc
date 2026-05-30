#!/usr/bin/env python3
"""
Post-hoc D3 crop reranking with CLIP/SigLIP-style encoders.

This is a lightweight pilot for region-level description verification:

1. Load detector COCO-format predictions.
2. For each image, crop the top scoring boxes.
3. Score each crop against the predicted D3 phrase prompt.
4. Fuse detector score and crop-text score.
5. Save reranked COCO-format predictions and optionally run COCO bbox eval.

The script intentionally runs outside the detector so a saved
coco_instances_results.json can be reused for multiple fusion sweeps.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm


EXPANDED_SCORE_CACHE_VERSION = 1


def _load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, separators=(",", ":"))


def _jsonable_args(args: argparse.Namespace) -> Dict[str, Any]:
    values = {}
    for key, value in vars(args).items():
        values[key] = str(value) if isinstance(value, Path) else value
    return values


def _load_categories(annotation: Mapping[str, Any], phrases_json: Optional[Path]) -> Dict[int, str]:
    if phrases_json is not None:
        phrases = _load_json(phrases_json)
        if not isinstance(phrases, list):
            raise ValueError(f"Expected phrase JSON list, got {type(phrases).__name__}.")
        return {idx + 1: str(phrase) for idx, phrase in enumerate(phrases)}

    categories = annotation.get("categories")
    if not categories:
        raise ValueError("Annotation JSON has no categories; pass --phrases-json explicitly.")
    return {int(cat["id"]): str(cat.get("name", cat.get("raw_sent", cat["id"]))) for cat in categories}


def _group_predictions(predictions: Sequence[Mapping[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        grouped[int(pred["image_id"])].append(dict(pred))
    for preds in grouped.values():
        preds.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return grouped


def _load_image_ids_from_jsonl(path: Path) -> List[int]:
    image_ids = set()
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image_ids.add(int(row["image_id"]))
    return sorted(image_ids)


def _limit_predictions(predictions: Sequence[Dict[str, Any]], keep_topk: int) -> List[Dict[str, Any]]:
    if keep_topk <= 0:
        return list(predictions)
    return list(predictions[:keep_topk])


def _xywh_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax, ay, aw, ah = [float(v) for v in box_a]
    bx, by, bw, bh = [float(v) for v in box_b]
    ax1, ay1 = ax + aw, ay + ah
    bx1, by1 = bx + bw, by + bh
    inter_w = max(0.0, min(ax1, bx1) - max(ax, bx))
    inter_h = max(0.0, min(ay1, by1) - max(ay, by))
    inter = inter_w * inter_h
    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _select_class_agnostic_proposals(
    predictions: Sequence[Mapping[str, Any]],
    *,
    topk: int,
    nms_thresh: float,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for pred in predictions:
        bbox = [float(v) for v in pred["bbox"]]
        if bbox[2] <= 1 or bbox[3] <= 1:
            continue
        if any(_xywh_iou(bbox, item["bbox"]) >= nms_thresh for item in selected):
            continue
        selected.append(
            {
                "bbox": bbox,
                "score": float(pred.get("score", 0.0)),
                "source_category_id": int(pred.get("category_id", -1)),
            }
        )
        if topk > 0 and len(selected) >= topk:
            break
    return selected


def _expanded_category_scores(
    proposals: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    category_to_row: Mapping[int, int],
    num_categories: int,
    match_iou: float,
) -> np.ndarray:
    scores = np.full((len(proposals), num_categories), np.nan, dtype=np.float32)
    proposal_boxes = [proposal["bbox"] for proposal in proposals]
    for pred in predictions:
        cat_id = int(pred.get("category_id", -1))
        if cat_id not in category_to_row:
            continue
        pred_box = pred["bbox"]
        best_idx = -1
        best_iou = 0.0
        for proposal_idx, proposal_box in enumerate(proposal_boxes):
            iou = _xywh_iou(pred_box, proposal_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = proposal_idx
        if best_idx >= 0 and best_iou >= match_iou:
            row = category_to_row[cat_id]
            current = scores[best_idx, row]
            pred_score = float(pred.get("score", 0.0))
            if not np.isfinite(current) or pred_score > current:
                scores[best_idx, row] = pred_score
    return scores


def _expanded_base_scores_from_category_scores(
    proposals: Sequence[Mapping[str, Any]],
    category_scores: np.ndarray,
    *,
    num_categories: int,
    mode: str,
    missing_category_score_scale: float,
) -> np.ndarray:
    proposal_scores = np.asarray([float(proposal["score"]) for proposal in proposals], dtype=np.float32)
    if mode == "objectness":
        return np.repeat(proposal_scores[:, None], num_categories, axis=1)
    if mode != "category_score":
        raise ValueError(f"Unsupported expanded base score mode: {mode}")

    fallback = np.clip(proposal_scores[:, None] * missing_category_score_scale, 1e-6, 1.0 - 1e-6)
    return np.where(np.isfinite(category_scores), category_scores, fallback).astype(np.float32)


def _resolve_image_path(image_root: Path, file_name: str) -> Path:
    file_path = Path(file_name)
    if file_path.is_absolute() and file_path.exists():
        return file_path

    candidates = [
        image_root / file_name,
        image_root / file_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _expanded_xyxy(
    bbox_xywh: Sequence[float],
    *,
    width: int,
    height: int,
    margin: float,
) -> Optional[Tuple[int, int, int, int]]:
    x, y, w, h = [float(v) for v in bbox_xywh]
    if w <= 1 or h <= 1:
        return None

    pad_x = w * margin
    pad_y = h * margin
    x0 = max(0.0, x - pad_x)
    y0 = max(0.0, y - pad_y)
    x1 = min(float(width), x + w + pad_x)
    y1 = min(float(height), y + h + pad_y)
    if x1 <= x0 + 1 or y1 <= y0 + 1:
        return None
    return int(math.floor(x0)), int(math.floor(y0)), int(math.ceil(x1)), int(math.ceil(y1))


def _draw_boxed_image(
    image: Image.Image,
    xyxy: Tuple[int, int, int, int],
    *,
    line_width: int,
) -> Image.Image:
    boxed = image.copy()
    draw = ImageDraw.Draw(boxed)
    x0, y0, x1, y1 = xyxy
    width = max(2, int(line_width))
    for offset in range(width):
        draw.rectangle((x0 - offset, y0 - offset, x1 + offset, y1 + offset), outline=(255, 0, 0), width=1)
    return boxed


def _make_region_image(
    image: Image.Image,
    bbox_xywh: Sequence[float],
    *,
    image_mode: str,
    crop_margin: float,
    box_line_width: int,
) -> Optional[Image.Image]:
    width, height = image.size
    tight_xyxy = _expanded_xyxy(bbox_xywh, width=width, height=height, margin=0.0)
    if tight_xyxy is None:
        return None
    if image_mode == "boxed":
        return _draw_boxed_image(image, tight_xyxy, line_width=box_line_width)
    if image_mode == "crop":
        crop_xyxy = _expanded_xyxy(bbox_xywh, width=width, height=height, margin=crop_margin)
        return image.crop(crop_xyxy) if crop_xyxy is not None else None
    raise ValueError(f"Unsupported image mode: {image_mode}")


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _logit(score: float) -> float:
    score = min(max(score, 1e-6), 1.0 - 1e-6)
    return math.log(score / (1.0 - score))


def _fuse_score(
    detector_score: float,
    clip_score: float,
    *,
    mode: str,
    fusion_weight: float,
    clip_scale: float,
    clip_center: float,
) -> float:
    clip_logit = clip_scale * (clip_score - clip_center)
    clip_prob = _sigmoid(clip_logit)

    if mode == "logit_add":
        return _sigmoid(_logit(detector_score) + fusion_weight * clip_logit)
    if mode == "linear":
        return (1.0 - fusion_weight) * detector_score + fusion_weight * clip_prob
    if mode == "replace":
        return clip_prob
    raise ValueError(f"Unsupported fusion mode: {mode}")


def _fuse_logit_score(
    detector_score: float,
    verifier_logit: float,
    *,
    mode: str,
    fusion_weight: float,
) -> float:
    verifier_prob = _sigmoid(verifier_logit)

    if mode == "logit_add":
        return _sigmoid(_logit(detector_score) + fusion_weight * verifier_logit)
    if mode == "linear":
        return (1.0 - fusion_weight) * detector_score + fusion_weight * verifier_prob
    if mode == "replace":
        return verifier_prob
    raise ValueError(f"Unsupported verifier fusion mode: {mode}")


def _fuse_clip_score_matrix(
    base_scores: np.ndarray,
    clip_scores: np.ndarray,
    *,
    mode: str,
    fusion_weight: float,
    clip_scale: float,
    clip_center: float,
) -> np.ndarray:
    clip_logits = clip_scale * (clip_scores - clip_center)
    clip_probs = 1.0 / (1.0 + np.exp(-clip_logits))
    base_scores = np.asarray(base_scores, dtype=np.float32)
    if mode == "logit_add":
        clipped = np.clip(base_scores, 1e-6, 1.0 - 1e-6)
        base_logits = np.log(clipped / (1.0 - clipped))
        logits = base_logits + fusion_weight * clip_logits
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    if mode == "linear":
        return ((1.0 - fusion_weight) * base_scores + fusion_weight * clip_probs).astype(np.float32)
    if mode == "replace":
        return clip_probs.astype(np.float32)
    raise ValueError(f"Unsupported fusion mode: {mode}")


def _fuse_verifier_logit_matrix(
    base_scores: np.ndarray,
    verifier_logits: np.ndarray,
    *,
    mode: str,
    fusion_weight: float,
) -> np.ndarray:
    base_scores = np.asarray(base_scores, dtype=np.float32)
    verifier_logits = np.asarray(verifier_logits, dtype=np.float32)
    verifier_probs = 1.0 / (1.0 + np.exp(-verifier_logits))
    if mode == "logit_add":
        clipped = np.clip(base_scores, 1e-6, 1.0 - 1e-6)
        base_logits = np.log(clipped / (1.0 - clipped))
        logits = base_logits + fusion_weight * verifier_logits
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    if mode == "linear":
        return ((1.0 - fusion_weight) * base_scores + fusion_weight * verifier_probs).astype(np.float32)
    if mode == "replace":
        return verifier_probs.astype(np.float32)
    raise ValueError(f"Unsupported verifier fusion mode: {mode}")


def _infer_clip_feature_dim(input_dim: int, feature_mode: str) -> Tuple[int, bool]:
    if feature_mode == "full":
        if (input_dim - 1) % 4 != 0:
            raise ValueError(f"Cannot infer CLIP feature dim from full input_dim={input_dim}.")
        return (input_dim - 1) // 4, True
    if feature_mode == "no_detector_score":
        if input_dim % 4 != 0:
            raise ValueError(f"Cannot infer CLIP feature dim from no_detector_score input_dim={input_dim}.")
        return input_dim // 4, False
    raise ValueError(f"gated_bilinear verifier requires text features; got feature_mode={feature_mode}.")


class CropDescriptionVerifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        second_hidden = max(64, hidden_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, second_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class GatedBilinearDescriptionVerifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        *,
        feature_mode: str,
        projection_dim: int,
    ) -> None:
        super().__init__()
        clip_dim, has_detector_score = _infer_clip_feature_dim(input_dim, feature_mode)
        self.clip_dim = clip_dim
        self.has_detector_score = has_detector_score
        self.feature_mode = feature_mode

        projection_dim = int(projection_dim)
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive for gated_bilinear.")

        self.image_norm = nn.LayerNorm(clip_dim)
        self.text_norm = nn.LayerNorm(clip_dim)
        self.pair_norm = nn.LayerNorm(input_dim)
        self.image_proj = nn.Linear(clip_dim, projection_dim)
        self.text_proj = nn.Linear(clip_dim, projection_dim)
        self.pair_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.interaction_gate = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.Sigmoid(),
        )

        score_dim = 2 if has_detector_score else 0
        head_dim = hidden_dim + projection_dim * 4 + 1 + score_dim
        second_hidden = max(64, hidden_dim // 2)
        self.head = nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, second_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        image = features[:, : self.clip_dim]
        text = features[:, self.clip_dim : self.clip_dim * 2]
        image_proj = self.image_proj(self.image_norm(image))
        text_proj = self.text_proj(self.text_norm(text))
        pair_proj = self.pair_proj(self.pair_norm(features))
        gate = self.interaction_gate(pair_proj)
        interaction = image_proj * text_proj * gate
        cosine = (image * text).sum(dim=-1, keepdim=True)

        parts = [
            pair_proj,
            image_proj,
            text_proj,
            interaction,
            torch.abs(image_proj - text_proj),
            cosine,
        ]
        if self.has_detector_score:
            score = features[:, -1:].clamp(1e-6, 1.0 - 1e-6)
            parts.extend([score, torch.logit(score)])
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)


def _build_verifier_model(
    *,
    arch: str,
    input_dim: int,
    hidden_dim: int,
    dropout: float,
    feature_mode: str,
    projection_dim: int,
) -> nn.Module:
    if arch == "mlp":
        return CropDescriptionVerifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    if arch == "gated_bilinear":
        return GatedBilinearDescriptionVerifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            feature_mode=feature_mode,
            projection_dim=projection_dim,
        )
    raise ValueError(f"Unsupported verifier architecture: {arch}")


def _load_torch_checkpoint(path: Path, device: str) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_verifier(path: Path, device: str) -> Tuple[nn.Module, str, Dict[str, Any]]:
    checkpoint = _load_torch_checkpoint(path, device)
    input_dim = int(checkpoint["input_dim"])
    hidden_dim = int(checkpoint.get("hidden_dim", 512))
    dropout = float(checkpoint.get("dropout", 0.0))
    feature_mode = str(checkpoint.get("feature_mode", "full"))
    arch = str(checkpoint.get("verifier_arch", "mlp"))
    projection_dim = int(checkpoint.get("projection_dim", 256))
    verifier = _build_verifier_model(
        arch=arch,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        feature_mode=feature_mode,
        projection_dim=projection_dim,
    )
    verifier.load_state_dict(checkpoint["model_state"])
    verifier.to(device)
    verifier.eval()
    print(
        f"loaded verifier: {path} "
        f"(epoch={checkpoint.get('epoch')}, score={checkpoint.get('score')}, "
        f"arch={arch}, feature_mode={feature_mode})"
    )
    metadata = {
        "encoder_backend": checkpoint.get("encoder_backend"),
        "encoder_model": checkpoint.get("encoder_model"),
        "encoder_pretrained": checkpoint.get("encoder_pretrained"),
    }
    return verifier, feature_mode, metadata


def _validate_verifier_encoder(args: argparse.Namespace, metadata: Mapping[str, Any]) -> None:
    expected_backend = metadata.get("encoder_backend")
    if expected_backend is None:
        return
    expected_model = metadata.get("encoder_model")
    expected_pretrained = metadata.get("encoder_pretrained")
    mismatches = []
    if str(args.encoder_backend) != str(expected_backend):
        mismatches.append(f"backend checkpoint={expected_backend!r} args={args.encoder_backend!r}")
    if expected_model is not None and str(args.model) != str(expected_model):
        mismatches.append(f"model checkpoint={expected_model!r} args={args.model!r}")
    if (
        str(expected_backend) == "open_clip"
        and expected_pretrained is not None
        and str(args.pretrained) != str(expected_pretrained)
    ):
        mismatches.append(f"pretrained checkpoint={expected_pretrained!r} args={args.pretrained!r}")
    if mismatches:
        raise ValueError(
            "Verifier checkpoint encoder does not match rerank encoder. "
            + "; ".join(mismatches)
        )


def _build_verifier_features(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    detector_scores: Sequence[float],
    feature_mode: str,
) -> torch.Tensor:
    score_tensor = torch.tensor(detector_scores, dtype=image_features.dtype).view(-1, 1)
    if feature_mode == "full":
        return torch.cat(
            [
                image_features,
                text_features,
                image_features * text_features,
                torch.abs(image_features - text_features),
                score_tensor,
            ],
            dim=-1,
        )
    if feature_mode == "no_text":
        return torch.cat([image_features, score_tensor], dim=-1)
    if feature_mode == "no_detector_score":
        return torch.cat(
            [
                image_features,
                text_features,
                image_features * text_features,
                torch.abs(image_features - text_features),
            ],
            dim=-1,
        )
    raise ValueError(f"Unsupported verifier feature mode: {feature_mode}")


@torch.no_grad()
def _score_expanded_pair_signal(
    *,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    base_scores: np.ndarray,
    verifier: Optional[nn.Module],
    verifier_feature_mode: str,
    args: argparse.Namespace,
) -> np.ndarray:
    base_scores_np = np.asarray(base_scores, dtype=np.float32)
    if base_scores_np.ndim == 1:
        base_scores_np = np.repeat(base_scores_np[:, None], int(text_features.shape[0]), axis=1)
    if verifier is None:
        return (image_features @ text_features.T).numpy().astype(np.float32)

    num_proposals = int(image_features.shape[0])
    num_categories = int(text_features.shape[0])
    flat_logits = np.empty((num_proposals * num_categories,), dtype=np.float32)
    all_indices = torch.arange(num_proposals * num_categories, dtype=torch.long)
    for start in range(0, int(all_indices.numel()), args.verifier_pair_batch_size):
        flat_idx = all_indices[start : start + args.verifier_pair_batch_size]
        proposal_idx = torch.div(flat_idx, num_categories, rounding_mode="floor")
        category_idx = flat_idx.remainder(num_categories)
        proposal_batch_scores = base_scores_np[proposal_idx.numpy(), category_idx.numpy()]
        features = _build_verifier_features(
            image_features[proposal_idx],
            text_features[category_idx],
            proposal_batch_scores.tolist(),
            verifier_feature_mode,
        ).to(args.device)
        logits = verifier(features).float().cpu().numpy()
        flat_logits[start : start + len(logits)] = logits
    return flat_logits.reshape(num_proposals, num_categories)


def _fuse_expanded_pair_scores(
    *,
    base_scores: np.ndarray,
    pair_signal: np.ndarray,
    verifier: Optional[nn.Module],
    args: argparse.Namespace,
) -> np.ndarray:
    if verifier is None:
        return _fuse_clip_score_matrix(
            base_scores,
            pair_signal,
            mode=args.fusion,
            fusion_weight=args.fusion_weight,
            clip_scale=args.clip_scale,
            clip_center=args.clip_center,
        )
    return _fuse_verifier_logit_matrix(
        base_scores,
        pair_signal,
        mode=args.verifier_fusion,
        fusion_weight=args.verifier_fusion_weight,
    )


def _top_matrix_indices(scores: np.ndarray, topk: int) -> List[Tuple[int, int]]:
    flat = scores.reshape(-1)
    if topk > 0 and flat.size > topk:
        candidate = np.argpartition(-flat, topk - 1)[:topk]
        candidate = candidate[np.argsort(-flat[candidate], kind="mergesort")]
    else:
        candidate = np.argsort(-flat, kind="mergesort")
    num_categories = scores.shape[1]
    return [(int(idx // num_categories), int(idx % num_categories)) for idx in candidate]


def _expanded_score_cache_path(cache_dir: Path, image_id: int) -> Path:
    return cache_dir / f"{int(image_id):012d}.pt"


def _expanded_feature_cache_meta(
    args: argparse.Namespace,
    *,
    category_ids: Sequence[int],
) -> Dict[str, Any]:
    return {
        "version": EXPANDED_SCORE_CACHE_VERSION,
        "category_ids": [int(category_id) for category_id in category_ids],
        "prompt_template": str(args.prompt_template),
        "encoder_backend": str(args.encoder_backend),
        "model": str(args.model),
        "pretrained": str(args.pretrained),
        "image_mode": str(args.image_mode),
        "crop_margin": float(args.crop_margin),
        "box_line_width": int(args.box_line_width),
        "proposal_topk_per_image": int(args.proposal_topk_per_image),
        "proposal_nms_thresh": float(args.proposal_nms_thresh),
        "category_score_match_iou": float(args.category_score_match_iou),
    }


def _expanded_signal_cache_meta(
    args: argparse.Namespace,
    *,
    verifier_feature_mode: str,
) -> Dict[str, Any]:
    return {
        "expanded_base_score": str(args.expanded_base_score),
        "missing_category_score_scale": float(args.missing_category_score_scale),
        "verifier_checkpoint": str(args.verifier_checkpoint) if args.verifier_checkpoint else None,
        "verifier_feature_mode": str(verifier_feature_mode),
        "signal": "verifier_logits" if args.verifier_checkpoint else "clip_scores",
    }


def _meta_matches(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    for key, expected_value in expected.items():
        if key not in actual:
            return False
        actual_value = actual[key]
        if isinstance(expected_value, float):
            try:
                if abs(float(actual_value) - expected_value) > 1e-9:
                    return False
            except (TypeError, ValueError):
                return False
        else:
            if actual_value != expected_value:
                return False
    return True


def _load_expanded_score_cache(
    path: Path,
    *,
    expected_feature_meta: Mapping[str, Any],
    expected_signal_meta: Mapping[str, Any],
) -> Tuple[Optional[Dict[str, Any]], bool]:
    if not path.exists():
        return None, False
    try:
        cache = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        cache = torch.load(path, map_location="cpu")
    if not isinstance(cache, Mapping):
        return None, False
    meta = cache.get("meta", {})
    if not isinstance(meta, Mapping):
        return None, False
    feature_meta = meta.get("feature")
    signal_meta = meta.get("signal")
    if not isinstance(feature_meta, Mapping) or not _meta_matches(expected_feature_meta, feature_meta):
        return None, False
    signal_valid = (
        isinstance(signal_meta, Mapping)
        and _meta_matches(expected_signal_meta, signal_meta)
        and "pair_signal" in cache
    )
    return dict(cache), signal_valid


def _save_expanded_score_cache(
    path: Path,
    *,
    feature_meta: Mapping[str, Any],
    signal_meta: Mapping[str, Any],
    proposals: Sequence[Mapping[str, Any]],
    category_scores: np.ndarray,
    image_features: torch.Tensor,
    pair_signal: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "meta": {"feature": dict(feature_meta), "signal": dict(signal_meta)},
            "bboxes": torch.tensor([proposal["bbox"] for proposal in proposals], dtype=torch.float32),
            "proposal_scores": torch.tensor([float(proposal["score"]) for proposal in proposals], dtype=torch.float32),
            "source_category_ids": torch.tensor(
                [int(proposal["source_category_id"]) for proposal in proposals],
                dtype=torch.long,
            ),
            "category_scores": torch.from_numpy(category_scores.astype(np.float16, copy=False)),
            "image_features": image_features.detach().cpu().to(dtype=torch.float16),
            "pair_signal": torch.from_numpy(pair_signal.astype(np.float16, copy=False)),
        },
        path,
    )


def _expanded_cache_to_arrays(
    cache: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], np.ndarray, Optional[np.ndarray], Optional[torch.Tensor]]:
    bboxes = cache["bboxes"].float().numpy()
    proposal_scores = cache["proposal_scores"].float().numpy()
    source_category_ids = cache["source_category_ids"].long().numpy()
    proposals = [
        {
            "bbox": [float(v) for v in bbox],
            "score": float(score),
            "source_category_id": int(source_category_id),
        }
        for bbox, score, source_category_id in zip(bboxes, proposal_scores, source_category_ids)
    ]
    category_scores = cache["category_scores"].float().numpy()
    pair_signal = cache["pair_signal"].float().numpy() if "pair_signal" in cache else None
    image_features = cache["image_features"].float() if "image_features" in cache else None
    return proposals, category_scores, pair_signal, image_features


def _load_openclip(model_name: str, pretrained: str, device: str):
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError("open_clip is required. Run this script in the lami env.") from exc

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    tokenizer = open_clip.tokenize
    model.eval()
    return model, preprocess, tokenizer


class _SiglipAdapter:
    def __init__(self, model) -> None:
        self.model = model

    def eval(self):
        self.model.eval()
        return self

    def encode_text(self, tokens) -> torch.Tensor:
        return self.model.get_text_features(**tokens)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model.get_image_features(pixel_values=pixel_values)


def _load_siglip(model_name: str, _pretrained: str, device: str):
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:
        raise ImportError("transformers is required for --encoder-backend siglip.") from exc

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    adapter = _SiglipAdapter(model)

    def preprocess(image: Image.Image) -> torch.Tensor:
        return processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

    def tokenize(texts: Sequence[str]):
        return processor(
            text=list(texts),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    return adapter, preprocess, tokenize


def _load_encoder(backend: str, model_name: str, pretrained: str, device: str):
    if backend == "open_clip":
        model, preprocess, tokenizer = _load_openclip(model_name, pretrained, device)
    elif backend == "siglip":
        model, preprocess, tokenizer = _load_siglip(model_name, pretrained, device)
    else:
        raise ValueError(f"Unsupported encoder backend: {backend}")
    setattr(model, "backend_name", backend)
    setattr(model, "model_name", model_name)
    setattr(model, "pretrained_name", pretrained)
    return model, preprocess, tokenizer


@torch.no_grad()
def _encode_texts(
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    features = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        tokens = tokenizer(batch).to(device)
        feats = model.encode_text(tokens)
        feats = F.normalize(feats.float(), p=2, dim=-1)
        features.append(feats.cpu())
    return torch.cat(features, dim=0)


@torch.no_grad()
def _encode_crops(
    model,
    tensors: Sequence[torch.Tensor],
    *,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    features = []
    for start in range(0, len(tensors), batch_size):
        batch = torch.stack(list(tensors[start : start + batch_size]), dim=0).to(device)
        feats = model.encode_image(batch)
        feats = F.normalize(feats.float(), p=2, dim=-1)
        features.append(feats.cpu())
    return torch.cat(features, dim=0)


def _evaluate_coco(
    annotation_path: Path,
    results: Sequence[Mapping[str, Any]],
    *,
    image_ids: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(annotation_path))
    coco_gt.dataset.setdefault("info", {})
    coco_gt.dataset.setdefault("licenses", [])
    coco_dt = coco_gt.loadRes(list(results))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.maxDets = [1, 10, 100]
    if image_ids is not None:
        evaluator.params.imgIds = sorted(set(int(image_id) for image_id in image_ids))
        print(f"evaluating image ids: {len(evaluator.params.imgIds)}")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = [
        "AP",
        "AP50",
        "AP75",
        "APs",
        "APm",
        "APl",
        "AR1",
        "AR10",
        "AR100",
        "ARs",
        "ARm",
        "ARl",
    ]
    return {name: float(value) for name, value in zip(names, evaluator.stats)}


def rerank(args: argparse.Namespace) -> List[Dict[str, Any]]:
    annotation = _load_json(args.annotation)
    predictions = _load_json(args.predictions)
    if not isinstance(predictions, list):
        raise ValueError(f"Expected prediction JSON list, got {type(predictions).__name__}.")

    image_infos = {int(item["id"]): item for item in annotation["images"]}
    categories = _load_categories(annotation, args.phrases_json)
    category_ids = sorted(categories)
    category_to_row = {cat_id: row for row, cat_id in enumerate(category_ids)}
    prompts = [args.prompt_template.format(phrase=categories[cat_id]) for cat_id in category_ids]

    grouped = _group_predictions(predictions)
    image_ids = sorted(grouped)
    if args.image_id_jsonl is not None:
        selected_ids = set(_load_image_ids_from_jsonl(args.image_id_jsonl))
        image_ids = [image_id for image_id in image_ids if image_id in selected_ids]
        print(f"using image ids from {args.image_id_jsonl}: {len(image_ids)}")
    if args.max_images is not None:
        image_ids = image_ids[: args.max_images]

    if args.skip_rerank:
        reranked_results = []
        for image_id in tqdm(image_ids, desc="selecting detector predictions"):
            preds = [dict(pred) for pred in grouped[image_id]]
            reranked_results.extend(_limit_predictions(preds, args.keep_topk_per_image))
        print(f"input predictions: {len(predictions)}")
        print(f"output predictions: {len(reranked_results)}")
        print("processed crops: 0")
        print("missing images: 0")
        print("invalid crops: 0")
        return reranked_results

    device = args.device
    verifier = None
    verifier_feature_mode = "full"
    verifier_encoder_meta: Dict[str, Any] = {}
    if args.verifier_checkpoint:
        verifier, verifier_feature_mode, verifier_encoder_meta = _load_verifier(args.verifier_checkpoint, device)
        _validate_verifier_encoder(args, verifier_encoder_meta)
    model, preprocess, tokenizer = _load_encoder(args.encoder_backend, args.model, args.pretrained, device)
    text_features = _encode_texts(
        model,
        tokenizer,
        prompts,
        batch_size=args.text_batch_size,
        device=device,
    )
    expanded_feature_cache_meta = _expanded_feature_cache_meta(
        args,
        category_ids=category_ids,
    )
    expanded_signal_cache_meta = _expanded_signal_cache_meta(
        args,
        verifier_feature_mode=verifier_feature_mode,
    )

    reranked_results: List[Dict[str, Any]] = []
    missing_images = 0
    invalid_crops = 0
    processed_crops = 0
    reused_expanded_score_caches = 0
    saved_expanded_score_caches = 0

    for image_id in tqdm(image_ids, desc="reranking images"):
        preds = grouped[image_id]
        image_info = image_infos.get(image_id)
        if image_info is None:
            continue

        if args.expand_all_phrases and args.expanded_score_cache_dir is not None:
            cache_path = _expanded_score_cache_path(args.expanded_score_cache_dir, image_id)
            cache = None
            signal_valid = False
            if args.reuse_expanded_score_cache and not args.overwrite_expanded_score_cache:
                cache, signal_valid = _load_expanded_score_cache(
                    cache_path,
                    expected_feature_meta=expanded_feature_cache_meta,
                    expected_signal_meta=expanded_signal_cache_meta,
                )
            if cache is not None:
                valid_proposals, category_scores, pair_signal, image_features = _expanded_cache_to_arrays(cache)
                base_scores = _expanded_base_scores_from_category_scores(
                    valid_proposals,
                    category_scores,
                    num_categories=len(category_ids),
                    mode=args.expanded_base_score,
                    missing_category_score_scale=args.missing_category_score_scale,
                )
                if not signal_valid:
                    if image_features is None:
                        cache = None
                    else:
                        pair_signal = _score_expanded_pair_signal(
                            image_features=image_features,
                            text_features=text_features,
                            base_scores=base_scores,
                            verifier=verifier,
                            verifier_feature_mode=verifier_feature_mode,
                            args=args,
                        )
                        _save_expanded_score_cache(
                            cache_path,
                            feature_meta=expanded_feature_cache_meta,
                            signal_meta=expanded_signal_cache_meta,
                            proposals=valid_proposals,
                            category_scores=category_scores,
                            image_features=image_features,
                            pair_signal=pair_signal,
                        )
                        saved_expanded_score_caches += 1
                if cache is None:
                    pass
                else:
                    assert pair_signal is not None
                    pair_scores = _fuse_expanded_pair_scores(
                        base_scores=base_scores,
                        pair_signal=pair_signal,
                        verifier=verifier,
                        args=args,
                    )
                    for proposal_idx, category_idx in _top_matrix_indices(pair_scores, args.keep_topk_per_image):
                        proposal = valid_proposals[proposal_idx]
                        cat_id = category_ids[category_idx]
                        pred = {
                            "image_id": int(image_id),
                            "category_id": int(cat_id),
                            "bbox": [float(v) for v in proposal["bbox"]],
                            "score": float(pair_scores[proposal_idx, category_idx]),
                        }
                        if args.include_debug_fields:
                            pred["proposal_score"] = float(proposal["score"])
                            pred["base_score"] = float(base_scores[proposal_idx, category_idx])
                            pred["source_category_id"] = int(proposal["source_category_id"])
                        reranked_results.append(pred)
                    reused_expanded_score_caches += 1
                    continue

        image_path = _resolve_image_path(args.image_root, str(image_info["file_name"]))
        if not image_path.exists():
            missing_images += 1
            if not args.drop_unreranked:
                reranked_results.extend(_limit_predictions(preds, args.keep_topk_per_image))
            continue

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        if args.expand_all_phrases:
            proposals = _select_class_agnostic_proposals(
                preds,
                topk=args.proposal_topk_per_image,
                nms_thresh=args.proposal_nms_thresh,
            )
            crop_tensors = []
            valid_proposals: List[Dict[str, Any]] = []
            for proposal in proposals:
                region_image = _make_region_image(
                    image,
                    proposal["bbox"],
                    image_mode=args.image_mode,
                    crop_margin=args.crop_margin,
                    box_line_width=args.box_line_width,
                )
                if region_image is None:
                    invalid_crops += 1
                    continue
                crop_tensors.append(preprocess(region_image))
                valid_proposals.append(proposal)

            if crop_tensors:
                image_features = _encode_crops(
                    model,
                    crop_tensors,
                    batch_size=args.image_batch_size,
                    device=device,
                )
                category_scores = _expanded_category_scores(
                    valid_proposals,
                    preds,
                    category_to_row=category_to_row,
                    num_categories=len(category_ids),
                    match_iou=args.category_score_match_iou,
                )
                base_scores = _expanded_base_scores_from_category_scores(
                    valid_proposals,
                    category_scores,
                    num_categories=len(category_ids),
                    mode=args.expanded_base_score,
                    missing_category_score_scale=args.missing_category_score_scale,
                )
                pair_signal = _score_expanded_pair_signal(
                    image_features=image_features,
                    text_features=text_features,
                    base_scores=base_scores,
                    verifier=verifier,
                    verifier_feature_mode=verifier_feature_mode,
                    args=args,
                )
                if args.expanded_score_cache_dir is not None:
                    _save_expanded_score_cache(
                        _expanded_score_cache_path(args.expanded_score_cache_dir, image_id),
                        feature_meta=expanded_feature_cache_meta,
                        signal_meta=expanded_signal_cache_meta,
                        proposals=valid_proposals,
                        category_scores=category_scores,
                        image_features=image_features,
                        pair_signal=pair_signal,
                    )
                    saved_expanded_score_caches += 1
                pair_scores = _fuse_expanded_pair_scores(
                    base_scores=base_scores,
                    pair_signal=pair_signal,
                    verifier=verifier,
                    args=args,
                )
                processed_crops += len(crop_tensors)

                for proposal_idx, category_idx in _top_matrix_indices(pair_scores, args.keep_topk_per_image):
                    proposal = valid_proposals[proposal_idx]
                    cat_id = category_ids[category_idx]
                    pred = {
                        "image_id": int(image_id),
                        "category_id": int(cat_id),
                        "bbox": [float(v) for v in proposal["bbox"]],
                        "score": float(pair_scores[proposal_idx, category_idx]),
                    }
                    if args.include_debug_fields:
                        pred["proposal_score"] = float(proposal["score"])
                        pred["base_score"] = float(base_scores[proposal_idx, category_idx])
                        pred["source_category_id"] = int(proposal["source_category_id"])
                    reranked_results.append(pred)
            image.close()
            continue

        output_preds = [dict(pred) for pred in preds]
        crop_tensors: List[torch.Tensor] = []
        crop_positions: List[int] = []
        crop_cat_rows: List[int] = []
        crop_detector_scores: List[float] = []

        for position, pred in enumerate(output_preds[: args.rerank_topk_per_image]):
            cat_id = int(pred["category_id"])
            if cat_id not in category_to_row:
                continue
            region_image = _make_region_image(
                image,
                pred["bbox"],
                image_mode=args.image_mode,
                crop_margin=args.crop_margin,
                box_line_width=args.box_line_width,
            )
            if region_image is None:
                invalid_crops += 1
                continue
            crop_tensors.append(preprocess(region_image))
            crop_positions.append(position)
            crop_cat_rows.append(category_to_row[cat_id])
            crop_detector_scores.append(float(pred.get("score", 0.0)))

        if crop_tensors:
            image_features = _encode_crops(
                model,
                crop_tensors,
                batch_size=args.image_batch_size,
                device=device,
            )
            text_for_crops = text_features[torch.tensor(crop_cat_rows, dtype=torch.long)]
            clip_scores = (image_features * text_for_crops).sum(dim=-1).numpy()
            verifier_logits = None
            if verifier is not None:
                verifier_features = _build_verifier_features(
                    image_features,
                    text_for_crops,
                    crop_detector_scores,
                    verifier_feature_mode,
                ).to(device)
                with torch.no_grad():
                    verifier_logits = verifier(verifier_features).float().cpu().numpy()
            processed_crops += len(crop_tensors)

            for score_idx, (position, clip_score) in enumerate(zip(crop_positions, clip_scores)):
                pred = output_preds[position]
                old_score = float(pred["score"])
                if verifier_logits is None:
                    new_score = _fuse_score(
                        old_score,
                        float(clip_score),
                        mode=args.fusion,
                        fusion_weight=args.fusion_weight,
                        clip_scale=args.clip_scale,
                        clip_center=args.clip_center,
                    )
                else:
                    new_score = _fuse_logit_score(
                        old_score,
                        float(verifier_logits[score_idx]),
                        mode=args.verifier_fusion,
                        fusion_weight=args.verifier_fusion_weight,
                    )
                pred["score"] = float(new_score)
                if args.include_debug_fields:
                    pred["det_score"] = old_score
                    pred["clip_score"] = float(clip_score)
                    if verifier_logits is not None:
                        pred["verifier_logit"] = float(verifier_logits[score_idx])
                        pred["verifier_score"] = float(_sigmoid(float(verifier_logits[score_idx])))

        if args.drop_unreranked:
            output_preds = output_preds[: args.rerank_topk_per_image]

        output_preds.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        output_preds = _limit_predictions(output_preds, args.keep_topk_per_image)
        reranked_results.extend(output_preds)
        image.close()

    print(f"input predictions: {len(predictions)}")
    print(f"output predictions: {len(reranked_results)}")
    print(f"processed crops: {processed_crops}")
    print(f"missing images: {missing_images}")
    print(f"invalid crops: {invalid_crops}")
    print(f"reused expanded score caches: {reused_expanded_score_caches}")
    print(f"saved expanded score caches: {saved_expanded_score_caches}")
    return reranked_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Detector COCO-format result JSON, e.g. coco_instances_results.json.",
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path("dataset/d3/annotations/d3_intra_full.json"),
        help="D3 COCO annotation JSON for evaluation and image metadata.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("dataset/d3/images"),
        help="Root directory containing D3 images.",
    )
    parser.add_argument(
        "--phrases-json",
        type=Path,
        default=Path("dataset/metadata/d3_phrases.json"),
        help="Optional ordered phrase list. Defaults to D3 phrase metadata.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output reranked COCO-format result JSON.",
    )
    parser.add_argument(
        "--image-id-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional JSONL file with image_id fields. If set, rerank/evaluate only those images, "
            "e.g. verifier_pairs_w075/val.jsonl for held-out verifier-val evaluation."
        ),
    )
    parser.add_argument(
        "--prompt-template",
        default="the described target is {phrase}",
        help="Text prompt used for crop-text scoring.",
    )
    parser.add_argument(
        "--encoder-backend",
        choices=("open_clip", "siglip"),
        default="open_clip",
        help="Feature encoder backend used for crop/text scoring.",
    )
    parser.add_argument(
        "--model",
        default="convnext_large_d_320",
        help="OpenCLIP model name or HuggingFace model id for --encoder-backend siglip.",
    )
    parser.add_argument(
        "--pretrained",
        default="laion2b_s29b_b131k_ft_soup",
        help="OpenCLIP pretrained tag. Ignored by --encoder-backend siglip.",
    )
    parser.add_argument(
        "--rerank-topk-per-image",
        type=int,
        default=50,
        help="Only crop-rerank this many highest detector-score predictions per image.",
    )
    parser.add_argument(
        "--keep-topk-per-image",
        type=int,
        default=100,
        help="Keep at most this many predictions per image in the output. Use 0 to keep all.",
    )
    parser.add_argument(
        "--drop-unreranked",
        action="store_true",
        help="Drop predictions outside --rerank-topk-per-image instead of keeping original scores.",
    )
    parser.add_argument(
        "--expand-all-phrases",
        action="store_true",
        help=(
            "Use predictions as class-agnostic proposal boxes and score every selected proposal "
            "against every D3 phrase. This can exploit proposal oracle headroom from sources "
            "such as OWLv2, instead of only reranking already-emitted category rows."
        ),
    )
    parser.add_argument(
        "--proposal-topk-per-image",
        type=int,
        default=100,
        help="Number of class-agnostic proposal boxes to keep before all-phrase expansion.",
    )
    parser.add_argument(
        "--proposal-nms-thresh",
        type=float,
        default=0.9,
        help="Class-agnostic IoU threshold used to deduplicate proposal boxes before expansion.",
    )
    parser.add_argument(
        "--expanded-base-score",
        choices=("objectness", "category_score"),
        default="objectness",
        help=(
            "Base score used for expanded box-phrase pairs. objectness uses the proposal's max "
            "OWLv2 score for every phrase. category_score preserves OWLv2's original score for "
            "matching proposal/category pairs and uses a scaled objectness fallback otherwise."
        ),
    )
    parser.add_argument(
        "--category-score-match-iou",
        type=float,
        default=0.9,
        help="IoU threshold for matching original category predictions back to deduplicated proposals.",
    )
    parser.add_argument(
        "--missing-category-score-scale",
        type=float,
        default=0.05,
        help="Fallback base score multiplier for box-phrase pairs without an original category score.",
    )
    parser.add_argument(
        "--skip-rerank",
        action="store_true",
        help="Only select/limit detector predictions. Useful for subset baseline evaluation.",
    )
    parser.add_argument(
        "--crop-margin",
        type=float,
        default=0.25,
        help="Relative bbox margin added before cropping.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("crop", "boxed"),
        default="crop",
        help="Region image passed to the encoder. boxed uses the full image with a red target box.",
    )
    parser.add_argument("--box-line-width", type=int, default=6)
    parser.add_argument(
        "--fusion",
        choices=("logit_add", "linear", "replace"),
        default="logit_add",
        help="How to fuse detector score and CLIP crop-text score.",
    )
    parser.add_argument(
        "--fusion-weight",
        type=float,
        default=0.25,
        help="Weight of the CLIP term in fusion.",
    )
    parser.add_argument(
        "--clip-scale",
        type=float,
        default=10.0,
        help="Scale applied to centered CLIP cosine before fusion.",
    )
    parser.add_argument(
        "--clip-center",
        type=float,
        default=0.25,
        help="Center subtracted from CLIP cosine before scaling.",
    )
    parser.add_argument(
        "--verifier-checkpoint",
        type=Path,
        default=None,
        help="Optional verifier_best.pt from train_d3_crop_verifier.py. If set, use verifier logits for fusion.",
    )
    parser.add_argument(
        "--verifier-fusion",
        choices=("logit_add", "linear", "replace"),
        default="logit_add",
        help="How to fuse detector score and verifier probability/logit.",
    )
    parser.add_argument(
        "--verifier-fusion-weight",
        type=float,
        default=0.5,
        help="Weight of verifier logit/probability in verifier fusion.",
    )
    parser.add_argument(
        "--verifier-pair-batch-size",
        type=int,
        default=8192,
        help="Batch size for verifier scoring in --expand-all-phrases mode.",
    )
    parser.add_argument(
        "--expanded-score-cache-dir",
        type=Path,
        default=None,
        help=(
            "Optional per-image cache directory for --expand-all-phrases pair signals. "
            "Caches selected proposals, matched category scores, and verifier logits or CLIP scores."
        ),
    )
    parser.add_argument(
        "--reuse-expanded-score-cache",
        action="store_true",
        help="Reuse compatible per-image expanded score caches when available.",
    )
    parser.add_argument(
        "--overwrite-expanded-score-cache",
        action="store_true",
        help="Recompute and overwrite expanded score caches even when compatible cache files exist.",
    )
    parser.add_argument(
        "--image-batch-size",
        type=int,
        default=64,
        help="Batch size for crop image encoding.",
    )
    parser.add_argument(
        "--text-batch-size",
        type=int,
        default=256,
        help="Batch size for prompt text encoding.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional smoke-test limit on number of images.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run COCO bbox evaluation after writing output.",
    )
    parser.add_argument(
        "--eval-output-images-only",
        action="store_true",
        help=(
            "Restrict COCO evaluation to image ids present in the output. "
            "This is automatically enabled when --max-images is set."
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional JSON path for args, output counts, and COCOeval metrics.",
    )
    parser.add_argument(
        "--include-debug-fields",
        action="store_true",
        help="Store det_score and clip_score in output predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = rerank(args)
    _save_json(args.output, results)
    print(f"saved reranked predictions to {args.output}")
    summary: Dict[str, Any] = {
        "args": _jsonable_args(args),
        "output_predictions": len(results),
        "output_images": len({int(result["image_id"]) for result in results}),
    }
    if args.eval:
        eval_image_ids = None
        output_image_ids = sorted({int(result["image_id"]) for result in results})
        if args.eval_output_images_only or args.max_images is not None or args.image_id_jsonl is not None:
            eval_image_ids = output_image_ids
        else:
            annotation = _load_json(args.annotation)
            annotation_image_ids = {int(item["id"]) for item in annotation.get("images", [])}
            if len(output_image_ids) < len(annotation_image_ids):
                eval_image_ids = output_image_ids
                print(
                    "evaluation output covers a subset of annotation images; "
                    f"restricting COCOeval to output image ids: {len(output_image_ids)}"
                )
        summary["eval_image_ids"] = None if eval_image_ids is None else len(set(eval_image_ids))
        summary["coco_eval"] = _evaluate_coco(args.annotation, results, image_ids=eval_image_ids)

    if args.summary_output is not None or args.eval:
        summary_output = args.summary_output or args.output.with_suffix(args.output.suffix + ".summary.json")
        _save_json(summary_output, summary)
        print(f"saved summary to {summary_output}")


if __name__ == "__main__":
    main()
