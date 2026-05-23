#!/usr/bin/env python3
"""
Post-hoc D3 crop reranking with OpenCLIP.

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
from PIL import Image
from tqdm import tqdm


def _load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, separators=(",", ":"))


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


def _load_torch_checkpoint(path: Path, device: str) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_verifier(path: Path, device: str) -> Tuple[CropDescriptionVerifier, str]:
    checkpoint = _load_torch_checkpoint(path, device)
    input_dim = int(checkpoint["input_dim"])
    hidden_dim = int(checkpoint.get("hidden_dim", 512))
    dropout = float(checkpoint.get("dropout", 0.0))
    feature_mode = str(checkpoint.get("feature_mode", "full"))
    verifier = CropDescriptionVerifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    verifier.load_state_dict(checkpoint["model_state"])
    verifier.to(device)
    verifier.eval()
    print(
        f"loaded verifier: {path} "
        f"(epoch={checkpoint.get('epoch')}, score={checkpoint.get('score')}, feature_mode={feature_mode})"
    )
    return verifier, feature_mode


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
def _score_expanded_pairs(
    *,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    proposal_scores: Sequence[float],
    verifier: Optional[nn.Module],
    verifier_feature_mode: str,
    args: argparse.Namespace,
) -> np.ndarray:
    proposal_scores_np = np.asarray(proposal_scores, dtype=np.float32)
    if verifier is None:
        clip_scores = (image_features @ text_features.T).numpy()
        clip_logits = args.clip_scale * (clip_scores - args.clip_center)
        clip_probs = 1.0 / (1.0 + np.exp(-clip_logits))
        proposal_probs = proposal_scores_np[:, None]
        if args.fusion == "logit_add":
            clipped = np.clip(proposal_probs, 1e-6, 1.0 - 1e-6)
            proposal_logits = np.log(clipped / (1.0 - clipped))
            logits = proposal_logits + args.fusion_weight * clip_logits
            return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
        if args.fusion == "linear":
            return ((1.0 - args.fusion_weight) * proposal_probs + args.fusion_weight * clip_probs).astype(np.float32)
        if args.fusion == "replace":
            return clip_probs.astype(np.float32)
        raise ValueError(f"Unsupported fusion mode: {args.fusion}")

    num_proposals = int(image_features.shape[0])
    num_categories = int(text_features.shape[0])
    flat_scores = np.empty((num_proposals * num_categories,), dtype=np.float32)
    all_indices = torch.arange(num_proposals * num_categories, dtype=torch.long)
    for start in range(0, int(all_indices.numel()), args.verifier_pair_batch_size):
        flat_idx = all_indices[start : start + args.verifier_pair_batch_size]
        proposal_idx = torch.div(flat_idx, num_categories, rounding_mode="floor")
        category_idx = flat_idx.remainder(num_categories)
        proposal_batch_scores = proposal_scores_np[proposal_idx.numpy()]
        features = _build_verifier_features(
            image_features[proposal_idx],
            text_features[category_idx],
            proposal_batch_scores.tolist(),
            verifier_feature_mode,
        ).to(args.device)
        logits = verifier(features).float().cpu().numpy()
        for offset, logit in enumerate(logits):
            flat_scores[start + offset] = _fuse_logit_score(
                float(proposal_batch_scores[offset]),
                float(logit),
                mode=args.verifier_fusion,
                fusion_weight=args.verifier_fusion_weight,
            )
    return flat_scores.reshape(num_proposals, num_categories)


def _top_matrix_indices(scores: np.ndarray, topk: int) -> List[Tuple[int, int]]:
    flat = scores.reshape(-1)
    if topk > 0 and flat.size > topk:
        candidate = np.argpartition(-flat, topk - 1)[:topk]
        candidate = candidate[np.argsort(-flat[candidate], kind="mergesort")]
    else:
        candidate = np.argsort(-flat, kind="mergesort")
    num_categories = scores.shape[1]
    return [(int(idx // num_categories), int(idx % num_categories)) for idx in candidate]


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
) -> None:
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
    model, preprocess, tokenizer = _load_openclip(args.model, args.pretrained, device)
    verifier = None
    verifier_feature_mode = "full"
    if args.verifier_checkpoint:
        verifier, verifier_feature_mode = _load_verifier(args.verifier_checkpoint, device)
    text_features = _encode_texts(
        model,
        tokenizer,
        prompts,
        batch_size=args.text_batch_size,
        device=device,
    )

    reranked_results: List[Dict[str, Any]] = []
    missing_images = 0
    invalid_crops = 0
    processed_crops = 0

    for image_id in tqdm(image_ids, desc="reranking images"):
        preds = grouped[image_id]
        image_info = image_infos.get(image_id)
        if image_info is None:
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
                xyxy = _expanded_xyxy(proposal["bbox"], width=width, height=height, margin=args.crop_margin)
                if xyxy is None:
                    invalid_crops += 1
                    continue
                crop_tensors.append(preprocess(image.crop(xyxy)))
                valid_proposals.append(proposal)

            if crop_tensors:
                image_features = _encode_crops(
                    model,
                    crop_tensors,
                    batch_size=args.image_batch_size,
                    device=device,
                )
                proposal_scores = [float(proposal["score"]) for proposal in valid_proposals]
                pair_scores = _score_expanded_pairs(
                    image_features=image_features,
                    text_features=text_features,
                    proposal_scores=proposal_scores,
                    verifier=verifier,
                    verifier_feature_mode=verifier_feature_mode,
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
            xyxy = _expanded_xyxy(pred["bbox"], width=width, height=height, margin=args.crop_margin)
            if xyxy is None:
                invalid_crops += 1
                continue
            crop = image.crop(xyxy)
            crop_tensors.append(preprocess(crop))
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
        "--model",
        default="convnext_large_d_320",
        help="OpenCLIP model name.",
    )
    parser.add_argument(
        "--pretrained",
        default="laion2b_s29b_b131k_ft_soup",
        help="OpenCLIP pretrained tag.",
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
    if args.eval:
        eval_image_ids = None
        if args.eval_output_images_only or args.max_images is not None or args.image_id_jsonl is not None:
            eval_image_ids = sorted({int(result["image_id"]) for result in results})
        _evaluate_coco(args.annotation, results, image_ids=eval_image_ids)


if __name__ == "__main__":
    main()
