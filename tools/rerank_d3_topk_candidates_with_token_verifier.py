#!/usr/bin/env python3
"""
Rerank D3 top-k candidates with a token-level crop/phrase verifier.

This is the inference side for tools/train_d3_token_cross_verifier.py. It
rebuilds topK boxes x topM phrase candidates from detector dumps, crops each
selected box, scores candidate pairs with the trained cross-attention verifier,
fuses detector/verifier scores, and writes COCO-format predictions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from train_d3_token_cross_verifier import (  # noqa: E402
    ClipTokenEncoder,
    TokenCrossVerifier,
    _expanded_xyxy,
    _jsonable_args,
    _resolve_image_path,
    _torch_load,
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, separators=(",", ":"))


def _load_categories(annotation: Mapping[str, Any], phrases_json: Optional[Path]) -> List[Tuple[int, str]]:
    if phrases_json is not None and phrases_json.exists():
        phrases = _load_json(phrases_json)
        if not isinstance(phrases, list):
            raise ValueError(f"Expected phrase JSON list, got {type(phrases).__name__}.")
        return [(idx + 1, str(phrase)) for idx, phrase in enumerate(phrases)]

    categories = annotation.get("categories")
    if not categories:
        raise ValueError("Annotation JSON has no categories; pass --phrases-json explicitly.")
    return [
        (int(cat["id"]), str(cat.get("name", cat.get("raw_sent", cat["id"]))))
        for cat in sorted(categories, key=lambda item: int(item["id"]))
    ]


def _saved_prediction_path(saved_output_dir: Path, file_name: str) -> Path:
    path = Path(file_name)
    return saved_output_dir / path.with_suffix(".pth").name


def _load_saved_prediction(path: Path) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    data = _torch_load(path)
    missing = [key for key in ("pred_boxes", "pred_logits") if key not in data]
    if missing:
        raise KeyError(f"{path} is missing {missing}.")

    boxes = data["pred_boxes"].float()
    logits = data["pred_logits"].float()
    roi_features = data.get("roi_features_ori")
    if roi_features is not None:
        roi_features = roi_features.float()

    if boxes.ndim == 3:
        boxes = boxes[0]
    if logits.ndim == 3:
        logits = logits[0]
    if roi_features is not None and roi_features.ndim == 3:
        roi_features = roi_features[0]

    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"{path} pred_boxes has unsupported shape {tuple(boxes.shape)}.")
    if logits.ndim != 2:
        raise ValueError(f"{path} pred_logits has unsupported shape {tuple(logits.shape)}.")
    return boxes.cpu(), logits.cpu(), roi_features.cpu() if roi_features is not None else None


def _load_vlm_query_embedding(path: Path) -> torch.Tensor:
    array = np.load(path)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D VLM query embedding, got shape {array.shape}.")
    return F.normalize(torch.from_numpy(array).float(), p=2, dim=-1)


def _candidate_scores(
    logits: torch.Tensor,
    roi_features: Optional[torch.Tensor],
    *,
    args: argparse.Namespace,
    vlm_query_embedding: Optional[torch.Tensor],
) -> torch.Tensor:
    scores = logits.sigmoid()
    if args.score_mode == "sigmoid":
        return scores

    if roi_features is None:
        raise KeyError("--score-mode score_ensemble requires roi_features_ori in saved dumps.")
    if vlm_query_embedding is None:
        raise ValueError("--score-mode score_ensemble requires --vlm-query-embedding.")
    if roi_features.shape[-1] != vlm_query_embedding.shape[-1]:
        raise ValueError(
            "ROI feature dim and VLM query dim mismatch: "
            f"{roi_features.shape[-1]} vs {vlm_query_embedding.shape[-1]}."
        )
    vlm_scores = roi_features.float() @ vlm_query_embedding.t()
    vlm_scores = (vlm_scores * float(args.vlm_temperature)).softmax(dim=-1)
    beta = float(args.beta)
    return scores.pow(1.0 - beta) * vlm_scores.pow(beta)


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(dim=-1)
    return torch.stack(
        [
            cx - 0.5 * w,
            cy - 0.5 * h,
            cx + 0.5 * w,
            cy + 0.5 * h,
        ],
        dim=-1,
    )


def _prediction_boxes_to_original_xyxy(
    pred_boxes_cxcywh: torch.Tensor,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    boxes = _cxcywh_to_xyxy(pred_boxes_cxcywh).clamp(min=0.0, max=1.0)
    scale = torch.tensor([width, height, width, height], dtype=boxes.dtype)
    return (boxes * scale).numpy().astype(np.float32)


def _xyxy_to_xywh(box: Sequence[float]) -> List[float]:
    return [
        float(box[0]),
        float(box[1]),
        float(max(0.0, float(box[2]) - float(box[0]))),
        float(max(0.0, float(box[3]) - float(box[1]))),
    ]


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
    raise ValueError(f"Unsupported fusion mode: {mode}")


def _evaluate_coco(
    annotation_path: Path,
    results: Sequence[Mapping[str, Any]],
    *,
    image_ids: Optional[Sequence[int]] = None,
) -> None:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not results:
        print("no results to evaluate")
        return

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


def _load_verifier(path: Path, device: str) -> Tuple[TokenCrossVerifier, Mapping[str, Any]]:
    checkpoint = _torch_load(path, map_location=device)
    verifier = TokenCrossVerifier(
        image_dim=int(checkpoint["image_dim"]),
        text_dim=int(checkpoint["text_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_heads=int(checkpoint["num_heads"]),
        dropout=float(checkpoint.get("dropout", 0.0)),
        use_detector_score=bool(checkpoint.get("use_detector_score", True)),
    )
    verifier.load_state_dict(checkpoint["model_state"])
    verifier.to(device)
    verifier.eval()
    print(
        f"loaded token verifier: {path} "
        f"(epoch={checkpoint.get('epoch')}, score={checkpoint.get('score')}, "
        f"backend={checkpoint.get('backend_name')})"
    )
    return verifier, checkpoint


def _build_candidates_for_image(
    *,
    image_id: int,
    width: int,
    height: int,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    category_ids: Sequence[int],
    category_names: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    if scores.shape[-1] != len(category_ids):
        raise ValueError(f"scores have {scores.shape[-1]} classes, expected {len(category_ids)}.")

    num_boxes = min(max(1, int(args.box_topk)), scores.shape[0])
    num_phrases = min(max(1, int(args.phrase_topk)), scores.shape[1])
    box_scores = scores.max(dim=-1).values
    top_box_scores, top_box_indexes = torch.topk(box_scores, num_boxes)
    pred_xyxy = _prediction_boxes_to_original_xyxy(boxes, width=width, height=height)

    candidates: List[Dict[str, Any]] = []
    for query_rank, (query_score, query_index) in enumerate(zip(top_box_scores.tolist(), top_box_indexes.tolist())):
        query_index = int(query_index)
        phrase_scores, phrase_indexes = torch.topk(scores[query_index], num_phrases)
        bbox = _xyxy_to_xywh(pred_xyxy[query_index])
        for phrase_rank, (phrase_score, phrase_index) in enumerate(zip(phrase_scores.tolist(), phrase_indexes.tolist())):
            phrase_index = int(phrase_index)
            candidates.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(category_ids[phrase_index]),
                    "phrase": str(category_names[phrase_index]),
                    "bbox": bbox,
                    "score": float(phrase_score),
                    "query_index": query_index,
                    "query_rank": int(query_rank),
                    "phrase_index": phrase_index,
                    "phrase_rank": int(phrase_rank),
                    "box_score": float(query_score),
                    "det_score": float(phrase_score),
                }
            )
    return candidates, [int(index) for index in top_box_indexes.tolist()]


@torch.no_grad()
def _encode_text_bank(
    *,
    clip_encoder: ClipTokenEncoder,
    prompts: Sequence[str],
    device: str,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    token_tensors = []
    mask_tensors = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="encoding phrase tokens"):
        token_ids = clip_encoder.tokenizer(list(prompts[start : start + batch_size]))
        text_tokens, text_mask = clip_encoder.encode_text_tokens(token_ids)
        token_tensors.append(text_tokens.cpu())
        mask_tensors.append(text_mask.cpu())
    return torch.cat(token_tensors, dim=0), torch.cat(mask_tensors, dim=0)


@torch.no_grad()
def _score_candidate_chunks(
    *,
    verifier: TokenCrossVerifier,
    image_tokens_by_query: Mapping[int, torch.Tensor],
    text_tokens: torch.Tensor,
    text_masks: torch.Tensor,
    candidates: Sequence[Mapping[str, Any]],
    device: str,
    batch_size: int,
) -> List[float]:
    logits: List[float] = []
    for start in range(0, len(candidates), batch_size):
        chunk = list(candidates[start : start + batch_size])
        image_batch = torch.stack([image_tokens_by_query[int(item["query_index"])] for item in chunk], dim=0).to(device)
        phrase_indexes = torch.tensor([int(item["phrase_index"]) for item in chunk], dtype=torch.long)
        text_batch = text_tokens[phrase_indexes].to(device)
        mask_batch = text_masks[phrase_indexes].to(device)
        detector_scores = torch.tensor([float(item["det_score"]) for item in chunk], dtype=torch.float32, device=device)
        chunk_logits = verifier(image_batch, text_batch, mask_batch, detector_scores)
        logits.extend(float(value) for value in chunk_logits.detach().cpu().tolist())
    return logits


def rerank(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    checkpoint_path = args.verifier_checkpoint
    verifier, checkpoint = _load_verifier(checkpoint_path, args.device)
    crop_margin = args.crop_margin if args.crop_margin is not None else float(checkpoint.get("crop_margin", 0.25))
    prompt_template = args.prompt_template or str(checkpoint.get("prompt_template", "the described target is {phrase}"))

    clip_encoder = ClipTokenEncoder(
        backend=args.clip_backend or str(checkpoint.get("clip_backend", "open_clip")),
        model_name=args.clip_model or str(checkpoint.get("clip_model", "ViT-L-14")),
        pretrained=args.clip_pretrained or str(checkpoint.get("clip_pretrained", "openai")),
        openai_clip_model=args.openai_clip_model or str(checkpoint.get("openai_clip_model", "ViT-L/14")),
        device=args.device,
    )
    print(f"loaded token backend: {clip_encoder.backend_name}")

    annotation = _load_json(args.annotation)
    image_infos = {int(item["id"]): item for item in annotation["images"]}
    categories = _load_categories(annotation, args.phrases_json)
    category_ids = [category_id for category_id, _ in categories]
    category_names = [name for _, name in categories]
    prompts = [prompt_template.format(phrase=name) for name in category_names]
    text_tokens, text_masks = _encode_text_bank(
        clip_encoder=clip_encoder,
        prompts=prompts,
        device=args.device,
        batch_size=args.text_batch_size,
    )

    vlm_query_embedding = None
    if args.score_mode == "score_ensemble":
        vlm_query_embedding = _load_vlm_query_embedding(args.vlm_query_embedding)

    image_ids = sorted(image_infos)
    if args.max_images is not None:
        image_ids = image_ids[: args.max_images]

    results: List[Dict[str, Any]] = []
    processed_image_ids: List[int] = []
    missing_outputs = 0
    missing_images = 0
    invalid_crops = 0
    processed_crops = 0

    for image_id in tqdm(image_ids, desc="token-verifier reranking"):
        image_info = image_infos[image_id]
        file_name = str(image_info["file_name"])
        saved_path = _saved_prediction_path(args.saved_output_dir, file_name)
        if not saved_path.exists():
            missing_outputs += 1
            continue
        width = int(image_info.get("width", 0))
        height = int(image_info.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError(f"Image {image_id} has invalid size {width}x{height}.")

        boxes, logits, roi_features = _load_saved_prediction(saved_path)
        scores = _candidate_scores(
            logits,
            roi_features,
            args=args,
            vlm_query_embedding=vlm_query_embedding,
        )
        candidates, top_box_indexes = _build_candidates_for_image(
            image_id=image_id,
            width=width,
            height=height,
            boxes=boxes,
            scores=scores,
            category_ids=category_ids,
            category_names=category_names,
            args=args,
        )
        processed_image_ids.append(image_id)

        image_path = _resolve_image_path(args.image_root, file_name)
        if not image_path.exists():
            missing_images += 1
            continue
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        crop_tensors = []
        crop_query_indexes = []
        query_to_bbox: Dict[int, Sequence[float]] = {}
        for candidate in candidates:
            query_index = int(candidate["query_index"])
            if query_index not in query_to_bbox:
                query_to_bbox[query_index] = candidate["bbox"]

        for query_index in top_box_indexes:
            bbox = query_to_bbox.get(query_index)
            if bbox is None:
                continue
            xyxy = _expanded_xyxy(bbox, width=image_width, height=image_height, margin=crop_margin)
            if xyxy is None:
                invalid_crops += 1
                continue
            crop_tensors.append(clip_encoder.preprocess(image.crop(xyxy)))
            crop_query_indexes.append(query_index)

        if not crop_tensors:
            continue

        image_tokens_by_query: Dict[int, torch.Tensor] = {}
        for start in range(0, len(crop_tensors), args.image_batch_size):
            batch = torch.stack(crop_tensors[start : start + args.image_batch_size], dim=0)
            with torch.no_grad():
                image_tokens = clip_encoder.encode_image_tokens(batch).cpu()
            for offset, query_index in enumerate(crop_query_indexes[start : start + args.image_batch_size]):
                image_tokens_by_query[int(query_index)] = image_tokens[offset]
        processed_crops += len(image_tokens_by_query)

        valid_candidates = [item for item in candidates if int(item["query_index"]) in image_tokens_by_query]
        verifier_logits = _score_candidate_chunks(
            verifier=verifier,
            image_tokens_by_query=image_tokens_by_query,
            text_tokens=text_tokens,
            text_masks=text_masks,
            candidates=valid_candidates,
            device=args.device,
            batch_size=args.verifier_batch_size,
        )

        output_candidates = []
        for candidate, verifier_logit in zip(valid_candidates, verifier_logits):
            old_score = float(candidate["score"])
            fused_score = _fuse_score(
                old_score,
                verifier_logit,
                mode=args.fusion,
                fusion_weight=args.fusion_weight,
            )
            output_candidate = dict(candidate)
            output_candidate["score"] = float(fused_score)
            if args.include_debug_fields:
                output_candidate["verifier_logit"] = float(verifier_logit)
                output_candidate["verifier_score"] = float(_sigmoid(verifier_logit))
            output_candidates.append(output_candidate)

        output_candidates.sort(key=lambda item: float(item["score"]), reverse=True)
        if args.keep_topk_per_image > 0:
            output_candidates = output_candidates[: args.keep_topk_per_image]
        for candidate in output_candidates:
            result = {
                "image_id": int(candidate["image_id"]),
                "category_id": int(candidate["category_id"]),
                "bbox": [float(v) for v in candidate["bbox"]],
                "score": float(candidate["score"]),
            }
            if args.include_debug_fields:
                for key in (
                    "det_score",
                    "box_score",
                    "verifier_logit",
                    "verifier_score",
                    "query_index",
                    "query_rank",
                    "phrase_index",
                    "phrase_rank",
                ):
                    if key in candidate:
                        result[key] = candidate[key]
            results.append(result)

    summary = {
        "args": _jsonable_args(args),
        "num_requested_images": len(image_ids),
        "num_processed_images": len(processed_image_ids),
        "missing_outputs": missing_outputs,
        "missing_images": missing_images,
        "invalid_crops": invalid_crops,
        "processed_crops": processed_crops,
        "output_predictions": len(results),
    }
    return results, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, default=Path("dataset/d3/annotations/d3_intra_full.json"))
    parser.add_argument("--image-root", type=Path, default=Path("dataset/d3/images"))
    parser.add_argument("--phrases-json", type=Path, default=Path("dataset/metadata/d3_phrases.json"))
    parser.add_argument("--saved-output-dir", type=Path, required=True)
    parser.add_argument("--verifier-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--box-topk", type=int, default=300)
    parser.add_argument("--phrase-topk", type=int, default=50)
    parser.add_argument("--keep-topk-per-image", type=int, default=100)
    parser.add_argument("--score-mode", choices=("sigmoid", "score_ensemble"), default="score_ensemble")
    parser.add_argument("--vlm-query-embedding", type=Path, default=Path("dataset/metadata/d3_clip_convnextl_sentences.npy"))
    parser.add_argument("--vlm-temperature", type=float, default=100.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--prompt-template", default=None)
    parser.add_argument("--clip-backend", choices=("auto", "open_clip", "openai_clip"), default=None)
    parser.add_argument("--clip-model", default=None)
    parser.add_argument("--clip-pretrained", default=None)
    parser.add_argument("--openai-clip-model", default=None)
    parser.add_argument("--crop-margin", type=float, default=None)
    parser.add_argument("--fusion", choices=("logit_add", "linear", "replace"), default="logit_add")
    parser.add_argument("--fusion-weight", type=float, default=1.0)
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--verifier-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--include-debug-fields", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval-output-images-only", action="store_true")
    parser.add_argument("--summary-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results, summary = rerank(args)
    _save_json(args.output, results)
    summary_output = args.summary_output or args.output.with_suffix(args.output.suffix + ".summary.json")
    _save_json(summary_output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"saved token-verifier predictions to {args.output}")
    print(f"saved summary to {summary_output}")
    if args.eval:
        eval_image_ids = None
        if args.eval_output_images_only or args.max_images is not None:
            eval_image_ids = sorted({int(result["image_id"]) for result in results})
        _evaluate_coco(args.annotation, results, image_ids=eval_image_ids)


if __name__ == "__main__":
    main()
