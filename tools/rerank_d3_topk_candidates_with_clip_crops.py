#!/usr/bin/env python3
"""
Rerank D3 top-k detector candidates with OpenCLIP crop-text scores.

This is an offline crop-level verifier for the top300 x top50 candidate path.
It does not train or modify the detector. It reads per-image detector dumps
saved by the D3 ROI-feature export config:

  pred_logits:       [1, num_queries, num_phrases]
  pred_boxes:        [1, num_queries, 4] normalized cxcywh boxes
  roi_features_ori:  optional, only needed for score_ensemble candidate scores

For each image it rebuilds:

  top K boxes by detector max phrase score
  x top M phrase candidates for each box

Then it crops each selected box, scores the crop against D3 phrase prompts with
OpenCLIP, fuses detector and crop-text scores, writes COCO-format predictions,
and can optionally run COCO bbox evaluation.
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


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, separators=(",", ":"))


def _jsonable_args(args: argparse.Namespace) -> Dict[str, Any]:
    values = {}
    for key, value in vars(args).items():
        values[key] = str(value) if isinstance(value, Path) else value
    return values


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


def _load_image_ids_from_jsonl(path: Path) -> List[int]:
    image_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image_ids.add(int(row["image_id"]))
    return sorted(image_ids)


def _saved_prediction_path(saved_output_dir: Path, file_name: str) -> Path:
    path = Path(file_name)
    return saved_output_dir / path.with_suffix(".pth").name


def _load_saved_prediction(path: Path) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")

    missing = [key for key in ("pred_boxes", "pred_logits") if key not in data]
    if missing:
        raise KeyError(
            f"{path} is missing {missing}. Re-export detector dumps with "
            "model.save_roi_features_only=False."
        )

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
    if boxes.shape[0] != logits.shape[0]:
        raise ValueError(
            f"{path} query count mismatch: boxes={tuple(boxes.shape)}, logits={tuple(logits.shape)}."
        )
    if roi_features is not None and roi_features.shape[0] != boxes.shape[0]:
        raise ValueError(
            f"{path} ROI query count mismatch: boxes={tuple(boxes.shape)}, "
            f"roi={tuple(roi_features.shape)}."
        )
    return boxes.cpu(), logits.cpu(), roi_features.cpu() if roi_features is not None else None


def _load_vlm_query_embedding(path: Path) -> torch.Tensor:
    array = np.load(path)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D VLM query embedding, got shape {array.shape}.")
    tensor = torch.from_numpy(array).float()
    return F.normalize(tensor, p=2, dim=-1)


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
    model.eval()
    return model, preprocess, open_clip.tokenize


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
    for start in tqdm(range(0, len(texts), batch_size), desc="encoding texts"):
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


def _build_candidates_for_image(
    *,
    image_id: int,
    file_name: str,
    width: int,
    height: int,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    category_ids: Sequence[int],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    if scores.shape[-1] != len(category_ids):
        raise ValueError(
            f"{file_name} has {scores.shape[-1]} classes, but annotation/phrases define "
            f"{len(category_ids)} categories."
        )

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


def rerank(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    annotation = _load_json(args.annotation)
    image_infos = {int(item["id"]): item for item in annotation["images"]}
    categories = _load_categories(annotation, args.phrases_json)
    category_ids = [category_id for category_id, _ in categories]
    prompts = [args.prompt_template.format(phrase=phrase) for _, phrase in categories]

    image_ids = sorted(image_infos)
    if args.image_id_jsonl is not None:
        selected_ids = set(_load_image_ids_from_jsonl(args.image_id_jsonl))
        image_ids = [image_id for image_id in image_ids if image_id in selected_ids]
        print(f"using image ids from {args.image_id_jsonl}: {len(image_ids)}")
    if args.max_images is not None:
        image_ids = image_ids[: args.max_images]

    vlm_query_embedding = None
    if args.score_mode == "score_ensemble":
        vlm_query_embedding = _load_vlm_query_embedding(args.vlm_query_embedding)

    device = args.device
    model = preprocess = tokenizer = text_features = None
    if not args.skip_rerank:
        model, preprocess, tokenizer = _load_openclip(args.model, args.pretrained, device)
        text_features = _encode_texts(
            model,
            tokenizer,
            prompts,
            batch_size=args.text_batch_size,
            device=device,
        )

    results: List[Dict[str, Any]] = []
    processed_image_ids: List[int] = []
    missing_outputs = 0
    missing_images = 0
    invalid_crops = 0
    processed_crops = 0
    emitted_candidates = 0

    for image_id in tqdm(image_ids, desc="reranking top-k candidates"):
        image_info = image_infos[image_id]
        file_name = str(image_info["file_name"])
        saved_path = _saved_prediction_path(args.saved_output_dir, file_name)
        if not saved_path.exists():
            missing_outputs += 1
            continue

        width = int(image_info.get("width", 0))
        height = int(image_info.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError(f"Image {image_id} has invalid width/height: {width}x{height}.")

        boxes, logits, roi_features = _load_saved_prediction(saved_path)
        scores = _candidate_scores(
            logits,
            roi_features,
            args=args,
            vlm_query_embedding=vlm_query_embedding,
        )
        candidates, top_box_indexes = _build_candidates_for_image(
            image_id=image_id,
            file_name=file_name,
            width=width,
            height=height,
            boxes=boxes,
            scores=scores,
            category_ids=category_ids,
            args=args,
        )
        processed_image_ids.append(image_id)

        if args.skip_rerank:
            output_candidates = candidates
        else:
            assert model is not None and preprocess is not None and text_features is not None
            image_path = _resolve_image_path(args.image_root, file_name)
            if not image_path.exists():
                missing_images += 1
                if args.drop_missing_images:
                    continue
                output_candidates = candidates
            else:
                image = Image.open(image_path).convert("RGB")
                image_width, image_height = image.size

                crop_tensors: List[torch.Tensor] = []
                crop_query_indexes: List[int] = []
                query_to_crop_row: Dict[int, int] = {}
                query_to_bbox: Dict[int, Sequence[float]] = {}
                for candidate in candidates:
                    query_index = int(candidate["query_index"])
                    if query_index in query_to_bbox:
                        continue
                    query_to_bbox[query_index] = candidate["bbox"]

                for query_index in top_box_indexes:
                    bbox = query_to_bbox.get(query_index)
                    if bbox is None:
                        continue
                    xyxy = _expanded_xyxy(
                        bbox,
                        width=image_width,
                        height=image_height,
                        margin=args.crop_margin,
                    )
                    if xyxy is None:
                        invalid_crops += 1
                        continue
                    query_to_crop_row[query_index] = len(crop_tensors)
                    crop_query_indexes.append(query_index)
                    crop_tensors.append(preprocess(image.crop(xyxy)))

                if crop_tensors:
                    crop_features = _encode_crops(
                        model,
                        crop_tensors,
                        batch_size=args.image_batch_size,
                        device=device,
                    )
                    clip_score_matrix = crop_features @ text_features.t()
                    processed_crops += len(crop_tensors)

                    output_candidates = []
                    for candidate in candidates:
                        query_index = int(candidate["query_index"])
                        crop_row = query_to_crop_row.get(query_index)
                        if crop_row is None:
                            if args.drop_invalid_crops:
                                continue
                            output_candidates.append(dict(candidate))
                            continue
                        phrase_index = int(candidate["phrase_index"])
                        clip_score = float(clip_score_matrix[crop_row, phrase_index])
                        old_score = float(candidate["score"])
                        fused_score = _fuse_score(
                            old_score,
                            clip_score,
                            mode=args.fusion,
                            fusion_weight=args.fusion_weight,
                            clip_scale=args.clip_scale,
                            clip_center=args.clip_center,
                        )
                        output_candidate = dict(candidate)
                        output_candidate["score"] = float(fused_score)
                        if args.include_debug_fields:
                            output_candidate["clip_score"] = clip_score
                        output_candidates.append(output_candidate)
                else:
                    output_candidates = [] if args.drop_invalid_crops else candidates

        output_candidates.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        if args.keep_topk_per_image > 0:
            output_candidates = output_candidates[: args.keep_topk_per_image]
        emitted_candidates += len(output_candidates)

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
                    "clip_score",
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
        "output_predictions": emitted_candidates,
    }
    return results, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, default=Path("dataset/d3/annotations/d3_intra_full.json"))
    parser.add_argument("--image-root", type=Path, default=Path("dataset/d3/images"))
    parser.add_argument("--phrases-json", type=Path, default=Path("dataset/metadata/d3_phrases.json"))
    parser.add_argument(
        "--saved-output-dir",
        type=Path,
        required=True,
        help="Directory of per-image .pth dumps with pred_logits and pred_boxes.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output COCO-format result JSON.")
    parser.add_argument(
        "--image-id-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL file with image_id fields for held-out subset evaluation.",
    )
    parser.add_argument("--box-topk", type=int, default=300)
    parser.add_argument("--phrase-topk", type=int, default=50)
    parser.add_argument("--keep-topk-per-image", type=int, default=100)
    parser.add_argument("--score-mode", choices=("sigmoid", "score_ensemble"), default="score_ensemble")
    parser.add_argument(
        "--vlm-query-embedding",
        type=Path,
        default=Path("dataset/metadata/d3_clip_convnextl_sentences.npy"),
    )
    parser.add_argument("--vlm-temperature", type=float, default=100.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--prompt-template", default="the described target is {phrase}")
    parser.add_argument("--model", default="convnext_large_d_320", help="OpenCLIP model name.")
    parser.add_argument("--pretrained", default="laion2b_s29b_b131k_ft_soup", help="OpenCLIP pretrained tag.")
    parser.add_argument("--crop-margin", type=float, default=0.25)
    parser.add_argument("--fusion", choices=("logit_add", "linear", "replace"), default="logit_add")
    parser.add_argument("--fusion-weight", type=float, default=0.25)
    parser.add_argument("--clip-scale", type=float, default=10.0)
    parser.add_argument("--clip-center", type=float, default=0.25)
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--skip-rerank", action="store_true", help="Only rebuild detector candidates and write top-k.")
    parser.add_argument(
        "--drop-missing-images",
        action="store_true",
        help="Drop candidate predictions when an image file is missing.",
    )
    parser.add_argument(
        "--drop-invalid-crops",
        action="store_true",
        help="Drop candidate predictions for boxes that cannot be cropped.",
    )
    parser.add_argument("--include-debug-fields", action="store_true")
    parser.add_argument("--eval", action="store_true", help="Run COCO bbox evaluation after writing output.")
    parser.add_argument(
        "--eval-output-images-only",
        action="store_true",
        help=(
            "Restrict COCO evaluation to image ids present in the output. "
            "This is automatically enabled when --max-images or --image-id-jsonl is set."
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional summary JSON path. Defaults to OUTPUT with .summary.json suffix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results, summary = rerank(args)
    _save_json(args.output, results)
    summary_output = args.summary_output
    if summary_output is None:
        summary_output = args.output.with_suffix(args.output.suffix + ".summary.json")
    _save_json(summary_output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"saved reranked predictions to {args.output}")
    print(f"saved summary to {summary_output}")

    if args.eval:
        eval_image_ids = None
        if args.eval_output_images_only or args.max_images is not None or args.image_id_jsonl is not None:
            eval_image_ids = sorted({int(result["image_id"]) for result in results})
        _evaluate_coco(args.annotation, results, image_ids=eval_image_ids)


if __name__ == "__main__":
    main()
