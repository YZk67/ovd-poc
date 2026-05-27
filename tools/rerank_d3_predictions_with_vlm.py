#!/usr/bin/env python3
"""
Rerank D3 box/phrase candidates with a vision-language model.

This is a slow smoke-test scorer for the current D3 reranking plateau. It uses
the same candidate preselection as the token verifier, then asks a VLM whether
the candidate box matches the phrase. VLM scores are cached in JSONL so fusion
weights can be swept without rerunning generation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rerank_d3_predictions_with_clip_crops import (  # noqa: E402
    _evaluate_coco,
    _expanded_base_scores_from_category_scores,
    _expanded_cache_to_arrays,
    _expanded_category_scores,
    _expanded_score_cache_path,
    _fuse_verifier_logit_matrix,
    _group_predictions,
    _load_categories,
    _load_image_ids_from_jsonl,
    _load_json,
    _resolve_image_path,
    _save_json,
    _select_class_agnostic_proposals,
    _top_matrix_indices,
)
from train_d3_token_cross_verifier import _expanded_xyxy, _jsonable_args  # noqa: E402


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _logit(score: float) -> float:
    score = min(max(float(score), 1e-6), 1.0 - 1e-6)
    return math.log(score / (1.0 - score))


def _fuse_score(base_score: float, vlm_score: float, *, mode: str, fusion_weight: float) -> float:
    vlm_score = min(max(float(vlm_score), 1e-6), 1.0 - 1e-6)
    if mode == "logit_add":
        return _sigmoid(_logit(base_score) + fusion_weight * _logit(vlm_score))
    if mode == "linear":
        return (1.0 - fusion_weight) * float(base_score) + fusion_weight * vlm_score
    if mode == "replace":
        return vlm_score
    raise ValueError(f"Unsupported VLM fusion mode: {mode}")


def _load_candidate_cache(cache_dir: Path, image_id: int) -> Optional[Mapping[str, Any]]:
    path = _expanded_score_cache_path(cache_dir, image_id)
    if not path.exists():
        return None
    try:
        cache = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        cache = torch.load(path, map_location="cpu")
    if not isinstance(cache, Mapping):
        return None
    return cache


def _candidate_mlp_scores(*, base_scores: np.ndarray, pair_signal: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    return _fuse_verifier_logit_matrix(
        base_scores,
        pair_signal,
        mode=args.candidate_verifier_fusion,
        fusion_weight=args.candidate_verifier_fusion_weight,
    )


def _select_candidate_pairs(
    *,
    proposals: Sequence[Mapping[str, Any]],
    base_scores: np.ndarray,
    candidate_scores: np.ndarray,
    category_ids: Sequence[int],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for proposal_idx, category_idx in _top_matrix_indices(candidate_scores, args.candidate_topk_per_image):
        base_score = float(base_scores[proposal_idx, category_idx])
        candidate_score = float(candidate_scores[proposal_idx, category_idx])
        token_base_score = candidate_score if args.vlm_base_score == "candidate_score" else base_score
        pairs.append(
            {
                "proposal_idx": int(proposal_idx),
                "category_idx": int(category_idx),
                "category_id": int(category_ids[category_idx]),
                "base_score": base_score,
                "candidate_score": candidate_score,
                "vlm_base_score": float(token_base_score),
            }
        )
    return pairs


def _build_uncached_inputs(
    *,
    preds: Sequence[Mapping[str, Any]],
    category_to_row: Mapping[int, int],
    category_ids: Sequence[int],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    proposals = _select_class_agnostic_proposals(
        preds,
        topk=args.proposal_topk_per_image,
        nms_thresh=args.proposal_nms_thresh,
    )
    category_scores = _expanded_category_scores(
        proposals,
        preds,
        category_to_row=category_to_row,
        num_categories=len(category_ids),
        match_iou=args.category_score_match_iou,
    )
    base_scores = _expanded_base_scores_from_category_scores(
        proposals,
        category_scores,
        num_categories=len(category_ids),
        mode=args.expanded_base_score,
        missing_category_score_scale=args.missing_category_score_scale,
    )
    return proposals, category_scores, base_scores


def _bbox_signature(bbox: Sequence[float]) -> str:
    return ",".join(f"{float(value):.2f}" for value in bbox)


def _cache_key(image_id: int, category_id: int, bbox: Sequence[float]) -> str:
    return f"{int(image_id)}|{int(category_id)}|{_bbox_signature(bbox)}"


def _load_vlm_score_cache(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    cache: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = str(row.get("key") or _cache_key(int(row["image_id"]), int(row["category_id"]), row["bbox"]))
            cache[key] = row
    return cache


def _append_vlm_score_cache(path: Optional[Path], rows: Sequence[Mapping[str, Any]]) -> None:
    if path is None or not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def _parse_vlm_score(text: str, default: float) -> Tuple[float, bool]:
    lowered = text.strip().lower()
    score_match = re.search(r'"?score"?\s*[:=]\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))', lowered)
    if score_match is None:
        score_match = re.search(r"\b([+-]?(?:\d+(?:\.\d+)?|\.\d+))\b", lowered)
    if score_match is not None:
        value = float(score_match.group(1))
        if 1.0 < value <= 100.0:
            value /= 100.0
        return min(max(value, 0.0), 1.0), True
    if re.search(r"\b(yes|true|match|matches|present)\b", lowered):
        return 1.0, True
    if re.search(r"\b(no|false|mismatch|not present|absent)\b", lowered):
        return 0.0, True
    return float(default), False


def _draw_boxed_image(image: Image.Image, xyxy: Tuple[int, int, int, int], *, line_width: int) -> Image.Image:
    boxed = image.copy()
    draw = ImageDraw.Draw(boxed)
    x0, y0, x1, y1 = xyxy
    width = max(2, int(line_width))
    for offset in range(width):
        draw.rectangle((x0 - offset, y0 - offset, x1 + offset, y1 + offset), outline=(255, 0, 0), width=1)
    return boxed


def _make_vlm_images(
    image: Image.Image,
    bbox: Sequence[float],
    *,
    image_mode: str,
    crop_margin: float,
    box_line_width: int,
) -> Optional[List[Image.Image]]:
    width, height = image.size
    tight_xyxy = _expanded_xyxy(bbox, width=width, height=height, margin=0.0)
    crop_xyxy = _expanded_xyxy(bbox, width=width, height=height, margin=crop_margin)
    if tight_xyxy is None or crop_xyxy is None:
        return None
    if image_mode == "boxed":
        return [_draw_boxed_image(image, tight_xyxy, line_width=box_line_width)]
    if image_mode == "crop":
        return [image.crop(crop_xyxy)]
    if image_mode == "both":
        return [
            _draw_boxed_image(image, tight_xyxy, line_width=box_line_width),
            image.crop(crop_xyxy),
        ]
    raise ValueError(f"Unsupported image mode: {image_mode}")


def _prompt_for_pair(phrase: str, *, image_mode: str) -> str:
    if image_mode == "crop":
        visual_context = "The image is a crop of one candidate detection region."
    elif image_mode == "both":
        visual_context = (
            "The first image is the full scene with the candidate region outlined in red. "
            "The second image is a crop of that same region."
        )
    else:
        visual_context = "The image is the full scene with one candidate region outlined in red."
    return (
        f"{visual_context}\n"
        "Judge only the candidate region, not other objects outside it.\n"
        f'Description: "{phrase}"\n'
        'Return only JSON in this exact format: {"score": 0.0}\n'
        "Use 1.0 for a clear match, 0.5 for uncertain/partial, and 0.0 for a clear mismatch."
    )


class QwenVLScorer:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise ImportError(
                "qwen-vl-utils is required for Qwen2.5-VL scoring. "
                "Install it in the lami env with: pip install qwen-vl-utils"
            ) from exc
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError("A recent transformers build with Qwen2.5-VL support is required.") from exc

        dtype = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.dtype]
        kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if dtype != "auto":
            kwargs["torch_dtype"] = dtype
        if args.device_map:
            kwargs["device_map"] = args.device_map
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_name, **kwargs)
        if not args.device_map:
            self.model.to(args.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
        self.process_vision_info = process_vision_info
        self.device = args.device
        self.max_new_tokens = int(args.max_new_tokens)

    @torch.no_grad()
    def score(self, tasks: Sequence[Mapping[str, Any]]) -> List[str]:
        conversations = []
        for task in tasks:
            content = [{"type": "image", "image": image} for image in task["images"]]
            content.append({"type": "text", "text": str(task["prompt"])})
            conversations.append([{"role": "user", "content": content}])
        texts = [
            self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            for conversation in conversations
        ]
        image_inputs, video_inputs = self.process_vision_info(conversations)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def _score_missing_pairs(
    *,
    scorer: QwenVLScorer,
    image: Image.Image,
    image_id: int,
    proposals: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    categories_by_id: Mapping[int, str],
    cache: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[int, int, int]:
    tasks: List[Dict[str, Any]] = []
    invalid_regions = 0
    for pair in pairs:
        proposal = proposals[int(pair["proposal_idx"])]
        key = _cache_key(image_id, int(pair["category_id"]), proposal["bbox"])
        if key in cache:
            continue
        images = _make_vlm_images(
            image,
            proposal["bbox"],
            image_mode=args.image_mode,
            crop_margin=args.crop_margin,
            box_line_width=args.box_line_width,
        )
        if images is None:
            invalid_regions += 1
            continue
        phrase = categories_by_id[int(pair["category_id"])]
        tasks.append(
            {
                "key": key,
                "image_id": int(image_id),
                "category_id": int(pair["category_id"]),
                "bbox": [float(value) for value in proposal["bbox"]],
                "phrase": phrase,
                "prompt": _prompt_for_pair(phrase, image_mode=args.image_mode),
                "images": images,
            }
        )

    scored_pairs = 0
    parse_failures = 0
    for start in range(0, len(tasks), args.vlm_batch_size):
        batch = tasks[start : start + args.vlm_batch_size]
        responses = scorer.score(batch)
        rows = []
        for task, response in zip(batch, responses):
            vlm_score, parse_ok = _parse_vlm_score(response, default=args.failed_score)
            parse_failures += 0 if parse_ok else 1
            row = {
                "key": task["key"],
                "image_id": task["image_id"],
                "category_id": task["category_id"],
                "bbox": task["bbox"],
                "vlm_score": float(vlm_score),
                "parse_ok": bool(parse_ok),
            }
            if args.store_responses:
                row["phrase"] = task["phrase"]
                row["response"] = response
            cache[task["key"]] = row
            rows.append(row)
        _append_vlm_score_cache(args.vlm_score_cache, rows)
        scored_pairs += len(rows)
    return scored_pairs, invalid_regions, parse_failures


def rerank(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    annotation = _load_json(args.annotation)
    predictions = _load_json(args.predictions)
    if not isinstance(predictions, list):
        raise ValueError(f"Expected prediction JSON list, got {type(predictions).__name__}.")

    image_infos = {int(item["id"]): item for item in annotation["images"]}
    categories_by_id = _load_categories(annotation, args.phrases_json)
    category_ids = sorted(categories_by_id)
    category_to_row = {category_id: idx for idx, category_id in enumerate(category_ids)}
    grouped = _group_predictions(predictions)

    image_ids = sorted(grouped)
    if args.image_id_jsonl is not None:
        selected_ids = set(_load_image_ids_from_jsonl(args.image_id_jsonl))
        image_ids = [image_id for image_id in image_ids if image_id in selected_ids]
        print(f"using image ids from {args.image_id_jsonl}: {len(image_ids)}")
    if args.max_images is not None:
        image_ids = image_ids[: args.max_images]

    vlm_cache = _load_vlm_score_cache(args.vlm_score_cache)
    scorer: Optional[QwenVLScorer] = None
    results: List[Dict[str, Any]] = []
    processed_image_ids: List[int] = []
    missing_images = 0
    missing_candidate_caches = 0
    invalid_regions = 0
    scored_vlm_pairs = 0
    reused_vlm_pairs = 0
    parse_failures = 0
    output_without_vlm = 0

    for image_id in tqdm(image_ids, desc="VLM reranking predictions"):
        preds = grouped[image_id]
        image_info = image_infos.get(image_id)
        if image_info is None:
            continue
        processed_image_ids.append(image_id)

        if args.candidate_source == "cache_signal":
            if args.candidate_score_cache_dir is None:
                raise ValueError("--candidate-source cache_signal requires --candidate-score-cache-dir.")
            cache = _load_candidate_cache(args.candidate_score_cache_dir, image_id)
            if cache is None:
                missing_candidate_caches += 1
                continue
            proposals, category_scores, pair_signal, _ = _expanded_cache_to_arrays(cache)
            if pair_signal is None:
                raise ValueError(f"Candidate cache for image {image_id} has no pair_signal.")
            base_scores = _expanded_base_scores_from_category_scores(
                proposals,
                category_scores,
                num_categories=len(category_ids),
                mode=args.expanded_base_score,
                missing_category_score_scale=args.missing_category_score_scale,
            )
            candidate_scores = _candidate_mlp_scores(base_scores=base_scores, pair_signal=pair_signal, args=args)
        else:
            proposals, _, base_scores = _build_uncached_inputs(
                preds=preds,
                category_to_row=category_to_row,
                category_ids=category_ids,
                args=args,
            )
            candidate_scores = base_scores

        if not proposals:
            continue
        candidate_pairs = _select_candidate_pairs(
            proposals=proposals,
            base_scores=base_scores,
            candidate_scores=candidate_scores,
            category_ids=category_ids,
            args=args,
        )
        if not candidate_pairs:
            continue

        image_path = _resolve_image_path(args.image_root, str(image_info["file_name"]))
        if not image_path.exists():
            missing_images += 1
            continue

        image = Image.open(image_path).convert("RGB")
        if not args.score_only_cached:
            if scorer is None:
                scorer = QwenVLScorer(args)
                print(f"loaded VLM scorer: {args.model_name}")
            scored, invalid, failed = _score_missing_pairs(
                scorer=scorer,
                image=image,
                image_id=image_id,
                proposals=proposals,
                pairs=candidate_pairs,
                categories_by_id=categories_by_id,
                cache=vlm_cache,
                args=args,
            )
            scored_vlm_pairs += scored
            invalid_regions += invalid
            parse_failures += failed
        image.close()

        output_candidates = []
        for pair in candidate_pairs:
            proposal = proposals[int(pair["proposal_idx"])]
            key = _cache_key(image_id, int(pair["category_id"]), proposal["bbox"])
            row = vlm_cache.get(key)
            if row is None:
                if args.drop_unscored:
                    continue
                vlm_score = args.failed_score
                output_without_vlm += 1
            else:
                vlm_score = float(row["vlm_score"])
                reused_vlm_pairs += 1
            fused_score = _fuse_score(
                float(pair["vlm_base_score"]),
                vlm_score,
                mode=args.vlm_fusion,
                fusion_weight=args.vlm_fusion_weight,
            )
            item = {
                "image_id": int(image_id),
                "category_id": int(pair["category_id"]),
                "bbox": [float(value) for value in proposal["bbox"]],
                "score": float(fused_score),
            }
            if args.include_debug_fields:
                item.update(
                    {
                        "proposal_score": float(proposal["score"]),
                        "source_category_id": int(proposal["source_category_id"]),
                        "base_score": float(pair["base_score"]),
                        "candidate_score": float(pair["candidate_score"]),
                        "vlm_base_score": float(pair["vlm_base_score"]),
                        "vlm_score": float(vlm_score),
                        "proposal_idx": int(pair["proposal_idx"]),
                        "category_idx": int(pair["category_idx"]),
                    }
                )
            output_candidates.append(item)

        output_candidates.sort(key=lambda item: float(item["score"]), reverse=True)
        if args.keep_topk_per_image > 0:
            output_candidates = output_candidates[: args.keep_topk_per_image]
        results.extend(output_candidates)

    summary = {
        "args": _jsonable_args(args),
        "num_requested_images": len(image_ids),
        "num_processed_images": len(processed_image_ids),
        "missing_images": missing_images,
        "missing_candidate_caches": missing_candidate_caches,
        "invalid_regions": invalid_regions,
        "scored_vlm_pairs": scored_vlm_pairs,
        "reused_or_loaded_vlm_pairs": reused_vlm_pairs,
        "parse_failures": parse_failures,
        "output_without_vlm": output_without_vlm,
        "output_predictions": len(results),
    }
    return results, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, default=Path("dataset/d3/annotations/d3_intra_full.json"))
    parser.add_argument("--image-root", type=Path, default=Path("dataset/d3/images"))
    parser.add_argument("--phrases-json", type=Path, default=Path("dataset/metadata/d3_phrases.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-id-jsonl", type=Path, default=None)
    parser.add_argument("--candidate-source", choices=("category_score", "cache_signal"), default="cache_signal")
    parser.add_argument("--candidate-score-cache-dir", type=Path, default=None)
    parser.add_argument("--candidate-topk-per-image", type=int, default=50)
    parser.add_argument("--candidate-verifier-fusion", choices=("logit_add", "linear", "replace"), default="logit_add")
    parser.add_argument("--candidate-verifier-fusion-weight", type=float, default=0.45)
    parser.add_argument("--proposal-topk-per-image", type=int, default=100)
    parser.add_argument("--proposal-nms-thresh", type=float, default=0.9)
    parser.add_argument("--expanded-base-score", choices=("objectness", "category_score"), default="category_score")
    parser.add_argument("--category-score-match-iou", type=float, default=0.9)
    parser.add_argument("--missing-category-score-scale", type=float, default=0.3)
    parser.add_argument("--vlm-base-score", choices=("candidate_score", "category_score"), default="candidate_score")
    parser.add_argument("--vlm-fusion", choices=("logit_add", "linear", "replace"), default="logit_add")
    parser.add_argument("--vlm-fusion-weight", type=float, default=0.25)
    parser.add_argument("--keep-topk-per-image", type=int, default=50)
    parser.add_argument("--crop-margin", type=float, default=0.25)
    parser.add_argument("--image-mode", choices=("boxed", "crop", "both"), default="boxed")
    parser.add_argument("--box-line-width", type=int, default=6)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--vlm-batch-size", type=int, default=4)
    parser.add_argument("--vlm-score-cache", type=Path, default=None)
    parser.add_argument("--score-only-cached", action="store_true")
    parser.add_argument("--drop-unscored", action="store_true")
    parser.add_argument("--failed-score", type=float, default=0.5)
    parser.add_argument("--store-responses", action="store_true")
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
    print(f"saved VLM-reranked predictions to {args.output}")
    print(f"saved summary to {summary_output}")
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
        _evaluate_coco(args.annotation, results, image_ids=eval_image_ids)


if __name__ == "__main__":
    main()
