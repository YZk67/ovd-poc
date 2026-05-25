#!/usr/bin/env python3
"""Run a GroundingDINO proposal source on D3 and write COCO predictions.

This mirrors tools/run_d3_owlv2_proposals.py so source diagnostics can compare
direct AP and oracle recall with the same downstream scripts. GroundingDINO is
caption based, so phrase chunks are joined into a single caption and returned
text labels are mapped back to the D3 phrase bank.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image
from tqdm import tqdm

try:
    from run_d3_owlv2_proposals import (
        _dtype_from_arg,
        _evaluate_coco,
        _jsonable_args,
        _load_categories,
        _load_image_ids_from_jsonl,
        _load_json,
        _move_inputs,
        _per_image_path,
        _resolve_image_path,
        _save_json,
        _xyxy_to_xywh,
    )
except ModuleNotFoundError:  # pragma: no cover - useful when imported as tools.*
    from tools.run_d3_owlv2_proposals import (
        _dtype_from_arg,
        _evaluate_coco,
        _jsonable_args,
        _load_categories,
        _load_image_ids_from_jsonl,
        _load_json,
        _move_inputs,
        _per_image_path,
        _resolve_image_path,
        _save_json,
        _xyxy_to_xywh,
    )


_NON_WORD_RE = re.compile(r"[^a-z0-9]+")


def _normalize_label(value: str) -> str:
    text = str(value).lower()
    for prefix in (
        "a photo of ",
        "an image of ",
        "the described target is ",
        "the target is ",
    ):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    text = _NON_WORD_RE.sub(" ", text)
    return " ".join(text.split())


def _token_set(value: str) -> set:
    return set(_normalize_label(value).split())


def _caption_from_prompts(prompts: Sequence[str], *, lowercase: bool) -> str:
    pieces = []
    for prompt in prompts:
        piece = str(prompt).strip().rstrip(".")
        if piece:
            pieces.append(piece)
    caption = ". ".join(pieces) + "."
    return caption.lower() if lowercase else caption


def _build_chunk_index(
    *,
    category_ids: Sequence[int],
    category_names: Sequence[str],
    prompts: Sequence[str],
) -> Tuple[Dict[str, int], List[Tuple[int, str, set]]]:
    exact: Dict[str, int] = {}
    candidates: List[Tuple[int, str, set]] = []
    for category_id, name, prompt in zip(category_ids, category_names, prompts):
        for value in (name, prompt):
            norm = _normalize_label(value)
            if norm:
                exact.setdefault(norm, int(category_id))
        norm_name = _normalize_label(name)
        if norm_name:
            candidates.append((int(category_id), norm_name, set(norm_name.split())))
    return exact, candidates


def _match_grounded_label(
    label: Any,
    *,
    exact_index: Mapping[str, int],
    candidates: Sequence[Tuple[int, str, set]],
    min_score: float,
) -> Tuple[Optional[int], float, str]:
    norm = _normalize_label(str(label))
    if not norm:
        return None, 0.0, norm
    if norm in exact_index:
        return int(exact_index[norm]), 1.0, norm

    tokens = set(norm.split())
    if not tokens:
        return None, 0.0, norm

    best_category_id: Optional[int] = None
    best_score = 0.0
    for category_id, candidate_norm, candidate_tokens in candidates:
        if not candidate_tokens:
            continue
        overlap = len(tokens & candidate_tokens)
        if overlap == 0:
            continue
        jaccard = overlap / len(tokens | candidate_tokens)
        coverage = overlap / len(tokens)
        substring_bonus = 0.15 if norm in candidate_norm or candidate_norm in norm else 0.0
        score = max(jaccard, 0.75 * coverage) + substring_bonus
        if score > best_score:
            best_score = float(score)
            best_category_id = int(category_id)

    if best_category_id is None or best_score < min_score:
        return None, float(best_score), norm
    return best_category_id, float(best_score), norm


def _load_hf_groundingdino(args: argparse.Namespace):
    try:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        model_cls = AutoModelForZeroShotObjectDetection
    except Exception:
        try:
            from transformers import AutoProcessor, GroundingDinoForObjectDetection

            model_cls = GroundingDinoForObjectDetection
        except Exception as exc:  # pragma: no cover - depends on remote env
            raise ImportError(
                "transformers with GroundingDINO support is required. In the lami env, "
                "install/upgrade transformers if this import fails."
            ) from exc

    dtype = _dtype_from_arg(args.dtype, device=args.device)
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = model_cls.from_pretrained(args.model_name, torch_dtype=dtype)
    model.to(args.device)
    model.eval()
    print(f"loaded GroundingDINO model: {args.model_name} dtype={dtype} device={args.device}")
    return processor, model, dtype


def _post_process_grounded(
    *,
    processor,
    outputs,
    inputs: Mapping[str, Any],
    target_sizes: torch.Tensor,
    args: argparse.Namespace,
) -> Mapping[str, Any]:
    post_process = getattr(processor, "post_process_grounded_object_detection", None)
    if post_process is None:
        raise RuntimeError(
            "The loaded GroundingDINO processor has no "
            "post_process_grounded_object_detection method. Upgrade transformers."
        )

    input_ids = inputs.get("input_ids")
    attempts = (
        lambda: post_process(
            outputs=outputs,
            input_ids=input_ids,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=target_sizes,
        ),
        lambda: post_process(
            outputs,
            input_ids,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=target_sizes,
        ),
        lambda: post_process(
            outputs=outputs,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=target_sizes,
        ),
    )
    last_error: Optional[Exception] = None
    for attempt in attempts:
        try:
            return attempt()[0]
        except TypeError as exc:
            last_error = exc
    raise RuntimeError("Could not call GroundingDINO post-process API.") from last_error


@torch.no_grad()
def _predict_one_image(
    *,
    processor,
    model,
    dtype: torch.dtype,
    image: Image.Image,
    image_id: int,
    category_ids: Sequence[int],
    category_names: Sequence[str],
    prompts: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    predictions: List[Dict[str, Any]] = []
    stats = {"raw_detections": 0, "unmatched_labels": 0}
    width, height = image.size
    target_sizes = torch.tensor([[height, width]], device=args.device)

    for start in range(0, len(prompts), args.text_chunk_size):
        chunk_prompts = list(prompts[start : start + args.text_chunk_size])
        chunk_category_ids = list(category_ids[start : start + args.text_chunk_size])
        chunk_category_names = list(category_names[start : start + args.text_chunk_size])
        exact_index, fuzzy_candidates = _build_chunk_index(
            category_ids=chunk_category_ids,
            category_names=chunk_category_names,
            prompts=chunk_prompts,
        )
        caption = _caption_from_prompts(chunk_prompts, lowercase=args.lowercase_caption)
        inputs = processor(images=image, text=caption, return_tensors="pt")
        inputs = _move_inputs(inputs, device=args.device, dtype=dtype)
        outputs = model(**inputs)
        result = _post_process_grounded(
            processor=processor,
            outputs=outputs,
            inputs=inputs,
            target_sizes=target_sizes,
            args=args,
        )

        boxes = result["boxes"].detach().cpu().tolist()
        scores = result["scores"].detach().cpu().tolist()
        labels = result.get("labels", result.get("text_labels", [""] * len(boxes)))
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().tolist()

        for box, score, label in zip(boxes, scores, labels):
            stats["raw_detections"] += 1
            category_id: Optional[int]
            match_score: float
            norm_label: str
            if isinstance(label, int):
                offset = int(label)
                category_id = chunk_category_ids[offset] if 0 <= offset < len(chunk_category_ids) else None
                match_score = 1.0 if category_id is not None else 0.0
                norm_label = str(label)
            else:
                category_id, match_score, norm_label = _match_grounded_label(
                    label,
                    exact_index=exact_index,
                    candidates=fuzzy_candidates,
                    min_score=args.label_match_threshold,
                )
            if category_id is None:
                stats["unmatched_labels"] += 1
                if args.drop_unmatched_labels:
                    continue
                continue

            prediction = {
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox": _xyxy_to_xywh(box),
                "score": float(score),
            }
            if args.include_debug_fields:
                prediction.update(
                    {
                        "grounding_label": str(label),
                        "normalized_label": norm_label,
                        "label_match_score": float(match_score),
                    }
                )
            predictions.append(prediction)

    predictions.sort(key=lambda item: float(item["score"]), reverse=True)
    if args.keep_topk_per_image > 0:
        predictions = predictions[: args.keep_topk_per_image]
    return predictions, stats


def run(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    annotation = _load_json(args.annotation)
    image_infos = {int(item["id"]): item for item in annotation["images"]}
    categories = _load_categories(annotation, args.phrases_json)
    category_ids = [category_id for category_id, _ in categories]
    category_names = [name for _, name in categories]
    prompts = [args.prompt_template.format(phrase=name) for name in category_names]

    image_ids = sorted(image_infos)
    if args.image_id_jsonl is not None:
        selected_ids = set(_load_image_ids_from_jsonl(args.image_id_jsonl))
        image_ids = [image_id for image_id in image_ids if image_id in selected_ids]
        print(f"using image ids from {args.image_id_jsonl}: {len(image_ids)}")
    if args.max_images is not None:
        image_ids = image_ids[: args.max_images]

    processor, model, dtype = _load_hf_groundingdino(args)

    if args.per_image_output_dir is not None:
        args.per_image_output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    processed_image_ids: List[int] = []
    missing_images = 0
    reused_images = 0
    raw_detections = 0
    unmatched_labels = 0

    for image_id in tqdm(image_ids, desc="running GroundingDINO on D3"):
        if args.per_image_output_dir is not None:
            cached_path = _per_image_path(args.per_image_output_dir, image_id)
            if args.reuse_per_image and cached_path.exists():
                cached = _load_json(cached_path)
                if not isinstance(cached, list):
                    raise ValueError(f"Expected list in cached prediction {cached_path}.")
                results.extend(cached)
                processed_image_ids.append(image_id)
                reused_images += 1
                continue

        image_info = image_infos[image_id]
        image_path = _resolve_image_path(args.image_root, str(image_info["file_name"]))
        if not image_path.exists():
            missing_images += 1
            if args.drop_missing_images:
                continue
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        predictions, image_stats = _predict_one_image(
            processor=processor,
            model=model,
            dtype=dtype,
            image=image,
            image_id=image_id,
            category_ids=category_ids,
            category_names=category_names,
            prompts=prompts,
            args=args,
        )
        raw_detections += int(image_stats["raw_detections"])
        unmatched_labels += int(image_stats["unmatched_labels"])
        results.extend(predictions)
        processed_image_ids.append(image_id)
        if args.per_image_output_dir is not None:
            _save_json(_per_image_path(args.per_image_output_dir, image_id), predictions)

    summary = {
        "args": _jsonable_args(args),
        "num_categories": len(category_ids),
        "num_requested_images": len(image_ids),
        "num_processed_images": len(processed_image_ids),
        "missing_images": missing_images,
        "reused_images": reused_images,
        "raw_detections": raw_detections,
        "unmatched_labels": unmatched_labels,
        "output_predictions": len(results),
    }
    return results, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, default=Path("dataset/d3/annotations/d3_intra_full.json"))
    parser.add_argument("--image-root", type=Path, default=Path("dataset/d3/images"))
    parser.add_argument("--phrases-json", type=Path, default=Path("dataset/metadata/d3_phrases.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--per-image-output-dir", type=Path, default=None)
    parser.add_argument("--reuse-per-image", action="store_true")
    parser.add_argument("--image-id-jsonl", type=Path, default=None)
    parser.add_argument("--model-name", default="IDEA-Research/grounding-dino-base")
    parser.add_argument(
        "--prompt-template",
        default="{phrase}",
        help="GroundingDINO works best with short phrases joined by periods.",
    )
    parser.add_argument("--text-chunk-size", type=int, default=16)
    parser.add_argument("--box-threshold", type=float, default=0.15)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument(
        "--label-match-threshold",
        type=float,
        default=0.45,
        help="Minimum fuzzy match score for mapping returned text labels back to D3 phrases.",
    )
    parser.add_argument("--keep-topk-per-image", type=int, default=300)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--drop-missing-images", action="store_true")
    parser.add_argument(
        "--drop-unmatched-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop GroundingDINO text labels that cannot be mapped back to a D3 phrase.",
    )
    parser.add_argument(
        "--lowercase-caption",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Lowercase the joined caption before tokenization.",
    )
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval-output-images-only", action="store_true")
    parser.add_argument("--include-debug-fields", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results, summary = run(args)
    _save_json(args.output, results)
    summary_output = args.summary_output or args.output.with_suffix(args.output.suffix + ".summary.json")
    _save_json(summary_output, summary, pretty=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"saved GroundingDINO predictions to {args.output}")
    print(f"saved summary to {summary_output}")
    if args.eval:
        eval_image_ids = None
        if args.eval_output_images_only or args.max_images is not None or args.image_id_jsonl is not None:
            eval_image_ids = sorted({int(result["image_id"]) for result in results})
        _evaluate_coco(args.annotation, results, image_ids=eval_image_ids)


if __name__ == "__main__":
    main()
