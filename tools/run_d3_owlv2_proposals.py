#!/usr/bin/env python3
"""Run an OWLv2 proposal source on D3 and write COCO-format predictions.

This is a detector-source diagnostic, not part of LaMI-DETR training. It runs a
HuggingFace OWLv2 model over D3 phrases, writes D3 COCO result JSON, and can run
COCO bbox evaluation on the same subset. Use the `lami` environment because it
may need HuggingFace/transformers packages and model downloads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image
from tqdm import tqdm


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, data: Any, *, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if pretty:
            json.dump(data, handle, indent=2, sort_keys=True)
        else:
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


def _xyxy_to_xywh(box: Sequence[float]) -> List[float]:
    return [
        float(box[0]),
        float(box[1]),
        float(max(0.0, float(box[2]) - float(box[0]))),
        float(max(0.0, float(box[3]) - float(box[1]))),
    ]


def _dtype_from_arg(value: str, *, device: str) -> torch.dtype:
    value = value.lower()
    if value == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype={value!r}.")


def _move_inputs(inputs: Mapping[str, Any], *, device: str, dtype: torch.dtype) -> Dict[str, Any]:
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


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


def _per_image_path(output_dir: Path, image_id: int) -> Path:
    return output_dir / f"{int(image_id):012d}.json"


def _load_hf_owlv2(args: argparse.Namespace):
    try:
        from transformers import AutoProcessor, Owlv2ForObjectDetection
    except Exception as exc:  # pragma: no cover - depends on remote env
        raise ImportError(
            "transformers with OWLv2 support is required. In the lami env, install/upgrade "
            "transformers if this import fails."
        ) from exc

    dtype = _dtype_from_arg(args.dtype, device=args.device)
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = Owlv2ForObjectDetection.from_pretrained(args.model_name, torch_dtype=dtype)
    model.to(args.device)
    model.eval()
    print(f"loaded OWLv2 model: {args.model_name} dtype={dtype} device={args.device}")
    return processor, model, dtype


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
) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []
    width, height = image.size
    target_sizes = torch.tensor([[height, width]], device=args.device)

    for start in range(0, len(prompts), args.text_chunk_size):
        chunk_prompts = list(prompts[start : start + args.text_chunk_size])
        inputs = processor(text=[chunk_prompts], images=image, return_tensors="pt")
        inputs = _move_inputs(inputs, device=args.device, dtype=dtype)
        outputs = model(**inputs)
        result = processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=args.threshold,
        )[0]

        boxes = result["boxes"].detach().cpu().tolist()
        scores = result["scores"].detach().cpu().tolist()
        labels = result["labels"].detach().cpu().tolist()
        for box, score, label in zip(boxes, scores, labels):
            category_index = start + int(label)
            if category_index < 0 or category_index >= len(category_ids):
                continue
            predictions.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(category_ids[category_index]),
                    "bbox": _xyxy_to_xywh(box),
                    "score": float(score),
                }
            )

    predictions.sort(key=lambda item: float(item["score"]), reverse=True)
    if args.keep_topk_per_image > 0:
        predictions = predictions[: args.keep_topk_per_image]
    return predictions


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

    processor, model, dtype = _load_hf_owlv2(args)

    if args.per_image_output_dir is not None:
        args.per_image_output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    processed_image_ids: List[int] = []
    missing_images = 0
    reused_images = 0

    for image_id in tqdm(image_ids, desc="running OWLv2 on D3"):
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
        predictions = _predict_one_image(
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
    parser.add_argument(
        "--per-image-output-dir",
        type=Path,
        default=None,
        help="Optional directory for per-image cached predictions so long runs can resume.",
    )
    parser.add_argument("--reuse-per-image", action="store_true", help="Reuse cached per-image predictions if present.")
    parser.add_argument(
        "--image-id-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL file with image_id fields for held-out subset evaluation.",
    )
    parser.add_argument("--model-name", default="google/owlv2-large-patch14-ensemble")
    parser.add_argument("--prompt-template", default="a photo of {phrase}")
    parser.add_argument("--text-chunk-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--keep-topk-per-image", type=int, default=300)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--drop-missing-images", action="store_true")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval-output-images-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results, summary = run(args)
    _save_json(args.output, results)
    summary_output = args.summary_output or args.output.with_suffix(args.output.suffix + ".summary.json")
    _save_json(summary_output, summary, pretty=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"saved OWLv2 predictions to {args.output}")
    print(f"saved summary to {summary_output}")
    if args.eval:
        eval_image_ids = None
        if args.eval_output_images_only or args.max_images is not None or args.image_id_jsonl is not None:
            eval_image_ids = sorted({int(result["image_id"]) for result in results})
        _evaluate_coco(args.annotation, results, image_ids=eval_image_ids)


if __name__ == "__main__":
    main()
