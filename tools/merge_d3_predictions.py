#!/usr/bin/env python3
"""Merge multiple D3 COCO-format prediction files into one candidate pool.

The downstream D3 rerankers select class-agnostic proposals by sorting detector
scores first, so source score calibration matters. This tool applies optional
per-source affine/power transforms before merging and keeps the highest scoring
candidates per image.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


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


def _group_by_image(predictions: Iterable[Mapping[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        grouped[int(pred["image_id"])].append(dict(pred))
    return grouped


def _transform_score(score: float, *, scale: float, bias: float, power: float) -> float:
    score = max(0.0, float(score))
    if power != 1.0:
        score = score**power
    return max(0.0, score * scale + bias)


def _pad_values(values: Optional[Sequence[Any]], *, count: int, default: Any) -> List[Any]:
    if values is None:
        return [default for _ in range(count)]
    padded = list(values)
    if len(padded) > count:
        raise ValueError(f"Got {len(padded)} values for {count} inputs.")
    padded.extend(default for _ in range(count - len(padded)))
    return padded


def _limit_sorted(rows: List[Dict[str, Any]], topk: int) -> List[Dict[str, Any]]:
    rows.sort(
        key=lambda item: (
            -float(item.get("score", 0.0)),
            str(item.get("source", "")),
            int(item.get("category_id", -1)),
        )
    )
    if topk > 0:
        return rows[:topk]
    return rows


def _jsonable_args(args: argparse.Namespace) -> Dict[str, Any]:
    values = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            values[key] = str(value)
        elif isinstance(value, list):
            values[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            values[key] = value
    return values


def merge_predictions(args: argparse.Namespace) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    inputs = [Path(path) for path in args.inputs]
    count = len(inputs)
    source_names = _pad_values(args.source_names, count=count, default=None)
    source_names = [
        str(name) if name is not None else path.parent.parent.name or path.stem
        for name, path in zip(source_names, inputs)
    ]
    score_scales = [float(value) for value in _pad_values(args.score_scales, count=count, default=1.0)]
    score_biases = [float(value) for value in _pad_values(args.score_biases, count=count, default=0.0)]
    score_powers = [float(value) for value in _pad_values(args.score_powers, count=count, default=1.0)]

    all_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    source_counts = Counter()
    source_kept_counts = Counter()

    for index, path in enumerate(inputs):
        predictions = _load_json(path)
        if not isinstance(predictions, list):
            raise ValueError(f"Expected prediction list in {path}, got {type(predictions).__name__}.")
        source = str(source_names[index])
        grouped = _group_by_image(predictions)
        source_counts[source] += len(predictions)
        for image_id, rows in grouped.items():
            transformed: List[Dict[str, Any]] = []
            for row in rows:
                score = _transform_score(
                    float(row.get("score", 0.0)),
                    scale=score_scales[index],
                    bias=score_biases[index],
                    power=score_powers[index],
                )
                item = dict(row)
                item["score"] = score
                item["source"] = source
                item["source_score"] = float(row.get("score", 0.0))
                transformed.append(item)
            transformed = _limit_sorted(transformed, args.per_source_topk_per_image)
            source_kept_counts[source] += len(transformed)
            all_by_image[int(image_id)].extend(transformed)

    merged: List[Dict[str, Any]] = []
    for image_id in sorted(all_by_image):
        merged.extend(_limit_sorted(all_by_image[image_id], args.keep_topk_per_image))

    summary = {
        "args": _jsonable_args(args),
        "num_inputs": count,
        "num_images": len(all_by_image),
        "num_predictions": len(merged),
        "source_counts": dict(sorted(source_counts.items())),
        "source_kept_counts": dict(sorted(source_kept_counts.items())),
    }
    return merged, summary


def _evaluate_coco(annotation_path: Path, results: Sequence[Mapping[str, Any]], *, image_ids: Optional[Sequence[int]]) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True, help="COCO result JSON files to merge.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--source-names", nargs="*", default=None)
    parser.add_argument("--score-scales", nargs="*", type=float, default=None)
    parser.add_argument("--score-biases", nargs="*", type=float, default=None)
    parser.add_argument("--score-powers", nargs="*", type=float, default=None)
    parser.add_argument("--per-source-topk-per-image", type=int, default=300)
    parser.add_argument("--keep-topk-per-image", type=int, default=1000)
    parser.add_argument("--annotation", type=Path, default=Path("dataset/d3/annotations/d3_intra_full.json"))
    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "--eval-output-images-only",
        action="store_true",
        help="Restrict COCOeval to image ids covered by merged predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged, summary = merge_predictions(args)
    _save_json(args.output, merged)
    summary_output = args.summary_output or args.output.with_suffix(args.output.suffix + ".summary.json")
    _save_json(summary_output, summary, pretty=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"saved merged predictions to {args.output}")
    print(f"saved summary to {summary_output}")
    if args.eval:
        eval_image_ids = sorted({int(row["image_id"]) for row in merged}) if args.eval_output_images_only else None
        _evaluate_coco(args.annotation, merged, image_ids=eval_image_ids)


if __name__ == "__main__":
    main()
