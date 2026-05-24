#!/usr/bin/env python3
"""Evaluate a COCO-format prediction JSON.

This is useful for expensive reranking runs where the prediction file already
exists and only COCOeval needs to be rerun. By default, evaluation is restricted
to the image ids present in the prediction JSON, which is the desired behavior
for D3 train/val subset files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def _evaluate(
    annotation_path: Path,
    predictions: Sequence[Mapping[str, Any]],
    *,
    image_ids: Optional[Sequence[int]],
) -> Dict[str, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(annotation_path))
    coco_gt.dataset.setdefault("info", {})
    coco_gt.dataset.setdefault("licenses", [])
    coco_dt = coco_gt.loadRes(list(predictions))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.maxDets = [1, 10, 100]
    if image_ids is not None:
        evaluator.params.imgIds = sorted(set(int(image_id) for image_id in image_ids))
        print(f"evaluating image ids: {len(evaluator.params.imgIds)}")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    keys = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
    return {key: float(value) for key, value in zip(keys, evaluator.stats)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--image-id-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL whose image_id values define the evaluation subset.",
    )
    parser.add_argument(
        "--all-annotation-images",
        action="store_true",
        help="Evaluate over all annotation images instead of prediction image ids.",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    predictions = _load_json(args.predictions)
    if not isinstance(predictions, list):
        raise ValueError(f"Expected prediction JSON list, got {type(predictions).__name__}.")

    image_ids = None
    if args.image_id_jsonl is not None:
        image_ids = _load_image_ids_from_jsonl(args.image_id_jsonl)
    elif not args.all_annotation_images:
        image_ids = sorted({int(pred["image_id"]) for pred in predictions})

    stats = _evaluate(args.annotation, predictions, image_ids=image_ids)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)


if __name__ == "__main__":
    main()
