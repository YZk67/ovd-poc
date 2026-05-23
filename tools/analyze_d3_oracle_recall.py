#!/usr/bin/env python3
"""Analyze D3 proposal recall and phrase-oracle upper bounds.

The detector can fail on D3 for two different reasons:

1. It does not propose boxes that overlap the described objects.
2. It proposes usable boxes, but assigns the wrong phrase/category score.

This script separates these cases from an existing COCO-format prediction file.
It reports class-agnostic proposal recall, same-category recall, and an oracle AP
where top-k proposals are greedily matched to GT boxes and assigned the GT
phrase category.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    boxes = boxes.astype(np.float32, copy=False)
    xyxy = boxes.copy()
    xyxy[:, 2] = xyxy[:, 0] + xyxy[:, 2]
    xyxy[:, 3] = xyxy[:, 1] + xyxy[:, 3]
    return xyxy


def _pairwise_iou_xywh(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    if gt_boxes.size == 0 or pred_boxes.size == 0:
        return np.zeros((gt_boxes.shape[0], pred_boxes.shape[0]), dtype=np.float32)

    gt = _xywh_to_xyxy(gt_boxes)
    pred = _xywh_to_xyxy(pred_boxes)

    x0 = np.maximum(gt[:, None, 0], pred[None, :, 0])
    y0 = np.maximum(gt[:, None, 1], pred[None, :, 1])
    x1 = np.minimum(gt[:, None, 2], pred[None, :, 2])
    y1 = np.minimum(gt[:, None, 3], pred[None, :, 3])

    inter = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)
    gt_area = np.maximum(0.0, gt[:, 2] - gt[:, 0]) * np.maximum(0.0, gt[:, 3] - gt[:, 1])
    pred_area = np.maximum(0.0, pred[:, 2] - pred[:, 0]) * np.maximum(0.0, pred[:, 3] - pred[:, 1])
    union = gt_area[:, None] + pred_area[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def _group_annotations(annotation: Mapping[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in annotation.get("annotations", []):
        if int(ann.get("iscrowd", 0)) == 1:
            continue
        grouped[int(ann["image_id"])].append(dict(ann))
    return grouped


def _group_predictions(predictions: Sequence[Mapping[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        grouped[int(pred["image_id"])].append(dict(pred))
    for preds in grouped.values():
        preds.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return grouped


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


def _recall_from_best_ious(best_ious: Sequence[float], threshold: float) -> float:
    if not best_ious:
        return 0.0
    arr = np.asarray(best_ious, dtype=np.float32)
    return float(np.mean(arr >= threshold))


def _average_recall(best_ious: Sequence[float]) -> float:
    if not best_ious:
        return 0.0
    return float(np.mean([_recall_from_best_ious(best_ious, thr) for thr in IOU_THRESHOLDS]))


def _greedy_oracle_predictions_for_image(
    image_id: int,
    gt_anns: Sequence[Mapping[str, Any]],
    preds: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    if not gt_anns or not preds:
        return []

    gt_boxes = np.asarray([ann["bbox"] for ann in gt_anns], dtype=np.float32)
    pred_boxes = np.asarray([pred["bbox"] for pred in preds], dtype=np.float32)
    ious = _pairwise_iou_xywh(gt_boxes, pred_boxes)

    pairs: List[Tuple[float, int, int]] = []
    for gt_idx in range(ious.shape[0]):
        for pred_idx in range(ious.shape[1]):
            iou = float(ious[gt_idx, pred_idx])
            if iou > 0.0:
                pairs.append((iou, gt_idx, pred_idx))
    pairs.sort(reverse=True, key=lambda item: item[0])

    used_gt = set()
    used_pred = set()
    rows: List[Dict[str, Any]] = []
    for iou, gt_idx, pred_idx in pairs:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        gt = gt_anns[gt_idx]
        pred = preds[pred_idx]
        # Use IoU as the oracle confidence. Low-IoU assignments naturally rank
        # below high-quality matches in COCO AP.
        rows.append(
            {
                "image_id": int(image_id),
                "category_id": int(gt["category_id"]),
                "bbox": [float(v) for v in pred["bbox"]],
                "score": float(iou),
            }
        )
    return rows


def _evaluate_coco_stats(
    annotation_path: Path,
    predictions: Sequence[Mapping[str, Any]],
    *,
    image_ids: Optional[Sequence[int]] = None,
    quiet: bool = True,
) -> Optional[Dict[str, Optional[float]]]:
    if not predictions:
        return None
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as exc:  # pragma: no cover - depends on external env
        print(f"[warn] pycocotools unavailable, skipping oracle AP: {exc}")
        return None

    stream = io.StringIO()
    output_context = contextlib.redirect_stdout(stream) if quiet else contextlib.nullcontext()
    with output_context:
        coco_gt = COCO(str(annotation_path))
        coco_gt.dataset.setdefault("info", {})
        coco_gt.dataset.setdefault("licenses", [])
        coco_dt = coco_gt.loadRes(list(predictions))
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.maxDets = [1, 10, 100]
        if image_ids is not None:
            evaluator.params.imgIds = sorted(set(int(image_id) for image_id in image_ids))
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    stats = evaluator.stats

    def stat(index: int) -> Optional[float]:
        value = float(stats[index])
        return value * 100.0 if value >= 0 else None

    return {
        "oracle_ap": stat(0),
        "oracle_ap50": stat(1),
        "oracle_ap75": stat(2),
        "oracle_aps": stat(3),
        "oracle_apm": stat(4),
        "oracle_apl": stat(5),
    }


def analyze(args: argparse.Namespace) -> List[Dict[str, Any]]:
    annotation = _load_json(args.annotation)
    predictions = _load_json(args.predictions)
    if not isinstance(predictions, list):
        raise ValueError(f"Expected prediction JSON list, got {type(predictions).__name__}.")

    gt_by_image = _group_annotations(annotation)
    pred_by_image = _group_predictions(predictions)
    if args.image_id_jsonl is not None:
        image_ids = _load_image_ids_from_jsonl(args.image_id_jsonl)
    elif args.all_annotation_images:
        image_ids = sorted(set(gt_by_image) | set(pred_by_image))
    else:
        image_ids = sorted(pred_by_image)

    rows: List[Dict[str, Any]] = []
    for topk in args.topk:
        any_best_ious: List[float] = []
        same_cat_best_ious: List[float] = []
        oracle_predictions: List[Dict[str, Any]] = []
        total_predictions = 0

        for image_id in image_ids:
            gt_anns = gt_by_image.get(image_id, [])
            preds = pred_by_image.get(image_id, [])[:topk]
            total_predictions += len(preds)
            if not gt_anns:
                continue

            if not preds:
                any_best_ious.extend([0.0] * len(gt_anns))
                same_cat_best_ious.extend([0.0] * len(gt_anns))
                continue

            gt_boxes = np.asarray([ann["bbox"] for ann in gt_anns], dtype=np.float32)
            pred_boxes = np.asarray([pred["bbox"] for pred in preds], dtype=np.float32)
            ious = _pairwise_iou_xywh(gt_boxes, pred_boxes)
            any_best = ious.max(axis=1) if ious.size else np.zeros((len(gt_anns),), dtype=np.float32)
            any_best_ious.extend(float(v) for v in any_best)

            pred_categories = np.asarray([int(pred["category_id"]) for pred in preds], dtype=np.int64)
            for gt_idx, ann in enumerate(gt_anns):
                same_mask = pred_categories == int(ann["category_id"])
                if np.any(same_mask):
                    same_cat_best_ious.append(float(ious[gt_idx, same_mask].max()))
                else:
                    same_cat_best_ious.append(0.0)

            if not args.skip_oracle_ap:
                oracle_predictions.extend(_greedy_oracle_predictions_for_image(image_id, gt_anns, preds))

        row: Dict[str, Any] = {
            "topk": int(topk),
            "num_images": int(len(image_ids)),
            "num_gt": int(len(any_best_ious)),
            "num_predictions": int(total_predictions),
            "any_mean_best_iou": float(np.mean(any_best_ious)) if any_best_ious else 0.0,
            "any_recall50": _recall_from_best_ious(any_best_ious, 0.5),
            "any_recall75": _recall_from_best_ious(any_best_ious, 0.75),
            "any_recall5095": _average_recall(any_best_ious),
            "same_cat_mean_best_iou": float(np.mean(same_cat_best_ious)) if same_cat_best_ious else 0.0,
            "same_cat_recall50": _recall_from_best_ious(same_cat_best_ious, 0.5),
            "same_cat_recall75": _recall_from_best_ious(same_cat_best_ious, 0.75),
            "same_cat_recall5095": _average_recall(same_cat_best_ious),
        }

        if not args.skip_oracle_ap:
            oracle_stats = _evaluate_coco_stats(
                args.annotation,
                oracle_predictions,
                image_ids=image_ids,
                quiet=not args.verbose_coco,
            )
            if oracle_stats is not None:
                row.update(oracle_stats)
            if args.oracle_output_dir:
                args.oracle_output_dir.mkdir(parents=True, exist_ok=True)
                out_path = args.oracle_output_dir / f"oracle_top{topk}.json"
                with out_path.open("w", encoding="utf-8") as handle:
                    json.dump(oracle_predictions, handle)
                row["oracle_predictions"] = str(out_path)

        rows.append(row)

    return rows


def _parse_topk(value: str) -> List[int]:
    topk = [int(item) for item in value.split(",") if item.strip()]
    if not topk:
        raise argparse.ArgumentTypeError("--topk must contain at least one integer.")
    if any(item <= 0 for item in topk):
        raise argparse.ArgumentTypeError("--topk values must be positive.")
    return sorted(set(topk))


def _write_csv(rows: Sequence[Mapping[str, Any]], output: Optional[Path]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        handle = output.open("w", encoding="utf-8", newline="")
    else:
        import sys

        handle = sys.stdout

    with contextlib.ExitStack() as stack:
        if output:
            stack.enter_context(handle)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True, help="D3 COCO annotation JSON.")
    parser.add_argument("--predictions", type=Path, required=True, help="COCO-format prediction JSON.")
    parser.add_argument("--topk", type=_parse_topk, default=_parse_topk("20,50,100,300"))
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--image-id-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL whose image_id values define the evaluation subset.",
    )
    parser.add_argument(
        "--all-annotation-images",
        action="store_true",
        help="Use every annotated image instead of defaulting to prediction image ids.",
    )
    parser.add_argument("--oracle-output-dir", type=Path, default=None)
    parser.add_argument("--skip-oracle-ap", action="store_true", help="Only compute recall metrics.")
    parser.add_argument("--verbose-coco", action="store_true", help="Show pycocotools evaluation logs.")
    args = parser.parse_args()

    rows = analyze(args)
    _write_csv(rows, args.output)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
