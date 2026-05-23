#!/usr/bin/env python3
"""
Build region-description verifier pairs from D3 predictions and GT.

This is the first training-side step after fixed text-prototype calibration:
turn detector predictions into binary region/phrase pairs.

Positive pair:
  By default, predicted box + predicted phrase c, where IoU(pred, GT of c)
  >= pos threshold. With --positive-source proposal_iou, the target phrase is
  taken from the best-overlapping GT box regardless of the detector's predicted
  phrase/category. This is useful for class-agnostic proposal sources such as
  OWLv2 where the box is good but the phrase score is misranked.

Negative pairs:
  1. same_phrase_bad_box: predicted class c but IoU(pred, GT of c) <= neg threshold.
  2. wrong_phrase_same_region: a positive predicted box paired with another
     phrase that does not overlap that box.

The output JSONL can later be used to train a lightweight verifier:
  verifier(region_feature, description_feature) -> match / non-match.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm


def _load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
            count += 1
    return count


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    xyxy = boxes.astype(np.float32).copy()
    xyxy[:, 2] = xyxy[:, 0] + xyxy[:, 2]
    xyxy[:, 3] = xyxy[:, 1] + xyxy[:, 3]
    return xyxy


def _pairwise_iou_xywh(box: Sequence[float], gt_boxes: np.ndarray) -> np.ndarray:
    if gt_boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)

    pred = _xywh_to_xyxy(np.asarray([box], dtype=np.float32))[0]
    gt = _xywh_to_xyxy(gt_boxes.astype(np.float32))

    x0 = np.maximum(pred[0], gt[:, 0])
    y0 = np.maximum(pred[1], gt[:, 1])
    x1 = np.minimum(pred[2], gt[:, 2])
    y1 = np.minimum(pred[3], gt[:, 3])

    inter_w = np.maximum(0.0, x1 - x0)
    inter_h = np.maximum(0.0, y1 - y0)
    inter = inter_w * inter_h

    pred_area = max(0.0, pred[2] - pred[0]) * max(0.0, pred[3] - pred[1])
    gt_area = np.maximum(0.0, gt[:, 2] - gt[:, 0]) * np.maximum(0.0, gt[:, 3] - gt[:, 1])
    union = pred_area + gt_area - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def _group_predictions(predictions: Sequence[Mapping[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        grouped[int(pred["image_id"])].append(dict(pred))
    for preds in grouped.values():
        preds.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return grouped


def _group_annotations(annotation: Mapping[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in annotation.get("annotations", []):
        grouped[int(ann["image_id"])].append(dict(ann))
    return grouped


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


def _best_iou_for_category(
    bbox: Sequence[float],
    gt_by_cat: Mapping[int, List[Mapping[str, Any]]],
    category_id: int,
) -> Tuple[float, Optional[int]]:
    anns = gt_by_cat.get(category_id, [])
    if not anns:
        return 0.0, None
    gt_boxes = np.asarray([ann["bbox"] for ann in anns], dtype=np.float32)
    ious = _pairwise_iou_xywh(bbox, gt_boxes)
    if len(ious) == 0:
        return 0.0, None
    idx = int(ious.argmax())
    best_iou = float(ious[idx])
    return best_iou, int(anns[idx]["id"]) if best_iou > 0.0 else None


def _best_iou_any(
    bbox: Sequence[float],
    gt_anns: Sequence[Mapping[str, Any]],
) -> Tuple[float, Optional[int], Optional[int]]:
    if not gt_anns:
        return 0.0, None, None
    gt_boxes = np.asarray([ann["bbox"] for ann in gt_anns], dtype=np.float32)
    ious = _pairwise_iou_xywh(bbox, gt_boxes)
    if len(ious) == 0:
        return 0.0, None, None
    idx = int(ious.argmax())
    best_iou = float(ious[idx])
    if best_iou <= 0.0:
        return best_iou, None, None
    ann = gt_anns[idx]
    return best_iou, int(ann["category_id"]), int(ann["id"])


def _base_row(
    *,
    split: str,
    image_info: Mapping[str, Any],
    pred: Mapping[str, Any],
    target_category_id: int,
    phrase: str,
    label: int,
    target_iou: float,
    matched_gt_id: Optional[int],
    best_any_iou: float,
    best_any_category_id: Optional[int],
    best_any_gt_id: Optional[int],
    negative_type: Optional[str],
    positive_source: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "split": split,
        "image_id": int(image_info["id"]),
        "file_name": str(image_info["file_name"]),
        "width": int(image_info.get("width", 0)),
        "height": int(image_info.get("height", 0)),
        "bbox": [float(v) for v in pred["bbox"]],
        "detector_category_id": int(pred["category_id"]),
        "target_category_id": int(target_category_id),
        "phrase": phrase,
        "detector_score": float(pred.get("score", 0.0)),
        "label": int(label),
        "target_iou": float(target_iou),
        "matched_gt_id": matched_gt_id,
        "best_any_iou": float(best_any_iou),
        "best_any_category_id": best_any_category_id,
        "best_any_gt_id": best_any_gt_id,
        "negative_type": negative_type,
        "positive_source": positive_source,
    }


def _choose_wrong_categories(
    *,
    rng: random.Random,
    bbox: Sequence[float],
    gt_by_cat: Mapping[int, List[Mapping[str, Any]]],
    present_categories: Sequence[int],
    all_categories: Sequence[int],
    positive_category_id: int,
    count: int,
    neg_iou_thresh: float,
) -> List[Tuple[int, float, Optional[int], str]]:
    selected: List[Tuple[int, float, Optional[int], str]] = []

    present = [cat for cat in present_categories if cat != positive_category_id]
    rng.shuffle(present)
    for cat in present:
        target_iou, matched_gt_id = _best_iou_for_category(bbox, gt_by_cat, cat)
        if target_iou <= neg_iou_thresh:
            selected.append((cat, target_iou, matched_gt_id, "present_wrong_phrase"))
        if len(selected) >= count:
            return selected

    global_candidates = [cat for cat in all_categories if cat != positive_category_id and cat not in present]
    rng.shuffle(global_candidates)
    for cat in global_candidates:
        target_iou, matched_gt_id = _best_iou_for_category(bbox, gt_by_cat, cat)
        if target_iou <= neg_iou_thresh:
            selected.append((cat, target_iou, matched_gt_id, "global_wrong_phrase"))
        if len(selected) >= count:
            break
    return selected


def _split_images(
    image_ids: Sequence[int],
    *,
    val_ratio: float,
    seed: int,
) -> Dict[int, str]:
    ids = list(image_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    val_count = int(round(len(ids) * val_ratio))
    val_ids = set(ids[:val_count])
    return {image_id: ("val" if image_id in val_ids else "train") for image_id in image_ids}


def _jsonable_args(args: argparse.Namespace) -> Dict[str, Any]:
    values = {}
    for key, value in vars(args).items():
        values[key] = str(value) if isinstance(value, Path) else value
    return values


def build_pairs(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(args.seed)
    annotation = _load_json(args.annotation)
    predictions = _load_json(args.predictions)
    if not isinstance(predictions, list):
        raise ValueError(f"Expected prediction JSON list, got {type(predictions).__name__}.")

    image_infos = {int(item["id"]): item for item in annotation["images"]}
    gt_by_image = _group_annotations(annotation)
    pred_by_image = _group_predictions(predictions)
    categories = _load_categories(annotation, args.phrases_json)
    all_categories = sorted(categories)

    image_ids = sorted(set(pred_by_image) & set(image_infos))
    if args.max_images is not None:
        image_ids = image_ids[: args.max_images]
    split_by_image = _split_images(image_ids, val_ratio=args.val_ratio, seed=args.seed)

    rows: List[Dict[str, Any]] = []
    stats = Counter()

    for image_id in tqdm(image_ids, desc="building verifier pairs"):
        split = split_by_image[image_id]
        image_info = image_infos[image_id]
        gt_anns = gt_by_image.get(image_id, [])
        gt_by_cat: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
        for ann in gt_anns:
            gt_by_cat[int(ann["category_id"])].append(ann)
        present_categories = sorted(gt_by_cat)

        preds = pred_by_image[image_id][: args.pred_topk_per_image]
        positives: List[Dict[str, Any]] = []
        same_phrase_negs: List[Dict[str, Any]] = []

        for pred in preds:
            pred_cat = int(pred["category_id"])
            if pred_cat not in categories:
                continue

            target_iou, matched_gt_id = _best_iou_for_category(pred["bbox"], gt_by_cat, pred_cat)
            best_any_iou, best_any_cat, best_any_gt_id = _best_iou_any(pred["bbox"], gt_anns)

            positive_category_id = pred_cat
            positive_iou = target_iou
            positive_gt_id = matched_gt_id
            positive_source = "detector_label"
            if args.positive_source == "proposal_iou":
                positive_category_id = int(best_any_cat) if best_any_cat is not None else pred_cat
                positive_iou = best_any_iou
                positive_gt_id = best_any_gt_id
                positive_source = "proposal_iou"

            if positive_iou >= args.pos_iou_thresh and positive_category_id in categories:
                positives.append(
                    _base_row(
                        split=split,
                        image_info=image_info,
                        pred=pred,
                        target_category_id=positive_category_id,
                        phrase=categories[positive_category_id],
                        label=1,
                        target_iou=positive_iou,
                        matched_gt_id=positive_gt_id,
                        best_any_iou=best_any_iou,
                        best_any_category_id=best_any_cat,
                        best_any_gt_id=best_any_gt_id,
                        negative_type=None,
                        positive_source=positive_source,
                    )
                )
            elif target_iou <= args.neg_iou_thresh:
                same_phrase_negs.append(
                    _base_row(
                        split=split,
                        image_info=image_info,
                        pred=pred,
                        target_category_id=pred_cat,
                        phrase=categories[pred_cat],
                        label=0,
                        target_iou=target_iou,
                        matched_gt_id=matched_gt_id,
                        best_any_iou=best_any_iou,
                        best_any_category_id=best_any_cat,
                        best_any_gt_id=best_any_gt_id,
                        negative_type="same_phrase_bad_box",
                    )
                )

        if args.max_pos_per_image > 0:
            positives = positives[: args.max_pos_per_image]
        rows.extend(positives)
        stats[(split, "positive")] += len(positives)

        wrong_phrase_negs: List[Dict[str, Any]] = []
        for pos in positives:
            wrong_cats = _choose_wrong_categories(
                rng=rng,
                bbox=pos["bbox"],
                gt_by_cat=gt_by_cat,
                present_categories=present_categories,
                all_categories=all_categories,
                positive_category_id=int(pos["target_category_id"]),
                count=args.wrong_phrase_neg_per_pos,
                neg_iou_thresh=args.neg_iou_thresh,
            )
            pred_for_row = {
                "bbox": pos["bbox"],
                "category_id": pos["detector_category_id"],
                "score": pos["detector_score"],
            }
            for wrong_cat, target_iou, matched_gt_id, source in wrong_cats:
                wrong_phrase_negs.append(
                    _base_row(
                        split=split,
                        image_info=image_info,
                        pred=pred_for_row,
                        target_category_id=wrong_cat,
                        phrase=categories[wrong_cat],
                        label=0,
                        target_iou=target_iou,
                        matched_gt_id=matched_gt_id,
                        best_any_iou=float(pos["best_any_iou"]),
                        best_any_category_id=pos["best_any_category_id"],
                        best_any_gt_id=pos["best_any_gt_id"],
                        negative_type=f"wrong_phrase_same_region:{source}",
                    )
                )

        if args.max_same_phrase_neg_per_image > 0:
            same_phrase_negs = same_phrase_negs[: args.max_same_phrase_neg_per_image]

        negatives = same_phrase_negs + wrong_phrase_negs
        if args.max_neg_per_image > 0 and len(negatives) > args.max_neg_per_image:
            negatives = rng.sample(negatives, args.max_neg_per_image)
        rows.extend(negatives)

        for row in negatives:
            stats[(split, row["negative_type"])] += 1

    summary = {
        "num_images": len(image_ids),
        "num_rows": len(rows),
        "num_train": sum(1 for row in rows if row["split"] == "train"),
        "num_val": sum(1 for row in rows if row["split"] == "val"),
        "num_positive": sum(1 for row in rows if row["label"] == 1),
        "num_negative": sum(1 for row in rows if row["label"] == 0),
        "stats": {f"{split}:{kind}": int(value) for (split, kind), value in sorted(stats.items())},
        "args": _jsonable_args(args),
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Detector COCO-format result JSON.",
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path("dataset/d3/annotations/d3_intra_full.json"),
        help="D3 COCO annotation JSON.",
    )
    parser.add_argument(
        "--phrases-json",
        type=Path,
        default=Path("dataset/metadata/d3_phrases.json"),
        help="Ordered D3 phrase list.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/d3/verifier_pairs_w075"),
        help="Directory for train.jsonl, val.jsonl, all.jsonl, and summary.json.",
    )
    parser.add_argument("--pred-topk-per-image", type=int, default=100)
    parser.add_argument(
        "--positive-source",
        choices=("detector_label", "proposal_iou"),
        default="detector_label",
        help=(
            "How to form positive pairs. detector_label requires the predicted category to match GT; "
            "proposal_iou pairs any high-IoU proposal box with the best-overlapping GT phrase."
        ),
    )
    parser.add_argument("--pos-iou-thresh", type=float, default=0.5)
    parser.add_argument("--neg-iou-thresh", type=float, default=0.3)
    parser.add_argument("--wrong-phrase-neg-per-pos", type=int, default=2)
    parser.add_argument("--max-pos-per-image", type=int, default=50, help="Use 0 to keep all positives.")
    parser.add_argument("--max-same-phrase-neg-per-image", type=int, default=50)
    parser.add_argument("--max-neg-per-image", type=int, default=150)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary = build_pairs(args)

    output_dir = args.output_dir
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]

    all_count = _write_jsonl(output_dir / "all.jsonl", rows)
    train_count = _write_jsonl(output_dir / "train.jsonl", train_rows)
    val_count = _write_jsonl(output_dir / "val.jsonl", val_rows)
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"saved all:   {output_dir / 'all.jsonl'} ({all_count})")
    print(f"saved train: {output_dir / 'train.jsonl'} ({train_count})")
    print(f"saved val:   {output_dir / 'val.jsonl'} ({val_count})")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
