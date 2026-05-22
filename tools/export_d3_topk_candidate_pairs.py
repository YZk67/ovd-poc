#!/usr/bin/env python3
"""
Export D3 top-k box/phrase candidate pairs for reranker training.

This script builds the training distribution used by the top300 box-phrase
reranker. It reads per-image detector dumps containing:

  pred_logits:       [1, num_queries, num_phrases] raw detector logits
  pred_boxes:        [1, num_queries, 4] normalized cxcywh boxes
  roi_features_ori:  optional [1, num_queries, dim] ROI features

For each image it scans:

  top K boxes by max phrase score
  x top M phrases for each selected box

and labels every candidate with:

  box_iou_label:      whether the box matches any D3 GT
  phrase_match_label: whether the box matches GT of the target phrase
  label:              same as phrase_match_label, for binary reranker training

The candidate universe is fully counted in summary.json. To keep JSONL size
manageable, rows are sampled per image by negative type before writing.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _jsonable_args(args: argparse.Namespace) -> Dict[str, Any]:
    values = {}
    for key, value in vars(args).items():
        values[key] = str(value) if isinstance(value, Path) else value
    return values


def _write_row(handles: Mapping[str, Any], row: Mapping[str, Any]) -> None:
    line = json.dumps(row, separators=(",", ":")) + "\n"
    handles["all"].write(line)
    handles[row["split"]].write(line)


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


def _group_annotations(annotation: Mapping[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in annotation.get("annotations", []):
        grouped[int(ann["image_id"])].append(dict(ann))
    return grouped


def _split_images(image_ids: Sequence[int], *, val_ratio: float, seed: int) -> Dict[int, str]:
    ids = list(image_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    val_count = int(round(len(ids) * val_ratio))
    val_ids = set(ids[:val_count])
    return {image_id: ("val" if image_id in val_ids else "train") for image_id in image_ids}


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


def _xyxy_to_xywh(box: np.ndarray) -> List[float]:
    return [
        float(box[0]),
        float(box[1]),
        float(max(0.0, box[2] - box[0])),
        float(max(0.0, box[3] - box[1])),
    ]


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    xyxy = boxes.astype(np.float32).copy()
    xyxy[:, 2] = xyxy[:, 0] + xyxy[:, 2]
    xyxy[:, 3] = xyxy[:, 1] + xyxy[:, 3]
    return xyxy


def _pairwise_iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)

    x0 = np.maximum(box[0], boxes[:, 0])
    y0 = np.maximum(box[1], boxes[:, 1])
    x1 = np.minimum(box[2], boxes[:, 2])
    y1 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x1 - x0)
    inter_h = np.maximum(0.0, y1 - y0)
    inter = inter_w * inter_h

    box_area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def _best_iou_any(
    box_xyxy: np.ndarray,
    gt_anns: Sequence[Mapping[str, Any]],
    gt_xyxy: np.ndarray,
) -> Tuple[float, Optional[int], Optional[int]]:
    if not gt_anns:
        return 0.0, None, None
    ious = _pairwise_iou_xyxy(box_xyxy, gt_xyxy)
    if len(ious) == 0:
        return 0.0, None, None
    idx = int(ious.argmax())
    best_iou = float(ious[idx])
    if best_iou <= 0.0:
        return best_iou, None, None
    ann = gt_anns[idx]
    return best_iou, int(ann["category_id"]), int(ann["id"])


def _best_iou_for_category(
    box_xyxy: np.ndarray,
    gt_by_cat: Mapping[int, Tuple[List[Mapping[str, Any]], np.ndarray]],
    category_id: int,
) -> Tuple[float, Optional[int]]:
    anns_and_boxes = gt_by_cat.get(category_id)
    if anns_and_boxes is None:
        return 0.0, None
    anns, boxes = anns_and_boxes
    ious = _pairwise_iou_xyxy(box_xyxy, boxes)
    if len(ious) == 0:
        return 0.0, None
    idx = int(ious.argmax())
    best_iou = float(ious[idx])
    return best_iou, int(anns[idx]["id"]) if best_iou > 0.0 else None


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


def _negative_type(
    *,
    target_iou: float,
    best_any_iou: float,
    best_any_category_id: Optional[int],
    target_category_id: int,
    pos_iou_thresh: float,
    neg_iou_thresh: float,
) -> str:
    if target_iou >= pos_iou_thresh:
        return "positive"
    if best_any_iou >= pos_iou_thresh and best_any_category_id != target_category_id:
        return "wrong_phrase_good_box"
    if target_iou <= neg_iou_thresh and best_any_iou <= neg_iou_thresh:
        return "background_bad_box"
    if target_iou <= neg_iou_thresh:
        return "same_phrase_bad_box"
    return "ambiguous_iou"


def _sample_rows_for_image(
    rows: Sequence[Dict[str, Any]],
    *,
    rng: random.Random,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row["negative_type"])].append(row)

    selected: List[Dict[str, Any]] = []

    def take(kind: str, limit: int) -> None:
        items = by_type.get(kind, [])
        if limit < 0 or len(items) <= limit:
            selected.extend(items)
            return
        selected.extend(rng.sample(items, limit))

    take("positive", args.max_pos_per_image)
    take("wrong_phrase_good_box", args.max_wrong_phrase_neg_per_image)
    take("same_phrase_bad_box", args.max_same_phrase_neg_per_image)
    take("background_bad_box", args.max_background_neg_per_image)
    if args.include_ambiguous:
        take("ambiguous_iou", args.max_ambiguous_per_image)

    selected.sort(
        key=lambda row: (
            int(row["query_rank"]),
            int(row["phrase_rank"]),
            int(row["target_category_id"]),
        )
    )
    return selected


def _build_gt_indices(
    gt_anns: Sequence[Mapping[str, Any]],
) -> Tuple[np.ndarray, Dict[int, Tuple[List[Mapping[str, Any]], np.ndarray]]]:
    if gt_anns:
        gt_xyxy = _xywh_to_xyxy(np.asarray([ann["bbox"] for ann in gt_anns], dtype=np.float32))
    else:
        gt_xyxy = np.zeros((0, 4), dtype=np.float32)

    anns_by_cat: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for ann in gt_anns:
        anns_by_cat[int(ann["category_id"])].append(ann)

    gt_by_cat = {}
    for category_id, anns in anns_by_cat.items():
        boxes = _xywh_to_xyxy(np.asarray([ann["bbox"] for ann in anns], dtype=np.float32))
        gt_by_cat[category_id] = (anns, boxes)
    return gt_xyxy, gt_by_cat


def export_candidates(args: argparse.Namespace) -> Dict[str, Any]:
    rng = random.Random(args.seed)
    annotation = _load_json(args.annotation)
    image_infos = {int(item["id"]): item for item in annotation["images"]}
    gt_by_image = _group_annotations(annotation)
    categories = _load_categories(annotation, args.phrases_json)
    category_ids = [category_id for category_id, _ in categories]
    category_names = {category_id: name for category_id, name in categories}

    image_ids = sorted(image_infos)
    if args.max_images is not None:
        image_ids = image_ids[: args.max_images]
    split_by_image = _split_images(image_ids, val_ratio=args.val_ratio, seed=args.seed)

    vlm_query_embedding = None
    if args.score_mode == "score_ensemble":
        vlm_query_embedding = _load_vlm_query_embedding(args.vlm_query_embedding)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        "all": (args.output_dir / "all.jsonl").open("w", encoding="utf-8"),
        "train": (args.output_dir / "train.jsonl").open("w", encoding="utf-8"),
        "val": (args.output_dir / "val.jsonl").open("w", encoding="utf-8"),
    }

    stats = Counter()
    selected_stats = Counter()
    missing_outputs = 0
    invalid_category_shape = 0

    try:
        for image_id in tqdm(image_ids, desc="exporting top-k candidate pairs"):
            image_info = image_infos[image_id]
            saved_path = _saved_prediction_path(args.saved_output_dir, str(image_info["file_name"]))
            if not saved_path.exists():
                missing_outputs += 1
                continue

            boxes, logits, roi_features = _load_saved_prediction(saved_path)
            if logits.shape[-1] != len(category_ids):
                invalid_category_shape += 1
                if args.skip_bad_category_shape:
                    continue
                raise ValueError(
                    f"{saved_path} has {logits.shape[-1]} classes, but annotation/phrases define "
                    f"{len(category_ids)} categories."
                )

            width = int(image_info.get("width", 0))
            height = int(image_info.get("height", 0))
            if width <= 0 or height <= 0:
                raise ValueError(f"Image {image_id} has invalid width/height: {width}x{height}.")

            scores = _candidate_scores(
                logits,
                roi_features,
                args=args,
                vlm_query_embedding=vlm_query_embedding,
            )
            pred_xyxy = _prediction_boxes_to_original_xyxy(boxes, width=width, height=height)
            gt_anns = gt_by_image.get(image_id, [])
            gt_xyxy, gt_by_cat = _build_gt_indices(gt_anns)

            num_boxes = min(max(1, int(args.box_topk)), scores.shape[0])
            num_phrases = min(max(1, int(args.phrase_topk)), scores.shape[1])
            box_scores = scores.max(dim=-1).values
            top_box_scores, top_box_indexes = torch.topk(box_scores, num_boxes)

            image_rows: List[Dict[str, Any]] = []
            for query_rank, (query_score, query_index) in enumerate(zip(top_box_scores.tolist(), top_box_indexes.tolist())):
                phrase_scores, phrase_indexes = torch.topk(scores[int(query_index)], num_phrases)
                box_xyxy = pred_xyxy[int(query_index)]
                bbox_xywh = _xyxy_to_xywh(box_xyxy)
                best_any_iou, best_any_category_id, best_any_gt_id = _best_iou_any(
                    box_xyxy,
                    gt_anns,
                    gt_xyxy,
                )

                for phrase_rank, (phrase_score, phrase_index) in enumerate(
                    zip(phrase_scores.tolist(), phrase_indexes.tolist())
                ):
                    target_category_id = int(category_ids[int(phrase_index)])
                    target_iou, matched_gt_id = _best_iou_for_category(
                        box_xyxy,
                        gt_by_cat,
                        target_category_id,
                    )
                    negative_type = _negative_type(
                        target_iou=target_iou,
                        best_any_iou=best_any_iou,
                        best_any_category_id=best_any_category_id,
                        target_category_id=target_category_id,
                        pos_iou_thresh=args.pos_iou_thresh,
                        neg_iou_thresh=args.neg_iou_thresh,
                    )
                    label = int(target_iou >= args.pos_iou_thresh)
                    box_iou_label = int(best_any_iou >= args.pos_iou_thresh)
                    phrase_match_label = label
                    row = {
                        "split": split_by_image[image_id],
                        "image_id": int(image_id),
                        "file_name": str(image_info["file_name"]),
                        "width": width,
                        "height": height,
                        "query_index": int(query_index),
                        "query_rank": int(query_rank),
                        "phrase_rank": int(phrase_rank),
                        "bbox": bbox_xywh,
                        "target_category_id": target_category_id,
                        "phrase": category_names[target_category_id],
                        "detector_score": float(phrase_score),
                        "box_score": float(query_score),
                        "label": label,
                        "box_iou_label": box_iou_label,
                        "phrase_match_label": phrase_match_label,
                        "target_iou": float(target_iou),
                        "matched_gt_id": matched_gt_id,
                        "best_any_iou": float(best_any_iou),
                        "best_any_category_id": best_any_category_id,
                        "best_any_gt_id": best_any_gt_id,
                        "negative_type": negative_type,
                    }
                    image_rows.append(row)
                    stats[(row["split"], negative_type)] += 1
                    stats[(row["split"], "candidate_pairs")] += 1

            selected_rows = _sample_rows_for_image(image_rows, rng=rng, args=args)
            for row in selected_rows:
                _write_row(handles, row)
                selected_stats[(row["split"], str(row["negative_type"]))] += 1
                selected_stats[(row["split"], "selected_pairs")] += 1
    finally:
        for handle in handles.values():
            handle.close()

    summary = {
        "args": _jsonable_args(args),
        "num_images": len(image_ids),
        "missing_outputs": missing_outputs,
        "invalid_category_shape": invalid_category_shape,
        "candidate_stats": {f"{split}:{kind}": int(value) for (split, kind), value in sorted(stats.items())},
        "selected_stats": {
            f"{split}:{kind}": int(value) for (split, kind), value in sorted(selected_stats.items())
        },
    }
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"saved all:   {args.output_dir / 'all.jsonl'}")
    print(f"saved train: {args.output_dir / 'train.jsonl'}")
    print(f"saved val:   {args.output_dir / 'val.jsonl'}")
    print(f"saved summary: {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, default=Path("dataset/d3/annotations/d3_intra_full.json"))
    parser.add_argument("--phrases-json", type=Path, default=Path("dataset/metadata/d3_phrases.json"))
    parser.add_argument(
        "--saved-output-dir",
        type=Path,
        required=True,
        help="Directory of per-image .pth dumps with pred_logits and pred_boxes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/d3/topk_candidate_pairs_w075_top300x50"),
    )
    parser.add_argument("--box-topk", type=int, default=300)
    parser.add_argument("--phrase-topk", type=int, default=50)
    parser.add_argument("--pos-iou-thresh", type=float, default=0.5)
    parser.add_argument("--neg-iou-thresh", type=float, default=0.3)
    parser.add_argument("--score-mode", choices=["sigmoid", "score_ensemble"], default="score_ensemble")
    parser.add_argument(
        "--vlm-query-embedding",
        type=Path,
        default=Path("dataset/metadata/d3_clip_convnextl_sentences.npy"),
    )
    parser.add_argument("--vlm-temperature", type=float, default=100.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-pos-per-image", type=int, default=50, help="Use -1 to keep all.")
    parser.add_argument("--max-wrong-phrase-neg-per-image", type=int, default=150, help="Use -1 to keep all.")
    parser.add_argument("--max-same-phrase-neg-per-image", type=int, default=50, help="Use -1 to keep all.")
    parser.add_argument("--max-background-neg-per-image", type=int, default=50, help="Use -1 to keep all.")
    parser.add_argument("--include-ambiguous", action="store_true")
    parser.add_argument("--max-ambiguous-per-image", type=int, default=50, help="Use -1 to keep all.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--skip-bad-category-shape", action="store_true")
    return parser.parse_args()


def main() -> None:
    export_candidates(parse_args())


if __name__ == "__main__":
    main()
