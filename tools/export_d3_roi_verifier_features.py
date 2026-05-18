#!/usr/bin/env python3
"""
Export detector ROI-feature caches for D3 region-description verifier training.

This is the detector-internal counterpart of tools/train_d3_crop_verifier.py's
OpenCLIP crop cache. It reads DINO per-image .pth dumps containing:

  pred_boxes:        [1, num_queries, 4] normalized cxcywh boxes
  roi_features_ori:  [1, num_queries, 768] detector ROI features

and converts D3 verifier pair JSONL rows into the same feature-cache schema used
by train_d3_crop_verifier.py:

  crop_feats, text_feats, labels, detector_scores, negative_types,
  target_category_ids, image_ids

The "crop_feats" key is kept for trainer compatibility; it stores detector ROI
features, not OpenCLIP crop-image features.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _jsonable_args(args: argparse.Namespace) -> Dict[str, Any]:
    values = {}
    for key, value in vars(args).items():
        values[key] = str(value) if isinstance(value, Path) else value
    return values


def _negative_kind(row: Mapping[str, Any]) -> str:
    value = row.get("negative_type")
    return str(value) if value is not None else "positive"


def _sample_count(base_count: int, ratio: float, available: int) -> int:
    if ratio < 0:
        return available
    return min(available, int(round(base_count * ratio)))


def _sample_total_for_positive_count(
    positive_count: int,
    *,
    same_phrase_neg_per_pos: float,
    wrong_phrase_neg_per_pos: float,
    same_available: int,
    wrong_available: int,
) -> int:
    same_count = _sample_count(positive_count, same_phrase_neg_per_pos, same_available)
    wrong_count = _sample_count(positive_count, wrong_phrase_neg_per_pos, wrong_available)
    return positive_count + same_count + wrong_count


def _fit_positive_count_to_budget(
    positive_limit: int,
    *,
    max_samples: Optional[int],
    same_phrase_neg_per_pos: float,
    wrong_phrase_neg_per_pos: float,
    same_available: int,
    wrong_available: int,
) -> int:
    if max_samples is None or max_samples <= 0:
        return positive_limit
    if positive_limit <= 0:
        return 0

    lo = 0
    hi = min(positive_limit, max_samples)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        total = _sample_total_for_positive_count(
            mid,
            same_phrase_neg_per_pos=same_phrase_neg_per_pos,
            wrong_phrase_neg_per_pos=wrong_phrase_neg_per_pos,
            same_available=same_available,
            wrong_available=wrong_available,
        )
        if total <= max_samples:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return best if best > 0 else min(positive_limit, max_samples)


def _sample_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    same_phrase_neg_per_pos: float,
    wrong_phrase_neg_per_pos: float,
    max_positives: Optional[int],
    max_samples: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    positives = [dict(row) for row in rows if int(row.get("label", 0)) == 1]
    same_phrase = [
        dict(row)
        for row in rows
        if int(row.get("label", 0)) == 0 and _negative_kind(row) == "same_phrase_bad_box"
    ]
    wrong_phrase = [
        dict(row)
        for row in rows
        if int(row.get("label", 0)) == 0 and _negative_kind(row).startswith("wrong_phrase_same_region")
    ]

    rng.shuffle(positives)
    rng.shuffle(same_phrase)
    rng.shuffle(wrong_phrase)

    positive_limit = len(positives)
    if max_positives is not None and max_positives > 0:
        positive_limit = min(positive_limit, max_positives)
    positive_limit = _fit_positive_count_to_budget(
        positive_limit,
        max_samples=max_samples,
        same_phrase_neg_per_pos=same_phrase_neg_per_pos,
        wrong_phrase_neg_per_pos=wrong_phrase_neg_per_pos,
        same_available=len(same_phrase),
        wrong_available=len(wrong_phrase),
    )

    positives = positives[:positive_limit]
    same_count = _sample_count(len(positives), same_phrase_neg_per_pos, len(same_phrase))
    wrong_count = _sample_count(len(positives), wrong_phrase_neg_per_pos, len(wrong_phrase))
    selected = positives + same_phrase[:same_count] + wrong_phrase[:wrong_count]

    if max_samples is not None and max_samples > 0 and len(selected) > max_samples:
        pos = [row for row in selected if int(row.get("label", 0)) == 1]
        neg = [row for row in selected if int(row.get("label", 0)) == 0]
        if len(pos) > max_samples:
            selected = rng.sample(pos, max_samples)
        else:
            selected = pos + rng.sample(neg, max_samples - len(pos))

    rng.shuffle(selected)
    return selected


def _summarize_rows(name: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts = Counter(_negative_kind(row) for row in rows)
    summary = {
        "total": len(rows),
        "positive": sum(1 for row in rows if int(row.get("label", 0)) == 1),
        "negative": sum(1 for row in rows if int(row.get("label", 0)) == 0),
    }
    summary.update({kind: int(count) for kind, count in sorted(counts.items())})
    print(f"{name} rows: {json.dumps(summary, sort_keys=True)}")
    return summary


def _load_text_bank(path: Path, *, text_index: Optional[int]) -> torch.Tensor:
    bank = np.load(path)
    if bank.ndim == 3:
        if text_index is None:
            if bank.shape[1] != 1:
                raise ValueError(
                    f"{path} has shape {bank.shape}; pass --text-index for a multi-prompt bank."
                )
            bank = bank[:, 0, :]
        else:
            bank = bank[:, text_index, :]
    if bank.ndim != 2:
        raise ValueError(f"Expected 2D text bank after selection, got shape {bank.shape}.")
    tensor = torch.from_numpy(bank).float()
    return F.normalize(tensor, p=2, dim=-1).cpu()


def _saved_prediction_path(saved_output_dir: Path, file_name: str) -> Path:
    path = Path(file_name)
    return saved_output_dir / path.with_suffix(".pth").name


def _load_saved_prediction(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")

    if "pred_boxes" not in data or "roi_features_ori" not in data:
        raise KeyError(f"{path} must contain pred_boxes and roi_features_ori.")

    boxes = data["pred_boxes"].float()
    roi_features = data["roi_features_ori"].float()
    if boxes.ndim == 3:
        boxes = boxes[0]
    if roi_features.ndim == 3:
        roi_features = roi_features[0]
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"{path} pred_boxes has unsupported shape {tuple(boxes.shape)}.")
    if roi_features.ndim != 2:
        raise ValueError(f"{path} roi_features_ori has unsupported shape {tuple(roi_features.shape)}.")
    if boxes.shape[0] != roi_features.shape[0]:
        raise ValueError(
            f"{path} query count mismatch: boxes={tuple(boxes.shape)}, roi={tuple(roi_features.shape)}."
        )
    return boxes.cpu(), F.normalize(roi_features.cpu(), p=2, dim=-1)


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


def _xywh_to_xyxy(box: Sequence[float]) -> np.ndarray:
    x, y, w, h = [float(v) for v in box]
    return np.asarray([x, y, x + w, y + h], dtype=np.float32)


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


def _best_query_match(row: Mapping[str, Any], pred_xyxy: np.ndarray) -> Tuple[int, float]:
    row_xyxy = _xywh_to_xyxy(row["bbox"])
    ious = _pairwise_iou_xyxy(row_xyxy, pred_xyxy)
    if len(ious) == 0:
        return -1, 0.0
    query_index = int(ious.argmax())
    return query_index, float(ious[query_index])


def _encode_rows_to_roi_cache(
    rows: Sequence[Mapping[str, Any]],
    *,
    saved_output_dir: Path,
    text_bank: torch.Tensor,
    min_match_iou: float,
    split: str,
) -> Dict[str, Any]:
    rows_by_file: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_file[str(row["file_name"])].append(row)

    roi_feats: List[torch.Tensor] = []
    text_feats: List[torch.Tensor] = []
    labels: List[float] = []
    detector_scores: List[float] = []
    negative_types: List[str] = []
    target_category_ids: List[int] = []
    image_ids: List[int] = []
    query_indices: List[int] = []
    matched_ious: List[float] = []

    missing_outputs = 0
    low_iou_matches = 0
    invalid_categories = 0
    match_hist = Counter()

    for file_name, file_rows in tqdm(rows_by_file.items(), desc=f"exporting {split} ROI features"):
        saved_path = _saved_prediction_path(saved_output_dir, file_name)
        if not saved_path.exists():
            missing_outputs += len(file_rows)
            continue

        pred_boxes, pred_roi_features = _load_saved_prediction(saved_path)
        first_row = file_rows[0]
        width = int(first_row.get("width", 0))
        height = int(first_row.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError(f"{file_name} has invalid width/height in verifier pairs.")
        pred_xyxy = _prediction_boxes_to_original_xyxy(pred_boxes, width=width, height=height)

        for row in file_rows:
            target_category_id = int(row.get("target_category_id", -1))
            text_index = target_category_id - 1
            if text_index < 0 or text_index >= text_bank.shape[0]:
                invalid_categories += 1
                continue

            query_index, matched_iou = _best_query_match(row, pred_xyxy)
            if query_index < 0 or matched_iou < min_match_iou:
                low_iou_matches += 1
                continue

            roi_feats.append(pred_roi_features[query_index])
            text_feats.append(text_bank[text_index])
            labels.append(float(row["label"]))
            detector_scores.append(float(row.get("detector_score", 0.0)))
            negative_types.append(_negative_kind(row))
            target_category_ids.append(target_category_id)
            image_ids.append(int(row.get("image_id", -1)))
            query_indices.append(query_index)
            matched_ious.append(matched_iou)
            bucket = min(9, int(matched_iou * 10))
            match_hist[f"{bucket / 10:.1f}-{(bucket + 1) / 10:.1f}"] += 1

    if not roi_feats:
        raise RuntimeError(
            "No ROI features were exported. Check --saved-output-dir and --min-match-iou."
        )

    matched_iou_array = np.asarray(matched_ious, dtype=np.float32)
    cache = {
        "crop_feats": torch.stack(roi_feats, dim=0).float(),
        "text_feats": torch.stack(text_feats, dim=0).float(),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "detector_scores": torch.tensor(detector_scores, dtype=torch.float32),
        "negative_types": negative_types,
        "target_category_ids": torch.tensor(target_category_ids, dtype=torch.long),
        "image_ids": torch.tensor(image_ids, dtype=torch.long),
        "query_indices": torch.tensor(query_indices, dtype=torch.long),
        "matched_ious": torch.tensor(matched_ious, dtype=torch.float32),
        "meta": {
            "feature_source": "detector_roi_features",
            "num_input_rows": len(rows),
            "num_encoded_rows": len(labels),
            "missing_outputs": missing_outputs,
            "low_iou_matches": low_iou_matches,
            "invalid_categories": invalid_categories,
            "min_match_iou": min_match_iou,
            "matched_iou_mean": float(matched_iou_array.mean()),
            "matched_iou_min": float(matched_iou_array.min()),
            "matched_iou_p01": float(np.quantile(matched_iou_array, 0.01)),
            "matched_iou_p50": float(np.quantile(matched_iou_array, 0.50)),
            "matched_iou_p99": float(np.quantile(matched_iou_array, 0.99)),
            "matched_iou_hist": dict(sorted(match_hist.items())),
        },
    }
    print(json.dumps(cache["meta"], indent=2, sort_keys=True))
    return cache


def _save_cache(path: Path, cache: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(cache), path)
    print(f"saved cache: {path}")


def _load_and_sample_rows(
    path: Path,
    *,
    split: str,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rows = _read_jsonl(path)
    rows = _sample_rows(
        rows,
        same_phrase_neg_per_pos=args.same_phrase_neg_per_pos,
        wrong_phrase_neg_per_pos=args.wrong_phrase_neg_per_pos,
        max_positives=args.max_positives,
        max_samples=args.max_train_samples if split == "train" else args.max_val_samples,
        seed=args.seed if split == "train" else args.seed + 1,
    )
    summary = _summarize_rows(f"selected {split}", rows)
    return rows, summary


def export_split(
    *,
    split: str,
    jsonl_path: Path,
    output_path: Path,
    text_bank: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    rows, row_summary = _load_and_sample_rows(jsonl_path, split=split, args=args)
    cache = _encode_rows_to_roi_cache(
        rows,
        saved_output_dir=args.saved_output_dir,
        text_bank=text_bank,
        min_match_iou=args.min_match_iou,
        split=split,
    )
    cache["meta"].update(
        {
            "split": split,
            "selected_row_summary": row_summary,
            "saved_output_dir": str(args.saved_output_dir),
            "text_embedding": str(args.text_embedding),
            "text_index": args.text_index,
        }
    )
    _save_cache(output_path, cache)
    return cache["meta"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=Path("dataset/d3/verifier_pairs_w075/train.jsonl"),
        help="Training verifier pair JSONL.",
    )
    parser.add_argument(
        "--val-jsonl",
        type=Path,
        default=Path("dataset/d3/verifier_pairs_w075/val.jsonl"),
        help="Validation verifier pair JSONL.",
    )
    parser.add_argument(
        "--saved-output-dir",
        type=Path,
        required=True,
        help="Directory of DINO per-image .pth dumps with roi_features_ori.",
    )
    parser.add_argument(
        "--text-embedding",
        type=Path,
        default=Path("dataset/metadata/d3_description_anchor_target_w100_bank_convnextl.npy"),
        help="Target phrase text embedding bank. Category id c maps to row c-1.",
    )
    parser.add_argument(
        "--text-index",
        type=int,
        default=None,
        help="Prompt index to select if --text-embedding is a [C,K,D] bank.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/d3_roi_verifier_features_w075/cache"),
        help="Directory for train_features.pt, val_features.pt, and summary.json.",
    )
    parser.add_argument("--same-phrase-neg-per-pos", type=float, default=1.0)
    parser.add_argument("--wrong-phrase-neg-per-pos", type=float, default=2.0)
    parser.add_argument("--max-positives", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--min-match-iou", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    text_bank = _load_text_bank(args.text_embedding, text_index=args.text_index)
    summary = {
        "args": _jsonable_args(args),
        "text_bank_shape": list(text_bank.shape),
    }
    summary["train"] = export_split(
        split="train",
        jsonl_path=args.train_jsonl,
        output_path=args.output_dir / "train_features.pt",
        text_bank=text_bank,
        args=args,
    )
    summary["val"] = export_split(
        split="val",
        jsonl_path=args.val_jsonl,
        output_path=args.output_dir / "val_features.pt",
        text_bank=text_bank,
        args=args,
    )

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
