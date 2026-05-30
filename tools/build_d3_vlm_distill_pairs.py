#!/usr/bin/env python3
"""
Build D3 crop-verifier distillation pairs from cached VLM scores.

The VLM cache is produced by tools/rerank_d3_predictions_with_vlm.py. This
script maps each cached (image, box, phrase) score back to the detector proposal
metadata and emits train/val JSONL rows consumable by train_d3_crop_verifier.py.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rerank_d3_predictions_with_clip_crops import (  # noqa: E402
    _expanded_base_scores_from_category_scores,
    _expanded_category_scores,
    _group_predictions,
    _load_categories,
    _load_image_ids_from_jsonl,
    _load_json,
    _select_class_agnostic_proposals,
)
from rerank_d3_predictions_with_vlm import _bbox_signature, _cache_key, _load_vlm_score_cache  # noqa: E402


def _jsonable_args(args: argparse.Namespace) -> Dict[str, Any]:
    values = {}
    for key, value in vars(args).items():
        values[key] = str(value) if isinstance(value, Path) else value
    return values


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def _split_images(image_ids: Sequence[int], *, val_ratio: float, seed: int) -> Tuple[set[int], set[int]]:
    ids = list(dict.fromkeys(int(image_id) for image_id in image_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    if val_ratio <= 0:
        return set(ids), set()
    val_count = max(1, int(round(len(ids) * val_ratio)))
    val = set(ids[:val_count])
    train = set(ids[val_count:])
    return train, val


def _group_vlm_rows(cache: Mapping[str, Mapping[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    rows_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in cache.values():
        rows_by_image[int(row["image_id"])].append(dict(row))
    return rows_by_image


def _negative_type(*, label: int, target_category_id: int, source_category_id: int) -> Optional[str]:
    if label == 1:
        return None
    if target_category_id == source_category_id:
        return "same_phrase_bad_box"
    return "wrong_phrase_same_region:vlm_soft"


def _rank_weight(rank: int, *, focus_topk: int, power: float, min_weight: float) -> float:
    if focus_topk <= 0:
        return 1.0
    rank = max(1, int(rank))
    weight = (float(focus_topk) / float(rank)) ** float(power)
    return max(float(min_weight), min(1.0, weight))


def build_pairs(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    annotation = _load_json(args.annotation)
    predictions = _load_json(args.predictions)
    if not isinstance(predictions, list):
        raise ValueError(f"Expected prediction JSON list, got {type(predictions).__name__}.")

    image_infos = {int(item["id"]): item for item in annotation["images"]}
    categories_by_id = _load_categories(annotation, args.phrases_json)
    category_ids = sorted(categories_by_id)
    category_to_row = {category_id: idx for idx, category_id in enumerate(category_ids)}
    grouped_predictions = _group_predictions(predictions)
    vlm_cache = _load_vlm_score_cache(args.vlm_score_cache)
    vlm_rows_by_image = _group_vlm_rows(vlm_cache)

    image_ids = sorted(set(grouped_predictions) & set(vlm_rows_by_image))
    if args.image_id_jsonl is not None:
        selected_ids = set(_load_image_ids_from_jsonl(args.image_id_jsonl))
        image_ids = [image_id for image_id in image_ids if image_id in selected_ids]
    if args.max_images is not None:
        image_ids = image_ids[: args.max_images]

    train_ids, val_ids = _split_images(image_ids, val_ratio=args.val_ratio, seed=args.seed)
    rows: List[Dict[str, Any]] = []
    stats: Counter[Tuple[str, str]] = Counter()
    missing_images = 0
    missing_proposals = 0
    invalid_categories = 0
    skipped_parse_failures = 0

    for image_id in tqdm(image_ids, desc="building VLM distill pairs"):
        image_info = image_infos.get(image_id)
        preds = grouped_predictions.get(image_id)
        if image_info is None or not preds:
            missing_images += 1
            continue

        split = "val" if image_id in val_ids else "train"
        image_items: List[Dict[str, Any]] = []
        if args.candidate_mode == "emitted":
            emitted_preds = sorted(preds, key=lambda item: float(item.get("score", 0.0)), reverse=True)
            if args.candidate_topk_per_image > 0:
                emitted_preds = emitted_preds[: args.candidate_topk_per_image]
            for proposal_idx, pred in enumerate(emitted_preds):
                target_category_id = int(pred["category_id"])
                category_idx = category_to_row.get(target_category_id)
                if category_idx is None:
                    invalid_categories += 1
                    continue
                vlm_row = vlm_cache.get(_cache_key(image_id, target_category_id, pred["bbox"]))
                if vlm_row is None:
                    missing_proposals += 1
                    continue
                if args.require_parse_ok and not bool(vlm_row.get("parse_ok", False)):
                    skipped_parse_failures += 1
                    continue
                base_score = float(pred.get("score", 0.0))
                soft_label = float(np.clip(float(vlm_row["vlm_score"]), 0.0, 1.0))
                image_items.append(
                    {
                        "vlm_row": vlm_row,
                        "proposal_idx": int(proposal_idx),
                        "proposal": {
                            "bbox": [float(value) for value in pred["bbox"]],
                            "score": base_score,
                            "source_category_id": target_category_id,
                        },
                        "category_idx": int(category_idx),
                        "target_category_id": int(target_category_id),
                        "base_score": base_score,
                        "soft_label": soft_label,
                    }
                )
        else:
            proposals = _select_class_agnostic_proposals(
                preds,
                topk=args.proposal_topk_per_image,
                nms_thresh=args.proposal_nms_thresh,
            )
            if not proposals:
                missing_proposals += len(vlm_rows_by_image[image_id])
                continue

            proposal_by_bbox = {
                _bbox_signature(proposal["bbox"]): (idx, proposal) for idx, proposal in enumerate(proposals)
            }
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

            for vlm_row in vlm_rows_by_image[image_id]:
                if args.require_parse_ok and not bool(vlm_row.get("parse_ok", False)):
                    skipped_parse_failures += 1
                    continue

                target_category_id = int(vlm_row["category_id"])
                category_idx = category_to_row.get(target_category_id)
                if category_idx is None:
                    invalid_categories += 1
                    continue
                proposal_item = proposal_by_bbox.get(_bbox_signature(vlm_row["bbox"]))
                if proposal_item is None:
                    missing_proposals += 1
                    continue

                proposal_idx, proposal = proposal_item
                base_score = float(base_scores[proposal_idx, category_idx])
                soft_label = float(np.clip(float(vlm_row["vlm_score"]), 0.0, 1.0))
                image_items.append(
                    {
                        "vlm_row": vlm_row,
                        "proposal_idx": int(proposal_idx),
                        "proposal": proposal,
                        "category_idx": int(category_idx),
                        "target_category_id": int(target_category_id),
                        "base_score": base_score,
                        "soft_label": soft_label,
                    }
                )

        image_items.sort(key=lambda item: float(item["base_score"]), reverse=True)
        for candidate_rank, item in enumerate(image_items, start=1):
            proposal = item["proposal"]
            proposal_idx = int(item["proposal_idx"])
            target_category_id = int(item["target_category_id"])
            category_idx = int(item["category_idx"])
            vlm_row = item["vlm_row"]
            base_score = float(item["base_score"])
            soft_label = float(item["soft_label"])
            label = int(soft_label >= args.positive_threshold)
            source_category_id = int(proposal.get("source_category_id", -1))
            negative_type = _negative_type(
                label=label,
                target_category_id=target_category_id,
                source_category_id=source_category_id,
            )
            sample_weight = _rank_weight(
                candidate_rank,
                focus_topk=args.rank_focus_topk,
                power=args.rank_weight_power,
                min_weight=args.min_rank_weight,
            )
            if args.soft_label_weight > 0:
                sample_weight *= 1.0 + args.soft_label_weight * soft_label
            row = {
                "split": split,
                "image_id": int(image_id),
                "file_name": str(image_info["file_name"]),
                "width": int(image_info.get("width", 0)),
                "height": int(image_info.get("height", 0)),
                "bbox": [float(value) for value in proposal["bbox"]],
                "detector_category_id": source_category_id,
                "target_category_id": target_category_id,
                "phrase": str(categories_by_id[target_category_id]),
                "detector_score": base_score,
                "proposal_score": float(proposal["score"]),
                "candidate_score": base_score,
                "proposal_idx": proposal_idx,
                "category_idx": category_idx,
                "candidate_rank": int(candidate_rank),
                "sample_weight": float(sample_weight),
                "soft_label": soft_label,
                "label": label,
                "vlm_score": soft_label,
                "parse_ok": bool(vlm_row.get("parse_ok", False)),
                "negative_type": negative_type,
                "positive_source": "vlm_soft" if label == 1 else None,
            }
            rows.append(row)
            stats[(split, "positive" if label == 1 else str(negative_type))] += 1

    summary = {
        "num_images": len(image_ids),
        "num_train_images": len(train_ids),
        "num_val_images": len(val_ids),
        "num_rows": len(rows),
        "num_train": sum(1 for row in rows if row["split"] == "train"),
        "num_val": sum(1 for row in rows if row["split"] == "val"),
        "num_positive": sum(1 for row in rows if row["label"] == 1),
        "num_negative": sum(1 for row in rows if row["label"] == 0),
        "missing_images": missing_images,
        "missing_proposals": missing_proposals,
        "invalid_categories": invalid_categories,
        "skipped_parse_failures": skipped_parse_failures,
        "stats": {f"{split}:{kind}": int(value) for (split, kind), value in sorted(stats.items())},
        "args": _jsonable_args(args),
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Detector COCO-format result JSON.")
    parser.add_argument("--vlm-score-cache", type=Path, required=True, help="VLM JSONL score cache.")
    parser.add_argument("--annotation", type=Path, required=True, help="D3 COCO annotation JSON.")
    parser.add_argument("--phrases-json", type=Path, default=Path("dataset/metadata/d3_phrases.json"))
    parser.add_argument("--image-id-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate-mode",
        choices=("expanded", "emitted"),
        default="expanded",
        help=(
            "Candidate interpretation for cached VLM scores. expanded matches the class-agnostic "
            "proposal x phrase cache; emitted matches detector-emitted category predictions directly."
        ),
    )
    parser.add_argument(
        "--candidate-topk-per-image",
        type=int,
        default=100,
        help="Detector-emitted predictions per image to use with --candidate-mode emitted.",
    )
    parser.add_argument("--proposal-topk-per-image", type=int, default=100)
    parser.add_argument("--proposal-nms-thresh", type=float, default=0.9)
    parser.add_argument("--expanded-base-score", choices=("category_score", "objectness"), default="category_score")
    parser.add_argument("--category-score-match-iou", type=float, default=0.9)
    parser.add_argument("--missing-category-score-scale", type=float, default=0.3)
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    parser.add_argument("--rank-focus-topk", type=int, default=20)
    parser.add_argument("--rank-weight-power", type=float, default=0.5)
    parser.add_argument("--min-rank-weight", type=float, default=0.25)
    parser.add_argument("--soft-label-weight", type=float, default=0.0)
    parser.add_argument("--require-parse-ok", action="store_true")
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
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"saved all:   {output_dir / 'all.jsonl'} ({all_count})")
    print(f"saved train: {output_dir / 'train.jsonl'} ({train_count})")
    print(f"saved val:   {output_dir / 'val.jsonl'} ({val_count})")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
