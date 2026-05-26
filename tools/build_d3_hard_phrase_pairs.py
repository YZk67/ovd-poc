#!/usr/bin/env python3
"""
Build D3 verifier pairs with score-mined hard wrong phrases.

This script differs from tools/build_d3_verifier_pairs.py in one important way:
wrong_phrase_same_region negatives are not sampled randomly. For each high-IoU
proposal, it reads the expanded score cache produced by
tools/rerank_d3_predictions_with_clip_crops.py and chooses the top-scoring
wrong phrases for that same box.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from build_d3_verifier_pairs import (
    _base_row,
    _best_iou_any,
    _best_iou_for_category,
    _group_annotations,
    _group_predictions,
    _load_categories,
    _load_json,
    _split_images,
    _write_jsonl,
)
from rerank_d3_predictions_with_clip_crops import (
    _expanded_base_scores_from_category_scores,
    _expanded_cache_to_arrays,
    _expanded_score_cache_path,
    _fuse_verifier_logit_matrix,
)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_cache(cache_dir: Path, image_id: int) -> Optional[Dict[str, Any]]:
    path = _expanded_score_cache_path(cache_dir, image_id)
    if not path.exists():
        return None
    cache = _torch_load(path)
    if not isinstance(cache, Mapping):
        return None
    return dict(cache)


def _cache_category_ids(cache: Mapping[str, Any], categories: Mapping[int, str], num_columns: int) -> List[int]:
    meta = cache.get("meta", {})
    feature_meta = meta.get("feature", {}) if isinstance(meta, Mapping) else {}
    category_ids = feature_meta.get("category_ids") if isinstance(feature_meta, Mapping) else None
    if isinstance(category_ids, Sequence) and not isinstance(category_ids, (str, bytes)):
        parsed = [int(category_id) for category_id in category_ids]
        if len(parsed) == num_columns:
            return parsed

    parsed = sorted(int(category_id) for category_id in categories)
    if len(parsed) != num_columns:
        raise ValueError(
            f"Cache has {num_columns} category columns, but phrase/category metadata has {len(parsed)} entries."
        )
    return parsed


def _score_matrix(
    *,
    proposals: Sequence[Mapping[str, Any]],
    category_scores: np.ndarray,
    pair_signal: Optional[np.ndarray],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    base_scores = _expanded_base_scores_from_category_scores(
        proposals,
        category_scores,
        num_categories=category_scores.shape[1],
        mode=args.expanded_base_score,
        missing_category_score_scale=args.missing_category_score_scale,
    )
    if args.hard_score_source == "category_score":
        return base_scores, base_scores
    if pair_signal is None:
        raise ValueError(
            "Cache has no pair_signal; either rebuild it with a verifier checkpoint or use "
            "--hard-score-source category_score."
        )
    fused_scores = _fuse_verifier_logit_matrix(
        base_scores,
        pair_signal,
        mode=args.verifier_fusion,
        fusion_weight=args.verifier_fusion_weight,
    )
    return base_scores, fused_scores


def _detector_score_for_pair(
    *,
    base_score: float,
    hard_score: float,
    args: argparse.Namespace,
) -> float:
    if args.detector_score_source == "category_score":
        return float(base_score)
    if args.detector_score_source == "hard_score":
        return float(hard_score)
    raise ValueError(f"Unsupported detector_score_source={args.detector_score_source!r}.")


def _jsonable_args(args: argparse.Namespace) -> Dict[str, Any]:
    values = {}
    for key, value in vars(args).items():
        values[key] = str(value) if isinstance(value, Path) else value
    return values


def _candidate_wrong_categories(
    *,
    scores: np.ndarray,
    proposal_idx: int,
    category_ids: Sequence[int],
    gt_by_cat: Mapping[int, List[Mapping[str, Any]]],
    bbox: Sequence[float],
    positive_category_id: int,
    topk: int,
    neg_iou_thresh: float,
    min_score: Optional[float],
) -> List[Tuple[int, float, Optional[int], float, int]]:
    selected: List[Tuple[int, float, Optional[int], float, int]] = []
    row_scores = scores[proposal_idx]
    for rank, category_col in enumerate(np.argsort(-row_scores, kind="mergesort")):
        category_id = int(category_ids[int(category_col)])
        if category_id == positive_category_id:
            continue
        score = float(row_scores[int(category_col)])
        if min_score is not None and score < min_score:
            break
        target_iou, matched_gt_id = _best_iou_for_category(bbox, gt_by_cat, category_id)
        if target_iou <= neg_iou_thresh:
            selected.append((category_id, target_iou, matched_gt_id, score, int(rank)))
            if len(selected) >= topk:
                break
    return selected


def build_pairs(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    annotation = _load_json(args.annotation)
    predictions = _load_json(args.predictions)
    if not isinstance(predictions, list):
        raise ValueError(f"Expected prediction JSON list, got {type(predictions).__name__}.")

    image_infos = {int(item["id"]): item for item in annotation["images"]}
    gt_by_image = _group_annotations(annotation)
    pred_by_image = _group_predictions(predictions)
    categories = _load_categories(annotation, args.phrases_json)

    image_ids = sorted(set(pred_by_image) & set(image_infos))
    if args.max_images is not None:
        image_ids = image_ids[: args.max_images]
    split_by_image = _split_images(image_ids, val_ratio=args.val_ratio, seed=args.seed)

    rows: List[Dict[str, Any]] = []
    stats = Counter()
    missing_caches = 0
    empty_caches = 0

    for image_id in tqdm(image_ids, desc="building hard phrase pairs"):
        cache = _load_cache(args.score_cache_dir, image_id)
        if cache is None:
            missing_caches += 1
            continue

        proposals, category_scores, pair_signal, _ = _expanded_cache_to_arrays(cache)
        if args.proposal_topk_per_image > 0:
            proposals = proposals[: args.proposal_topk_per_image]
            category_scores = category_scores[: args.proposal_topk_per_image]
            if pair_signal is not None:
                pair_signal = pair_signal[: args.proposal_topk_per_image]
        if not proposals:
            empty_caches += 1
            continue

        category_ids = _cache_category_ids(cache, categories, category_scores.shape[1])
        category_to_col = {int(category_id): idx for idx, category_id in enumerate(category_ids)}
        base_scores, hard_scores = _score_matrix(
            proposals=proposals,
            category_scores=category_scores,
            pair_signal=pair_signal,
            args=args,
        )

        split = split_by_image[image_id]
        image_info = image_infos[image_id]
        gt_anns = gt_by_image.get(image_id, [])
        gt_by_cat: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
        for ann in gt_anns:
            gt_by_cat[int(ann["category_id"])].append(ann)

        positives: List[Dict[str, Any]] = []
        hard_negs: List[Dict[str, Any]] = []
        same_phrase_negs: List[Dict[str, Any]] = []

        for proposal_idx, proposal in enumerate(proposals):
            bbox = proposal["bbox"]
            best_any_iou, best_any_cat, best_any_gt_id = _best_iou_any(bbox, gt_anns)
            source_cat = int(proposal.get("source_category_id", -1))
            source_score = float(proposal.get("score", 0.0))

            if (
                best_any_cat is not None
                and best_any_iou >= args.pos_iou_thresh
                and int(best_any_cat) in category_to_col
                and int(best_any_cat) in categories
            ):
                positive_category_id = int(best_any_cat)
                positive_col = category_to_col[positive_category_id]
                positive_base = float(base_scores[proposal_idx, positive_col])
                positive_hard = float(hard_scores[proposal_idx, positive_col])
                positive_score = _detector_score_for_pair(
                    base_score=positive_base,
                    hard_score=positive_hard,
                    args=args,
                )
                positive = _base_row(
                    split=split,
                    image_info=image_info,
                    pred={"bbox": bbox, "category_id": source_cat, "score": positive_score},
                    target_category_id=positive_category_id,
                    phrase=categories[positive_category_id],
                    label=1,
                    target_iou=best_any_iou,
                    matched_gt_id=best_any_gt_id,
                    best_any_iou=best_any_iou,
                    best_any_category_id=best_any_cat,
                    best_any_gt_id=best_any_gt_id,
                    negative_type=None,
                    positive_source="proposal_iou_hard_phrase",
                )
                positive.update(
                    {
                        "proposal_idx": int(proposal_idx),
                        "base_score": positive_base,
                        "hard_score": positive_hard,
                        "source_proposal_score": source_score,
                        "hard_phrase_rank": 0,
                    }
                )
                positives.append(positive)

                wrong_categories = _candidate_wrong_categories(
                    scores=hard_scores,
                    proposal_idx=proposal_idx,
                    category_ids=category_ids,
                    gt_by_cat=gt_by_cat,
                    bbox=bbox,
                    positive_category_id=positive_category_id,
                    topk=args.topk_wrong_phrases,
                    neg_iou_thresh=args.neg_iou_thresh,
                    min_score=args.min_hard_negative_score,
                )
                for wrong_cat, target_iou, matched_gt_id, hard_score, rank in wrong_categories:
                    wrong_col = category_to_col[wrong_cat]
                    base_score = float(base_scores[proposal_idx, wrong_col])
                    detector_score = _detector_score_for_pair(
                        base_score=base_score,
                        hard_score=hard_score,
                        args=args,
                    )
                    negative = _base_row(
                        split=split,
                        image_info=image_info,
                        pred={"bbox": bbox, "category_id": source_cat, "score": detector_score},
                        target_category_id=wrong_cat,
                        phrase=categories[wrong_cat],
                        label=0,
                        target_iou=target_iou,
                        matched_gt_id=matched_gt_id,
                        best_any_iou=best_any_iou,
                        best_any_category_id=best_any_cat,
                        best_any_gt_id=best_any_gt_id,
                        negative_type="wrong_phrase_same_region:hard_top_score",
                    )
                    negative.update(
                        {
                            "proposal_idx": int(proposal_idx),
                            "base_score": base_score,
                            "hard_score": float(hard_score),
                            "source_proposal_score": source_score,
                            "hard_phrase_rank": int(rank),
                        }
                    )
                    hard_negs.append(negative)

            if args.max_same_phrase_neg_per_image != 0 and source_cat in categories:
                source_iou, source_gt_id = _best_iou_for_category(bbox, gt_by_cat, source_cat)
                if source_iou <= args.neg_iou_thresh:
                    source_col = category_to_col.get(source_cat)
                    if source_col is not None:
                        base_score = float(base_scores[proposal_idx, source_col])
                        hard_score = float(hard_scores[proposal_idx, source_col])
                    else:
                        base_score = source_score
                        hard_score = source_score
                    detector_score = _detector_score_for_pair(
                        base_score=base_score,
                        hard_score=hard_score,
                        args=args,
                    )
                    row = _base_row(
                        split=split,
                        image_info=image_info,
                        pred={"bbox": bbox, "category_id": source_cat, "score": detector_score},
                        target_category_id=source_cat,
                        phrase=categories[source_cat],
                        label=0,
                        target_iou=source_iou,
                        matched_gt_id=source_gt_id,
                        best_any_iou=best_any_iou,
                        best_any_category_id=best_any_cat,
                        best_any_gt_id=best_any_gt_id,
                        negative_type="same_phrase_bad_box",
                    )
                    row.update(
                        {
                            "proposal_idx": int(proposal_idx),
                            "base_score": base_score,
                            "hard_score": hard_score,
                            "source_proposal_score": source_score,
                            "hard_phrase_rank": None,
                        }
                    )
                    same_phrase_negs.append(row)

        if args.max_pos_per_image > 0:
            positives = positives[: args.max_pos_per_image]
        if args.max_hard_neg_per_image > 0:
            hard_negs = hard_negs[: args.max_hard_neg_per_image]
        if args.max_same_phrase_neg_per_image > 0:
            same_phrase_negs = same_phrase_negs[: args.max_same_phrase_neg_per_image]

        image_rows = positives + hard_negs + same_phrase_negs
        rows.extend(image_rows)
        for row in image_rows:
            kind = "positive" if int(row["label"]) == 1 else str(row["negative_type"])
            stats[(split, kind)] += 1

    summary = {
        "num_images": len(image_ids),
        "num_rows": len(rows),
        "num_train": sum(1 for row in rows if row["split"] == "train"),
        "num_val": sum(1 for row in rows if row["split"] == "val"),
        "num_positive": sum(1 for row in rows if row["label"] == 1),
        "num_negative": sum(1 for row in rows if row["label"] == 0),
        "missing_caches": int(missing_caches),
        "empty_caches": int(empty_caches),
        "stats": {f"{split}:{kind}": int(value) for (split, kind), value in sorted(stats.items())},
        "args": _jsonable_args(args),
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Detector COCO-format result JSON.")
    parser.add_argument(
        "--score-cache-dir",
        type=Path,
        required=True,
        help="Expanded score cache directory from rerank_d3_predictions_with_clip_crops.py.",
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
        required=True,
        help="Directory for train.jsonl, val.jsonl, all.jsonl, and summary.json.",
    )
    parser.add_argument("--proposal-topk-per-image", type=int, default=100)
    parser.add_argument("--topk-wrong-phrases", type=int, default=5)
    parser.add_argument("--pos-iou-thresh", type=float, default=0.5)
    parser.add_argument("--neg-iou-thresh", type=float, default=0.3)
    parser.add_argument("--expanded-base-score", choices=("objectness", "category_score"), default="category_score")
    parser.add_argument("--missing-category-score-scale", type=float, default=0.3)
    parser.add_argument("--hard-score-source", choices=("category_score", "cache_signal"), default="cache_signal")
    parser.add_argument("--verifier-fusion", choices=("logit_add", "linear", "replace"), default="logit_add")
    parser.add_argument("--verifier-fusion-weight", type=float, default=0.45)
    parser.add_argument(
        "--detector-score-source",
        choices=("category_score", "hard_score"),
        default="hard_score",
        help="Score stored in detector_score for the token verifier feature.",
    )
    parser.add_argument("--min-hard-negative-score", type=float, default=None)
    parser.add_argument("--max-pos-per-image", type=int, default=100, help="Use 0 to keep all positives.")
    parser.add_argument("--max-hard-neg-per-image", type=int, default=500, help="Use 0 to keep all hard negatives.")
    parser.add_argument(
        "--max-same-phrase-neg-per-image",
        type=int,
        default=0,
        help="Use 0 to disable same_phrase_bad_box rows; use -1 to keep all.",
    )
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
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"saved all:   {output_dir / 'all.jsonl'} ({all_count})")
    print(f"saved train: {output_dir / 'train.jsonl'} ({train_count})")
    print(f"saved val:   {output_dir / 'val.jsonl'} ({val_count})")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
