#!/usr/bin/env python3
"""Create deterministic D3 group/scenario JSONL splits.

D3 images are organized by scenario groups in ``groups.pkl``. Random image-level
splits leak almost every description into validation. This tool splits whole
groups, then writes image-id JSONL files compatible with
``tools/filter_coco_annotations_by_image_ids.py``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, ensure_ascii=False)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _split_counts(total: int, *, val_ratio: float, test_ratio: float) -> tuple[int, int]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("--val-ratio and --test-ratio must be non-negative and sum to < 1.")
    val_count = int(round(total * val_ratio)) if val_ratio > 0 else 0
    test_count = int(round(total * test_ratio)) if test_ratio > 0 else 0
    if val_ratio > 0:
        val_count = max(1, val_count)
    if test_ratio > 0:
        test_count = max(1, test_count)
    while val_count + test_count >= total and total > 1:
        if test_count >= val_count and test_count > 0:
            test_count -= 1
        elif val_count > 0:
            val_count -= 1
        else:
            break
    return val_count, test_count


def _write_shards(output_dir: Path, split_name: str, rows: Sequence[Mapping[str, Any]], shard_size: int) -> int:
    if shard_size <= 0:
        return 0
    shard_dir = output_dir / f"{split_name}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for start in range(0, len(rows), shard_size):
        path = shard_dir / f"shard_{count:03d}.jsonl"
        _write_jsonl(path, rows[start : start + shard_size])
        count += 1
    return count


def _group_rows(groups: Mapping[int, Mapping[str, Any]], group_ids: Sequence[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group_id in group_ids:
        group = groups[int(group_id)]
        rows.append(
            {
                "group_id": int(group_id),
                "group_name": str(group.get("group_name", "")),
                "scene": str(group.get("scene", "")),
                "num_images": len(group.get("img_id", [])),
                "num_inner_sentences": len(group.get("inner_sent_id", [])),
                "num_positive_sentences": len(group.get("pos_sent_id", [])),
            }
        )
    return rows


def _category_rows(
    category_ids: Sequence[int],
    category_name_by_id: Mapping[int, str],
) -> List[Dict[str, Any]]:
    return [
        {
            "category_id": int(category_id),
            "name": str(category_name_by_id.get(int(category_id), "")),
        }
        for category_id in sorted(category_ids)
    ]


def _image_rows(
    groups: Mapping[int, Mapping[str, Any]],
    image_info_by_id: Mapping[int, Mapping[str, Any]],
    group_ids: Sequence[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: Set[int] = set()
    for group_id in group_ids:
        group = groups[int(group_id)]
        for image_id in group.get("img_id", []):
            image_id = int(image_id)
            if image_id in seen or image_id not in image_info_by_id:
                continue
            seen.add(image_id)
            info = image_info_by_id[image_id]
            rows.append(
                {
                    "image_id": image_id,
                    "file_name": str(info.get("file_name", "")),
                    "width": int(info.get("width", 0)),
                    "height": int(info.get("height", 0)),
                    "group_id": int(group_id),
                    "group_name": str(group.get("group_name", "")),
                    "scene": str(group.get("scene", "")),
                }
            )
    rows.sort(key=lambda item: int(item["image_id"]))
    return rows


def _annotation_category_ids(annotation: Mapping[str, Any], image_rows: Sequence[Mapping[str, Any]]) -> Set[int]:
    image_ids = {int(row["image_id"]) for row in image_rows}
    return {
        int(ann["category_id"])
        for ann in annotation.get("annotations", [])
        if int(ann["image_id"]) in image_ids
    }


def _annotation_count(annotation: Mapping[str, Any], image_rows: Sequence[Mapping[str, Any]]) -> int:
    image_ids = {int(row["image_id"]) for row in image_rows}
    return sum(1 for ann in annotation.get("annotations", []) if int(ann["image_id"]) in image_ids)


def _prompt_texts_for_groups(
    groups: Mapping[int, Mapping[str, Any]],
    sentences: Mapping[int, Mapping[str, Any]],
    group_ids: Sequence[int],
) -> Set[str]:
    texts: Set[str] = set()
    for group_id in group_ids:
        for sent_id in groups[int(group_id)].get("pos_sent_id", []):
            sent = sentences.get(int(sent_id))
            if sent is not None:
                texts.add(str(sent["raw_sent"]))
    return texts


def _category_ids_with_texts(
    category_ids: Set[int],
    category_name_by_id: Mapping[int, str],
    texts: Set[str],
) -> Set[int]:
    return {category_id for category_id in category_ids if category_name_by_id.get(category_id, "") in texts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True, help="D3 COCO annotation JSON.")
    parser.add_argument("--pkl-root", type=Path, required=True, help="Directory containing D3 groups.pkl.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=200)
    args = parser.parse_args()

    annotation = _load_json(args.annotation)
    groups_raw = _load_pickle(args.pkl_root / "groups.pkl")
    sentences_raw = _load_pickle(args.pkl_root / "sentences.pkl")
    groups = {int(group_id): group for group_id, group in groups_raw.items()}
    sentences = {int(sent_id): sent for sent_id, sent in sentences_raw.items()}

    image_info_by_id = {int(item["id"]): item for item in annotation.get("images", [])}
    category_name_by_id = {int(item["id"]): str(item["name"]) for item in annotation.get("categories", [])}
    annotation_image_ids = set(image_info_by_id)
    available_group_ids = [
        int(group_id)
        for group_id, group in groups.items()
        if any(int(image_id) in annotation_image_ids for image_id in group.get("img_id", []))
    ]
    available_group_ids.sort()
    if not available_group_ids:
        raise ValueError("No D3 groups contain images from the annotation JSON.")

    rng = random.Random(args.seed)
    rng.shuffle(available_group_ids)
    if args.max_groups is not None:
        available_group_ids = available_group_ids[: args.max_groups]

    val_count, test_count = _split_counts(
        len(available_group_ids), val_ratio=args.val_ratio, test_ratio=args.test_ratio
    )
    val_group_ids = sorted(available_group_ids[:val_count])
    test_group_ids = sorted(available_group_ids[val_count : val_count + test_count])
    train_group_ids = sorted(available_group_ids[val_count + test_count :])
    all_group_ids = sorted(available_group_ids)

    group_ids_by_split = {
        "all": all_group_ids,
        "train": train_group_ids,
        "val": val_group_ids,
        "test": test_group_ids,
    }
    image_rows_by_split = {
        split_name: _image_rows(groups, image_info_by_id, group_ids)
        for split_name, group_ids in group_ids_by_split.items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, Dict[str, int]] = {}
    shard_counts: Dict[str, int] = {}
    category_ids_by_split: Dict[str, Set[int]] = {}
    prompt_texts_by_split: Dict[str, Set[str]] = {}
    for split_name, group_ids in group_ids_by_split.items():
        group_rows = _group_rows(groups, group_ids)
        image_rows = image_rows_by_split[split_name]
        counts[split_name] = {
            "groups": _write_jsonl(args.output_dir / f"{split_name}_groups.jsonl", group_rows),
            "images": _write_jsonl(args.output_dir / f"{split_name}.jsonl", image_rows),
            "annotations": _annotation_count(annotation, image_rows),
        }
        shard_counts[split_name] = _write_shards(args.output_dir, split_name, image_rows, args.shard_size)
        category_ids_by_split[split_name] = _annotation_category_ids(annotation, image_rows)
        prompt_texts_by_split[split_name] = _prompt_texts_for_groups(groups, sentences, group_ids)

    train_cats = category_ids_by_split["train"]
    val_cats = category_ids_by_split["val"]
    test_cats = category_ids_by_split["test"]
    train_prompt_texts = prompt_texts_by_split["train"]
    val_prompt_seen_cats = _category_ids_with_texts(val_cats, category_name_by_id, train_prompt_texts)
    test_prompt_seen_cats = _category_ids_with_texts(test_cats, category_name_by_id, train_prompt_texts)
    val_prompt_novel_cats = val_cats - val_prompt_seen_cats
    test_prompt_novel_cats = test_cats - test_prompt_seen_cats
    _write_jsonl(
        args.output_dir / "val_prompt_seen_categories.jsonl",
        _category_rows(sorted(val_prompt_seen_cats), category_name_by_id),
    )
    _write_jsonl(
        args.output_dir / "val_prompt_novel_categories.jsonl",
        _category_rows(sorted(val_prompt_novel_cats), category_name_by_id),
    )
    _write_jsonl(
        args.output_dir / "test_prompt_seen_categories.jsonl",
        _category_rows(sorted(test_prompt_seen_cats), category_name_by_id),
    )
    _write_jsonl(
        args.output_dir / "test_prompt_novel_categories.jsonl",
        _category_rows(sorted(test_prompt_novel_cats), category_name_by_id),
    )
    scene_counts = {
        split_name: dict(Counter(str(groups[group_id].get("scene", "")) for group_id in group_ids))
        for split_name, group_ids in group_ids_by_split.items()
    }
    summary = {
        "args": {key: _jsonable(value) for key, value in vars(args).items()},
        "annotation_images": len(annotation.get("images", [])),
        "annotation_annotations": len(annotation.get("annotations", [])),
        "available_groups": len(available_group_ids),
        "counts": counts,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "shard_size": args.shard_size,
        "shard_counts": shard_counts,
        "scene_counts": scene_counts,
        "positive_phrase_counts": {
            split_name: len(category_ids) for split_name, category_ids in category_ids_by_split.items()
        },
        "prompt_phrase_counts": {
            split_name: len(prompt_texts) for split_name, prompt_texts in prompt_texts_by_split.items()
        },
        "positive_phrase_overlap": {
            "train_val": len(train_cats & val_cats),
            "train_test": len(train_cats & test_cats),
            "val_test": len(val_cats & test_cats),
            "val_seen_in_train_ratio": len(train_cats & val_cats) / max(1, len(val_cats)),
            "test_seen_in_train_ratio": len(train_cats & test_cats) / max(1, len(test_cats)),
        },
        "prompt_level_phrase_overlap": {
            "val_positive_seen_in_train_prompt": len(val_prompt_seen_cats),
            "val_positive_prompt_novel": len(val_prompt_novel_cats),
            "val_positive_seen_in_train_prompt_ratio": len(val_prompt_seen_cats) / max(1, len(val_cats)),
            "test_positive_seen_in_train_prompt": len(test_prompt_seen_cats),
            "test_positive_prompt_novel": len(test_prompt_novel_cats),
            "test_positive_seen_in_train_prompt_ratio": len(test_prompt_seen_cats) / max(1, len(test_cats)),
            "val_prompt_seen_categories_jsonl": str(args.output_dir / "val_prompt_seen_categories.jsonl"),
            "val_prompt_novel_categories_jsonl": str(args.output_dir / "val_prompt_novel_categories.jsonl"),
            "test_prompt_seen_categories_jsonl": str(args.output_dir / "test_prompt_seen_categories.jsonl"),
            "test_prompt_novel_categories_jsonl": str(args.output_dir / "test_prompt_novel_categories.jsonl"),
        },
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
