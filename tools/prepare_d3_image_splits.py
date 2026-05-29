#!/usr/bin/env python3
"""Create deterministic D3 image-id JSONL splits and optional shards."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


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
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _load_image_ids_from_jsonl(path: Path) -> Set[int]:
    image_ids: Set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image_ids.add(int(row["image_id"]))
    return image_ids


def _load_prediction_image_ids(path: Path) -> Set[int]:
    rows = _load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected COCO result list in {path}, got {type(rows).__name__}.")
    return {int(row["image_id"]) for row in rows}


def _image_rows(annotation: Mapping[str, Any], image_ids: Sequence[int]) -> List[Dict[str, Any]]:
    info_by_id = {int(item["id"]): item for item in annotation.get("images", [])}
    rows: List[Dict[str, Any]] = []
    for image_id in image_ids:
        info = info_by_id[int(image_id)]
        rows.append(
            {
                "image_id": int(image_id),
                "file_name": str(info.get("file_name", "")),
                "width": int(info.get("width", 0)),
                "height": int(info.get("height", 0)),
            }
        )
    return rows


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True, help="D3 COCO annotation JSON.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Optional COCO result JSON; keep only annotation images that have predictions.",
    )
    parser.add_argument(
        "--exclude-image-id-jsonl",
        type=Path,
        action="append",
        default=[],
        help="Image-id JSONL to exclude. Repeat to exclude multiple held-out sets.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=200)
    args = parser.parse_args()

    annotation = _load_json(args.annotation)
    image_ids = {int(item["id"]) for item in annotation.get("images", [])}
    initial_image_count = len(image_ids)

    prediction_image_count = None
    if args.predictions is not None:
        prediction_ids = _load_prediction_image_ids(args.predictions)
        prediction_image_count = len(prediction_ids)
        image_ids &= prediction_ids

    excluded_ids: Set[int] = set()
    for path in args.exclude_image_id_jsonl:
        excluded_ids |= _load_image_ids_from_jsonl(path)
    image_ids -= excluded_ids

    ids = sorted(image_ids)
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    if args.max_images is not None:
        ids = ids[: args.max_images]
    if not ids:
        raise ValueError(
            "No images remain after filtering. Check --predictions and --exclude-image-id-jsonl; "
            "if all annotation images are held out, you need a separate training annotation/prediction source."
        )

    val_count, test_count = _split_counts(len(ids), val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    val_ids = sorted(ids[:val_count])
    test_ids = sorted(ids[val_count : val_count + test_count])
    train_ids = sorted(ids[val_count + test_count :])
    all_ids = sorted(ids)

    rows_by_split = {
        "all": _image_rows(annotation, all_ids),
        "train": _image_rows(annotation, train_ids),
        "val": _image_rows(annotation, val_ids),
        "test": _image_rows(annotation, test_ids),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    shard_counts = {}
    for split_name, rows in rows_by_split.items():
        counts[split_name] = _write_jsonl(args.output_dir / f"{split_name}.jsonl", rows)
        shard_counts[split_name] = _write_shards(args.output_dir, split_name, rows, args.shard_size)

    summary = {
        "annotation_images": initial_image_count,
        "prediction_images": prediction_image_count,
        "excluded_images": len(excluded_ids),
        "available_images": len(all_ids),
        "counts": counts,
        "shard_counts": shard_counts,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "shard_size": args.shard_size,
        "args": {key: _jsonable(value) for key, value in vars(args).items()},
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
