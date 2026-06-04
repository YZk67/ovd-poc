#!/usr/bin/env python3
"""Filter a COCO annotation file to category ids listed in a JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Set


def _load_category_ids(path: Path) -> Set[int]:
    category_ids: Set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            value: Any = json.loads(line)
            if isinstance(value, dict):
                value = value["category_id"]
            category_ids.add(int(value))
    return category_ids


def filter_coco(
    annotation: Dict[str, Any],
    category_ids: Set[int],
    *,
    drop_empty_images: bool,
    filter_categories: bool,
) -> Dict[str, Any]:
    filtered = dict(annotation)
    filtered["annotations"] = [
        item for item in annotation.get("annotations", []) if int(item["category_id"]) in category_ids
    ]
    if drop_empty_images:
        kept_image_ids = {int(item["image_id"]) for item in filtered["annotations"]}
        filtered["images"] = [
            item for item in annotation.get("images", []) if int(item["id"]) in kept_image_ids
        ]
    else:
        filtered["images"] = list(annotation.get("images", []))
    if filter_categories:
        filtered["categories"] = [
            item for item in annotation.get("categories", []) if int(item["id"]) in category_ids
        ]
    else:
        filtered["categories"] = list(annotation.get("categories", []))
    return filtered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--category-id-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--drop-empty-images",
        action="store_true",
        help=(
            "Drop images with no remaining annotations. Leave this off for eval "
            "subsets so false positives on selected categories are still counted."
        ),
    )
    parser.add_argument(
        "--filter-categories",
        action="store_true",
        help=(
            "Drop category metadata not listed in --category-id-jsonl. By default "
            "the full category bank is preserved for D3/DOD compatibility."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.annotation.open("r", encoding="utf-8") as handle:
        annotation = json.load(handle)
    category_ids = _load_category_ids(args.category_id_jsonl)
    filtered = filter_coco(
        annotation,
        category_ids,
        drop_empty_images=args.drop_empty_images,
        filter_categories=args.filter_categories,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(filtered, handle, separators=(",", ":"))
    summary = {
        "input_images": len(annotation.get("images", [])),
        "input_annotations": len(annotation.get("annotations", [])),
        "input_categories": len(annotation.get("categories", [])),
        "requested_categories": len(category_ids),
        "output_images": len(filtered["images"]),
        "output_annotations": len(filtered["annotations"]),
        "output_categories": len(filtered["categories"]),
        "drop_empty_images": args.drop_empty_images,
        "filter_categories": args.filter_categories,
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
