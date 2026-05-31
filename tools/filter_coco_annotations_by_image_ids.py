#!/usr/bin/env python3
"""Filter a COCO annotation file to image ids listed in a JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Set


def _load_image_ids(path: Path) -> Set[int]:
    image_ids: Set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            value: Any = json.loads(line)
            if isinstance(value, dict):
                value = value["image_id"]
            image_ids.add(int(value))
    return image_ids


def filter_coco(annotation: Dict[str, Any], image_ids: Set[int]) -> Dict[str, Any]:
    filtered = dict(annotation)
    filtered["images"] = [item for item in annotation.get("images", []) if int(item["id"]) in image_ids]
    kept = {int(item["id"]) for item in filtered["images"]}
    filtered["annotations"] = [
        item for item in annotation.get("annotations", []) if int(item["image_id"]) in kept
    ]
    return filtered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--image-id-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.annotation.open("r", encoding="utf-8") as handle:
        annotation = json.load(handle)
    image_ids = _load_image_ids(args.image_id_jsonl)
    filtered = filter_coco(annotation, image_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(filtered, handle, separators=(",", ":"))
    summary = {
        "input_images": len(annotation.get("images", [])),
        "input_annotations": len(annotation.get("annotations", [])),
        "requested_images": len(image_ids),
        "output_images": len(filtered["images"]),
        "output_annotations": len(filtered["annotations"]),
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
