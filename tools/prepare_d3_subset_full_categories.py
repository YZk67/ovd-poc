#!/usr/bin/env python3
"""Build D3 pres/abs JSONs with the full 422-category list.

Official D3 pres/abs COCO JSONs only contain the positive/negative category
subsets. Detectron2 then maps those subset category ids to contiguous ids, which
is incompatible with a detector that always predicts the full 422 D3 phrase
bank. This helper keeps the official images/annotations but replaces categories
with the full D3 category list.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle)


def build_subset(full_json: Path, subset_json: Path, output_json: Path) -> None:
    full = load_json(full_json)
    subset = load_json(subset_json)
    if len(full.get("categories", [])) < len(subset.get("categories", [])):
        raise ValueError("full_json has fewer categories than subset_json.")
    subset["categories"] = full["categories"]
    dump_json(subset, output_json)
    print(
        f"saved {output_json}: "
        f"images={len(subset.get('images', []))} "
        f"annotations={len(subset.get('annotations', []))} "
        f"categories={len(subset.get('categories', []))}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-json",
        type=Path,
        default=Path("dataset/d3/d3_json/d3_full_annotations.json"),
    )
    parser.add_argument(
        "--pres-json",
        type=Path,
        default=Path("dataset/d3/d3_json/d3_pres_annotations.json"),
    )
    parser.add_argument(
        "--abs-json",
        type=Path,
        default=Path("dataset/d3/d3_json/d3_abs_annotations.json"),
    )
    parser.add_argument(
        "--pres-output",
        type=Path,
        default=Path("dataset/d3/annotations/d3_pres_fullcats.json"),
    )
    parser.add_argument(
        "--abs-output",
        type=Path,
        default=Path("dataset/d3/annotations/d3_abs_fullcats.json"),
    )
    args = parser.parse_args()

    build_subset(args.full_json, args.pres_json, args.pres_output)
    build_subset(args.full_json, args.abs_json, args.abs_output)


if __name__ == "__main__":
    main()
