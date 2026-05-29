#!/usr/bin/env python3
"""Merge sharded D3 VLM score JSONL caches with key-level de-duplication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def _iter_input_paths(paths: Iterable[Path]) -> List[Path]:
    result: List[Path] = []
    for path in paths:
        if path.is_dir():
            result.extend(sorted(path.rglob("*.jsonl")))
        elif path.exists():
            result.append(path)
    return sorted(dict.fromkeys(result))


def _fallback_key(row: Mapping[str, Any]) -> str:
    bbox = ",".join(f"{float(value):.3f}" for value in row.get("bbox", []))
    return f'{int(row["image_id"])}:{int(row["category_id"])}:{bbox}'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Input cache JSONL files or directories.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows_by_key: Dict[str, Dict[str, Any]] = {}
    input_rows = 0
    duplicate_rows = 0
    input_paths = _iter_input_paths(args.inputs)
    for path in input_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                input_rows += 1
                key = str(row.get("key") or _fallback_key(row))
                if key in rows_by_key:
                    duplicate_rows += 1
                rows_by_key[key] = row

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image_ids = set()
    with args.output.open("w", encoding="utf-8") as handle:
        for key in sorted(rows_by_key):
            row = rows_by_key[key]
            image_ids.add(int(row["image_id"]))
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "input_files": len(input_paths),
        "input_rows": input_rows,
        "duplicate_rows": duplicate_rows,
        "output_rows": len(rows_by_key),
        "output_images": len(image_ids),
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
