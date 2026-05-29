#!/usr/bin/env python3
"""Summarize D3 rerank summary JSON files into a compact table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


METRIC_FIELDS = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100"]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_summary_paths(paths: Iterable[Path]) -> List[Path]:
    result: List[Path] = []
    for path in paths:
        if path.is_dir():
            result.extend(sorted(path.rglob("*.summary.json")))
        elif path.exists():
            result.append(path)
    return sorted(dict.fromkeys(result))


def _row_from_summary(path: Path) -> Dict[str, Any]:
    summary = _load_json(path)
    args = summary.get("args", {})
    metrics = summary.get("coco_eval", {})
    output = str(args.get("output", ""))
    row: Dict[str, Any] = {
        "name": path.name.removesuffix(".summary.json"),
        "parent": path.parent.name,
        "predictions": args.get("predictions", ""),
        "output": output,
        "checkpoint": args.get("verifier_checkpoint", ""),
        "fusion": args.get("verifier_fusion", args.get("fusion", "")),
        "weight": args.get("verifier_fusion_weight", args.get("fusion_weight", "")),
        "proposal_topk": args.get("proposal_topk_per_image", ""),
        "keep_topk": args.get("keep_topk_per_image", ""),
        "base_score": args.get("expanded_base_score", ""),
        "output_predictions": summary.get("output_predictions", ""),
        "output_images": summary.get("output_images", ""),
        "eval_image_ids": summary.get("eval_image_ids", ""),
    }
    for field in METRIC_FIELDS:
        row[field] = metrics.get(field, "")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Summary JSON files or directories to scan.")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV path.")
    args = parser.parse_args()

    rows = [_row_from_summary(path) for path in _iter_summary_paths(args.paths)]
    fields = [
        "name",
        "parent",
        "AP",
        "AP50",
        "AP75",
        "APs",
        "APm",
        "APl",
        "AR1",
        "AR10",
        "AR100",
        "weight",
        "checkpoint",
        "predictions",
        "output",
        "fusion",
        "proposal_topk",
        "keep_topk",
        "base_score",
        "output_predictions",
        "output_images",
        "eval_image_ids",
    ]

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
