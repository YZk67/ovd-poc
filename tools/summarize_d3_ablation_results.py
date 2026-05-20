#!/usr/bin/env python
"""Summarize D3 frozen ROI-verifier ablation logs.

Example:
  python tools/summarize_d3_ablation_results.py \
    --root /root/autodl-tmp/LaMI-DETR-output/d3_frozen_roi_verifier_ablation
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


COPypaste_RE = re.compile(
    r"copypaste:\s+"
    r"(?P<ap>-?\d+(?:\.\d+)?),"
    r"(?P<ap50>-?\d+(?:\.\d+)?),"
    r"(?P<ap75>-?\d+(?:\.\d+)?),"
    r"(?P<aps>-?\d+(?:\.\d+)?),"
    r"(?P<apm>-?\d+(?:\.\d+)?),"
    r"(?P<apl>-?\d+(?:\.\d+)?)"
)


def parse_last_copypaste(log_path: Path) -> dict[str, str] | None:
    last_match = None
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = COPypaste_RE.search(line)
            if match:
                last_match = match
    return last_match.groupdict() if last_match else None


def iter_results(root: Path):
    for log_path in sorted(root.glob("*/*/log.txt")):
        metrics = parse_last_copypaste(log_path)
        if metrics is None:
            continue
        split = log_path.parent.parent.name
        setting = log_path.parent.name
        yield {
            "split": split,
            "setting": setting,
            **metrics,
            "log": str(log_path),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/root/autodl-tmp/LaMI-DETR-output/d3_frozen_roi_verifier_ablation"),
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV path.")
    args = parser.parse_args()

    rows = list(iter_results(args.root))
    fields = ["split", "setting", "ap", "ap50", "ap75", "aps", "apm", "apl", "log"]

    if args.output:
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
