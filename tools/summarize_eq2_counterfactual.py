#!/usr/bin/env python3
"""Summarize the same-checkpoint calibrated-vs-legacy Eq. (2) evaluation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List


METRICS = ("AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf")
RESULT_PATTERN = re.compile(r"copypaste:\s*((?:\d+\.\d+,){8}\d+\.\d+)")


def read_last_result(path: Path) -> List[float]:
    if not path.is_file():
        raise FileNotFoundError(f"evaluation log not found: {path}")
    matches = RESULT_PATTERN.findall(path.read_text(errors="ignore"))
    if not matches:
        raise ValueError(f"no LVIS copypaste result found in: {path}")
    return [float(value) for value in matches[-1].split(",")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrated-log", type=Path, required=True)
    parser.add_argument("--legacy-log", type=Path, required=True)
    args = parser.parse_args()

    calibrated = read_last_result(args.calibrated_log)
    legacy = read_last_result(args.legacy_log)
    delta = [old - new for old, new in zip(legacy, calibrated)]

    print("\nEq. (2) same-checkpoint counterfactual")
    print(f"{'variant':>12} " + " ".join(f"{name:>8}" for name in METRICS))
    for name, values in (("calibrated", calibrated), ("legacy", legacy), ("legacy-cal", delta)):
        print(f"{name:>12} " + " ".join(f"{value:8.4f}" for value in values))

    print("\nDecision metrics")
    print(f"delta_AP={delta[0]:+.4f}")
    print(f"delta_APr={delta[6]:+.4f}")


if __name__ == "__main__":
    main()
