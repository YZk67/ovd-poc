#!/usr/bin/env python3
"""
Mix two text-prototype slots with fixed scalar weights.

The intended D3 usage is:

    prototype = normalize((1 - w) * phrase + w * target_prompt)

where the input bank has shape [num_classes, 2, dim] and slot 0 is the raw
phrase embedding while slot 1 is the target-framed prompt embedding.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


def _weight_tag(weight: float) -> str:
    value = int(round(weight * 100))
    return f"w{value:03d}"


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(arr, axis=-1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return arr / denom


def _format_output_path(template: str, weight: float) -> Path:
    return Path(template.format(weight=weight, weight_tag=_weight_tag(weight)))


def mix_bank(
    source: np.ndarray,
    *,
    weights: Iterable[float],
    phrase_index: int,
    target_index: int,
    output_template: str,
) -> None:
    if source.ndim != 3:
        raise ValueError(f"Expected source bank shape [classes, prompts, dim], got {source.shape}.")
    if source.shape[1] <= max(phrase_index, target_index):
        raise ValueError(
            "Source bank does not have enough prompt slots for "
            f"phrase_index={phrase_index}, target_index={target_index}: shape={source.shape}."
        )

    phrase = source[:, phrase_index, :].astype(np.float32)
    target = source[:, target_index, :].astype(np.float32)

    for weight in weights:
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Weight must be in [0, 1], got {weight}.")
        mixed = (1.0 - weight) * phrase + weight * target
        mixed = _normalize_rows(mixed).astype(np.float32)

        output = _format_output_path(output_template, weight)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, mixed)
        print(f"saved w={weight:.4f} -> {output} shape={mixed.shape} dtype={mixed.dtype}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/metadata/d3_description_anchor_target_bank_convnextl.npy"),
        help="Input .npy bank with shape [num_classes, num_prompts, dim].",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0],
        help="Mixture weights for the target prompt.",
    )
    parser.add_argument(
        "--phrase-index",
        type=int,
        default=0,
        help="Prompt slot containing the raw phrase embedding.",
    )
    parser.add_argument(
        "--target-index",
        type=int,
        default=1,
        help="Prompt slot containing the target-framed prompt embedding.",
    )
    parser.add_argument(
        "--output-template",
        default="dataset/metadata/d3_description_anchor_target_{weight_tag}_bank_convnextl.npy",
        help=(
            "Output template. Supports {weight} and {weight_tag}; "
            "{weight_tag} formats 0.25 as w025."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = np.load(args.input)
    print(f"loaded {args.input} shape={source.shape} dtype={source.dtype}")
    mix_bank(
        source,
        weights=args.weights,
        phrase_index=args.phrase_index,
        target_index=args.target_index,
        output_template=args.output_template,
    )


if __name__ == "__main__":
    main()
