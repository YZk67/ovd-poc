#!/usr/bin/env python3
"""
Build a small ordered multi-description prompt bank for D3/DOD phrases.

The output is a JSON object accepted by tools/generate_text_embeddings.py.
Each D3 phrase receives the same number of prompts so --aggregate none produces
a dense embedding array with shape [num_phrases, num_prompts, dim].
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List


DEFAULT_TEMPLATES = [
    "{phrase}",
    "a photo of {phrase}",
    "a cropped photo of {phrase}",
    "the image region showing {phrase}",
    "the object or person described as {phrase}",
    "visual attributes and relations: {phrase}",
]


def _load_phrases(path: Path) -> List[str]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of phrases in {path}, got {type(data).__name__}.")
    phrases = [str(item).strip() for item in data]
    if any(not phrase for phrase in phrases):
        raise ValueError("D3 phrase list contains an empty phrase.")
    return phrases


def _build_prompts(phrase: str, templates: Iterable[str]) -> List[str]:
    prompts = []
    seen = set()
    for template in templates:
        prompt = template.format(phrase=phrase).strip()
        if prompt and prompt not in seen:
            prompts.append(prompt)
            seen.add(prompt)
    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phrases-json",
        type=Path,
        default=Path("dataset/metadata/d3_phrases.json"),
        help="Ordered D3 phrase list produced by tools/prepare_d3_metadata.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/metadata/d3_description_prompts.json"),
        help="Output JSON mapping ordered phrase keys to prompt lists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phrases = _load_phrases(args.phrases_json)

    prompt_bank = OrderedDict()
    expected_count = None
    for idx, phrase in enumerate(phrases, start=1):
        prompts = _build_prompts(phrase, DEFAULT_TEMPLATES)
        if expected_count is None:
            expected_count = len(prompts)
        if len(prompts) != expected_count:
            raise ValueError(
                f"Prompt count mismatch for phrase {idx}: expected {expected_count}, got {len(prompts)}."
            )
        prompt_bank[f"{idx:03d}: {phrase}"] = prompts

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(prompt_bank, f, indent=2)

    print(f"Saved D3 description prompts to {args.output}")
    print(f"num_phrases: {len(prompt_bank)}")
    print(f"num_prompts_per_phrase: {expected_count or 0}")
    if prompt_bank:
        first_key = next(iter(prompt_bank))
        print(f"first: {first_key} -> {prompt_bank[first_key]}")


if __name__ == "__main__":
    main()
