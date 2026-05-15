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
import re
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List


GENERIC_TEMPLATES = [
    "{phrase}",
    "a photo of {phrase}",
    "a cropped photo of {phrase}",
    "the image region showing {phrase}",
    "the object or person described as {phrase}",
    "visual attributes and relations: {phrase}",
]

ANCHOR_FALLBACK_TEMPLATES = [
    "the described target is {phrase}",
    "a visible region matching {phrase}",
]


def _article_for(text: str) -> str:
    first = text.strip().split(" ", 1)[0].lower()
    if first[:1] in {"a", "e", "i", "o", "u"}:
        return "an"
    return "a"


def _noun_phrase(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    first = text.split(" ", 1)[0].lower()
    if first in {"a", "an", "the", "this", "that", "these", "those", "one"}:
        return text
    return f"{_article_for(text)} {text}"


def _add_unique(items: List[str], candidate: str) -> None:
    candidate = " ".join(candidate.strip().split())
    if candidate and candidate not in items:
        items.append(candidate)


def _heuristic_paraphrases(phrase: str) -> List[str]:
    paraphrases: List[str] = []
    lower = phrase.lower()

    if lower.endswith(" in hand"):
        obj = phrase[: -len(" in hand")]
        _add_unique(paraphrases, f"a person holding {_noun_phrase(obj)}")
        _add_unique(paraphrases, f"{_noun_phrase(obj)} held in a hand")

    if " led by rope" in lower:
        subj = phrase[: lower.index(" led by rope")]
        _add_unique(paraphrases, f"{_noun_phrase(subj)} attached to a rope")
        _add_unique(paraphrases, f"{_noun_phrase(subj)} on a leash")

    held_match = re.match(r"(.+?) held by (someone|a person|people|human|humans)$", lower)
    if held_match:
        obj = phrase[: held_match.end(1)]
        holder = held_match.group(2)
        _add_unique(paraphrases, f"{holder} holding {_noun_phrase(obj)}")

    carried_match = re.match(r"(.+?) carried by (someone|a person|people|human|humans)$", lower)
    if carried_match:
        obj = phrase[: carried_match.end(1)]
        holder = carried_match.group(2)
        _add_unique(paraphrases, f"{holder} carrying {_noun_phrase(obj)}")

    on_match = re.match(r"(.+?) on (the |a |an )?(.+)$", phrase, flags=re.IGNORECASE)
    if on_match and " not on " not in lower:
        subj = on_match.group(1)
        surface = on_match.group(3)
        _add_unique(paraphrases, f"{_noun_phrase(subj)} located on {surface}")

    with_match = re.match(r"(.+?) with (.+)$", phrase, flags=re.IGNORECASE)
    if with_match and " without " not in lower:
        subj, attr = with_match.group(1), with_match.group(2)
        _add_unique(paraphrases, f"{subj} that has {attr}")

    without_match = re.match(r"(.+?) without (.+)$", phrase, flags=re.IGNORECASE)
    if without_match:
        subj, attr = without_match.group(1), without_match.group(2)
        _add_unique(paraphrases, f"{subj} with no {attr}")

    if lower.startswith("a person who "):
        rest = phrase[len("a person who ") :]
        if rest.lower().startswith("is "):
            rest = rest[3:]
        elif rest.lower().startswith("are "):
            rest = rest[4:]
        _add_unique(paraphrases, f"a person {rest}")
    elif lower.startswith("the person who "):
        rest = phrase[len("the person who ") :]
        if rest.lower().startswith("is "):
            rest = rest[3:]
        elif rest.lower().startswith("are "):
            rest = rest[4:]
        _add_unique(paraphrases, f"a person {rest}")

    return paraphrases


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


def _build_anchored_prompts(phrase: str, prompt_count: int) -> List[str]:
    prompts = [phrase]
    for paraphrase in _heuristic_paraphrases(phrase):
        _add_unique(prompts, paraphrase)
    for template in ANCHOR_FALLBACK_TEMPLATES:
        _add_unique(prompts, template.format(phrase=phrase))
    return prompts[:prompt_count]


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
        default=Path("dataset/metadata/d3_description_anchor_prompts.json"),
        help="Output JSON mapping ordered phrase keys to prompt lists.",
    )
    parser.add_argument(
        "--preset",
        choices=("anchored", "generic6"),
        default="anchored",
        help="Prompt recipe. anchored keeps the original phrase dominant; generic6 is the older broad-template bank.",
    )
    parser.add_argument(
        "--prompt-count",
        type=int,
        default=3,
        help="Number of prompts per phrase for the anchored preset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phrases = _load_phrases(args.phrases_json)

    prompt_bank = OrderedDict()
    expected_count = None
    for idx, phrase in enumerate(phrases, start=1):
        if args.preset == "generic6":
            prompts = _build_prompts(phrase, GENERIC_TEMPLATES)
        else:
            prompts = _build_anchored_prompts(phrase, args.prompt_count)
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
