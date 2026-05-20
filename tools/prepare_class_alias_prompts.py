#!/usr/bin/env python3
"""Build ordered multi-description alias prompts for class-level OVD datasets.

The output JSON is accepted by tools/generate_text_embeddings.py. Each class gets
the same number of prompts, so running the embedding script with
``--aggregate none`` produces a dense [num_classes, num_aliases, dim] bank.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence


VISUAL_TEMPLATES = [
    "{name}",
    "a photo of {noun}",
    "a cropped region showing {noun}",
    "the target object is {noun}",
    "a visible {name} in the image",
    "the image region containing {noun}",
    "visual appearance of {noun}",
    "an object described as {name}",
]


def _article_for(text: str) -> str:
    first = text.strip().split(" ", 1)[0].lower()
    if first[:1] in {"a", "e", "i", "o", "u"}:
        return "an"
    return "a"


def _noun_phrase(text: str) -> str:
    text = " ".join(text.strip().split())
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


def _load_categories(path: Path) -> List[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, Mapping) and "categories" in data:
        raw_categories = data["categories"]
    elif isinstance(data, Mapping):
        raw_categories = [
            {"name": str(name), "aliases": aliases}
            for name, aliases in data.items()
        ]
    elif isinstance(data, list):
        raw_categories = data
    else:
        raise ValueError(f"Unsupported class JSON format in {path}.")

    categories: List[Mapping[str, Any]] = []
    for idx, item in enumerate(raw_categories):
        if isinstance(item, str):
            categories.append({"id": idx + 1, "name": item})
        elif isinstance(item, Mapping):
            if "name" not in item:
                raise ValueError(f"Category item lacks a 'name' field: {item!r}")
            categories.append(item)
        else:
            raise ValueError(f"Unsupported category item: {item!r}")

    return sorted(categories, key=lambda cat: int(cat.get("id", len(categories))))


def _category_aliases(category: Mapping[str, Any], max_synonyms: int) -> List[str]:
    name = str(category["name"]).strip()
    aliases = [name]
    for key in ("aliases", "synonyms"):
        values = category.get(key, [])
        if isinstance(values, str):
            values = [values]
        for value in values[:max_synonyms]:
            _add_unique(aliases, str(value))
    return aliases


def build_prompts(
    name: str,
    aliases: Sequence[str],
    *,
    prompt_count: int,
    templates: Iterable[str] = VISUAL_TEMPLATES,
) -> List[str]:
    prompts: List[str] = []
    for alias in aliases:
        _add_unique(prompts, alias)
        if len(prompts) >= prompt_count:
            return prompts[:prompt_count]

    for template in templates:
        _add_unique(
            prompts,
            template.format(name=name, noun=_noun_phrase(name)),
        )
        if len(prompts) >= prompt_count:
            return prompts[:prompt_count]

    raise ValueError(f"Could not build {prompt_count} prompts for class {name!r}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classes-json",
        type=Path,
        required=True,
        help=(
            "Class source. Supports a JSON list of names, a COCO/LVIS JSON with "
            "categories, a list of category dicts, or a name->aliases mapping."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output ordered prompt mapping JSON.",
    )
    parser.add_argument("--prompt-count", type=int, default=5)
    parser.add_argument("--max-synonyms", type=int, default=4)
    args = parser.parse_args()

    if args.prompt_count <= 0:
        raise ValueError("--prompt-count must be positive.")

    categories = _load_categories(args.classes_json)
    prompt_bank = OrderedDict()
    for idx, category in enumerate(categories, start=1):
        name = str(category["name"]).strip()
        aliases = _category_aliases(category, max_synonyms=args.max_synonyms)
        prompts = build_prompts(name, aliases, prompt_count=args.prompt_count)
        prompt_bank[f"{idx:04d}: {name}"] = prompts

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(prompt_bank, handle, indent=2)

    print(f"saved {args.output}")
    print(f"num_classes: {len(prompt_bank)}")
    print(f"num_prompts_per_class: {args.prompt_count}")
    if prompt_bank:
        first_key = next(iter(prompt_bank))
        print(f"first: {first_key} -> {prompt_bank[first_key]}")


if __name__ == "__main__":
    main()
