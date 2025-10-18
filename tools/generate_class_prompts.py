#!/usr/bin/env python3
"""
Utility script to extract category names (and optional prompt templates)
from LVIS-style annotation files. This helps prepare the 1203-class
vocabulary and phrase bank required by the Text Prototype Aggregator.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


DEFAULT_TEMPLATES = [
    "a photo of {}",
    "a close-up photo of {}",
    "a detailed photo of {}",
    "{} on a plain background",
    "a side view of {}",
]

def get_article(word: str) -> str:
    """
    Return 'an' if word starts with a vowel sound, otherwise 'a'.
    This is a simple heuristic and may not be perfect for all cases.
    """
    if not word:
        return "a"
    
    # Check if starts with vowel
    first_char = word[0].lower()
    if first_char in ['a', 'e', 'i', 'o', 'u']:
        return "an"
    
    # Special cases for words starting with 'h' (silent h)
    # e.g., "an hour", "an honor", but "a house"
    silent_h_words = ['hour', 'honor', 'honest', 'heir']
    if word.lower().startswith('h'):
        for silent_word in silent_h_words:
            if word.lower().startswith(silent_word):
                return "an"
    
    return "a"


def load_categories(path: Path) -> List[Dict]:
    with path.open("r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "categories" in data:
        categories = data["categories"]
    else:
        categories = data
    if not isinstance(categories, list):
        raise ValueError(f"Unrecognized annotation format in {path}")
    return categories


def build_prompts(name: str, synonyms: List[str], max_synonyms: int, use_synonyms: bool = True) -> List[str]:
    """
    Build prompts from class name and synonyms.
    
    Args:
        name: The class name from the annotation file
        synonyms: List of synonyms (in LVIS, first element is usually the canonical name)
        max_synonyms: Maximum number of synonyms to use
        use_synonyms: If True, use multiple synonyms; if False, only use canonical name
    
    Note: In LVIS, the synonyms list's first element is the canonical/preferred name.
    """
    phrases = []
    
    if use_synonyms:
        # Use synonyms (which usually includes the name)
        for syn in synonyms[:max_synonyms]:
            syn = syn.strip()
            if syn:
                phrases.append(syn)
        
        # Ensure the original name is included (in case synonyms doesn't have it)
        if name.strip() not in phrases:
            phrases.insert(0, name.strip())
    else:
        # Only use canonical name (first synonym, or name if no synonyms)
        canonical_name = synonyms[0].strip() if synonyms else name.strip()
        phrases = [canonical_name]
    
    prompts = []
    for phrase in dict.fromkeys(phrases):  # preserve order / remove duplicates
        # Get the appropriate article (a/an) for this phrase
        article = get_article(phrase)
        phrase_with_article = f"{article} {phrase}"
        
        for template in DEFAULT_TEMPLATES:
            # Simply format the template with the phrase (with article already added)
            prompt = template.format(phrase_with_article)
            prompts.append(prompt)
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ann",
        type=Path,
        required=True,
        help="Path to LVIS-style annotation file (e.g., lvis_v1_train_norare_cat_info.json).",
    )
    parser.add_argument(
        "--class-output",
        type=Path,
        default=None,
        help="Where to save the ordered class-name list (JSON). Default: print to stdout.",
    )
    parser.add_argument(
        "--prompt-output",
        type=Path,
        default=None,
        help="Optional path to save the {class_name: [phrases]} mapping (JSON).",
    )
    parser.add_argument(
        "--max-synonyms",
        type=int,
        default=5,
        help="Maximum synonyms per class to consider (additional synonyms are ignored).",
    )
    parser.add_argument(
        "--canonical-only",
        action="store_true",
        help="Only use canonical name (first synonym) instead of all synonyms. "
             "This generates exactly 5 prompts per class.",
    )
    args = parser.parse_args()

    categories = load_categories(args.ann)
    categories = sorted(categories, key=lambda cat: cat.get("id", 0))

    class_names = [cat["name"] for cat in categories]

    if args.class_output:
        args.class_output.parent.mkdir(parents=True, exist_ok=True)
        with args.class_output.open("w") as f:
            json.dump(class_names, f, indent=2)
    else:
        print(json.dumps(class_names, indent=2))

    if args.prompt_output:
        args.prompt_output.parent.mkdir(parents=True, exist_ok=True)
        prompts = {
            cat["name"]: build_prompts(
                cat["name"],
                cat.get("synonyms", []) or [],
                args.max_synonyms,
                use_synonyms=not args.canonical_only,
            )
            for cat in categories
        }
        with args.prompt_output.open("w") as f:
            json.dump(prompts, f, indent=2)


if __name__ == "__main__":
    main()

# cd /Users/zhengjiankang/Downloads/research/research/ovd-poc
# source .venv/bin/activate

# python tools/generate_class_prompts.py \
#   --ann dataset2/lvis/lvis_v1_train_norare_cat_info.json \
#   --class-output dataset2/metadata/lvis_class_names.json \
#   --prompt-output dataset2/metadata/lvis_prompts.json \
#   --max-synonyms 5
