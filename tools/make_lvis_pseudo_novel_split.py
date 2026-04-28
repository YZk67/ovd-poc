#!/usr/bin/env python3
"""Create a held-out base split for pseudo-novel calibration experiments."""

import argparse
import json
import random
from pathlib import Path


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def take_n(items, n, rng):
    if n <= 0 or not items:
        return []
    items = list(items)
    rng.shuffle(items)
    return items[: min(n, len(items))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-classes", default="dataset/lvis/lvis_v1_all_classes.json")
    parser.add_argument("--seen-classes", default="dataset/lvis/lvis_v1_seen_classes.json")
    parser.add_argument("--cat-info", default="dataset/lvis/lvis_v1_train_norare_cat_info.json")
    parser.add_argument("--output", default="dataset/lvis/pseudo_novel_base100_seed42.json")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--common-frac",
        type=float,
        default=0.7,
        help="Fraction of held-out classes sampled from common classes when available.",
    )
    args = parser.parse_args()

    all_classes = load_json(args.all_classes)
    seen_classes = set(load_json(args.seen_classes))
    cat_info = sorted(load_json(args.cat_info), key=lambda x: x["id"])

    name_to_idx = {name: idx for idx, name in enumerate(all_classes)}
    freq_by_idx = {
        idx: cat_info[idx].get("frequency", "")
        for idx in range(min(len(all_classes), len(cat_info)))
    }

    candidates = [name_to_idx[name] for name in seen_classes if name in name_to_idx]
    common = [idx for idx in candidates if freq_by_idx.get(idx) == "c"]
    frequent = [idx for idx in candidates if freq_by_idx.get(idx) == "f"]
    other = [idx for idx in candidates if idx not in set(common) and idx not in set(frequent)]

    rng = random.Random(args.seed)
    num_common = round(args.num_classes * args.common_frac)
    selected = take_n(common, num_common, rng)
    selected_set = set(selected)
    remaining = args.num_classes - len(selected)
    selected += take_n([idx for idx in frequent if idx not in selected_set], remaining, rng)
    selected_set = set(selected)
    remaining = args.num_classes - len(selected)
    selected += take_n([idx for idx in other if idx not in selected_set], remaining, rng)
    selected = sorted(selected)

    payload = {
        "seed": args.seed,
        "num_classes": len(selected),
        "class_ids": selected,
        "class_names": [all_classes[idx] for idx in selected],
        "frequencies": [freq_by_idx.get(idx, "") for idx in selected],
        "sources": {
            "all_classes": args.all_classes,
            "seen_classes": args.seen_classes,
            "cat_info": args.cat_info,
        },
        "sampling": {
            "requested_num_classes": args.num_classes,
            "common_frac": args.common_frac,
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    freq_counts = {}
    for freq in payload["frequencies"]:
        freq_counts[freq] = freq_counts.get(freq, 0) + 1
    print(f"wrote {output} with {len(selected)} classes; freq_counts={freq_counts}")


if __name__ == "__main__":
    main()
