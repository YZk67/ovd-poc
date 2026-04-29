"""One-shot builder for `data/mda/confusable_index.json`.

Equivalent to step1 + (step2 alternative) + step3, but bypasses the offline
CLIP encode in step2 by reusing pre-computed CLIP text-similarity candidates
(e.g. OV-DQUO's `lvis_clip_candidates.json`).

Output format matches mmdetection's `confusable_index.json`:
    {"42": [78, 103, 215], "78": [42, 89, 156], ...}
where keys/values are LVIS continuous indices (= category_id - 1).

Usage:
    python tools/mda/build_confusable_index.py \
        --candidates /path/to/lvis_clip_candidates.json \
        --out data/mda/confusable_index.json \
        [--top-k 3] [--exclude-rare-negatives]
"""
import argparse
import importlib.util
import json
import os
import sys

LVIS_CATS_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir,
    "detectron2/detectron2/data/datasets/lvis_v1_categories.py",
)


def load_lvis_categories():
    spec = importlib.util.spec_from_file_location(
        "lvc", os.path.abspath(LVIS_CATS_PATH))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.LVIS_CATEGORIES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates", required=True,
        help="JSON of {class_name: [[neighbor_name, score], ...]}, "
             "e.g. OV-DQUO's lvis_clip_candidates.json")
    parser.add_argument(
        "--out", default="data/mda/confusable_index.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--exclude-rare-negatives",
        action="store_true",
        help="OV-LVIS mode: never use rare/novel classes as margin negatives.",
    )
    args = parser.parse_args()

    cats = load_lvis_categories()
    name_to_cont = {c["name"]: c["id"] - 1 for c in cats}
    base_conts = [c["id"] - 1 for c in cats if c["frequency"] in ("c", "f")]
    rare_conts = {c["id"] - 1 for c in cats if c["frequency"] == "r"}

    with open(args.candidates) as f:
        cand = json.load(f)

    result = {}
    miss_pos = miss_neg = 0
    for cont in base_conts:
        name = cats[cont]["name"]
        if name not in cand:
            miss_pos += 1
            continue
        negs = []
        for entry in cand[name]:
            n_name = entry[0] if isinstance(entry, (list, tuple)) else entry
            if n_name == name:
                continue
            n_cont = name_to_cont.get(n_name)
            if n_cont is None:
                miss_neg += 1
                continue
            if args.exclude_rare_negatives and n_cont in rare_conts:
                continue
            negs.append(n_cont)
            if len(negs) == args.top_k:
                break
        if negs:
            result[str(cont)] = negs

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Built {len(result)} entries (skipped pos={miss_pos}, neg={miss_neg})")
    if args.exclude_rare_negatives:
        all_negs = [n for v in result.values() for n in v]
        rare_hits = sum(1 for n in all_negs if n in rare_conts)
        print(f"Rare negatives: {rare_hits}/{len(all_negs)}")
    print(f"Saved to {args.out}")
    print("\nSample:")
    cont_to_name = {c["id"] - 1: c["name"] for c in cats}
    for k in list(result.keys())[:5]:
        pos = cont_to_name[int(k)]
        neg = [cont_to_name[i] for i in result[k]]
        print(f"  cont[{k:>4}] {pos:30s} -> {neg}")


if __name__ == "__main__":
    main()
