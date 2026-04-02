#!/usr/bin/env python3
"""Fix string-vlm annotations by reading crops and generating proper VLM JSON.

This script is meant to be run interactively by Claude Code (the AI reads the images).
It outputs a batch file that can then be applied to the description JSONs.

Usage:
    # Generate the work list
    python tools/claude_fix_string_vlm.py --mode list

    # After Claude Code processes batches, apply results
    python tools/claude_fix_string_vlm.py --mode apply \
        --results /tmp/claude_crop_results.json
"""

import json
import os
import sys
from collections import defaultdict

DESC_DIR = "dataset/vlm_targets_crop/descriptions"
CROP_DIR = "dataset/vlm_targets_crop/crops"
RESULTS_FILE = "/tmp/claude_crop_results.json"

ALL_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "kite", "skateboard", "surfboard", "bottle", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "pizza", "donut", "cake", "chair", "couch", "bed", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "toothbrush"
]


def apply_results(results_file):
    """Apply Claude-generated results back to description JSON files."""
    with open(results_file) as f:
        results = json.load(f)

    # Group by image_id
    by_img = defaultdict(dict)
    for r in results:
        by_img[r["img_id"]][r["ann_id"]] = r["vlm"]

    updated = 0
    anns_fixed = 0
    for img_id, ann_vlms in sorted(by_img.items()):
        fpath = os.path.join(DESC_DIR, f"{img_id}.json")
        if not os.path.exists(fpath):
            print(f"WARNING: {fpath} not found")
            continue

        with open(fpath) as f:
            data = json.load(f)

        modified = False
        for a in data["annotations"]:
            if not isinstance(a, dict):
                continue
            if a["ann_id"] in ann_vlms:
                new_vlm = ann_vlms[a["ann_id"]]
                if new_vlm and isinstance(new_vlm, dict):
                    a["vlm"] = new_vlm
                    a["crop_mode"] = "crop_claude_fix"
                    modified = True
                    anns_fixed += 1

        if modified:
            with open(fpath, "w") as f:
                json.dump(data, f, indent=2)
            updated += 1

    print(f"Applied: {anns_fixed} annotations fixed across {updated} files")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["list", "apply"], required=True)
    parser.add_argument("--results", default=RESULTS_FILE)
    args = parser.parse_args()

    if args.mode == "apply":
        apply_results(args.results)
    else:
        # List mode: just show stats
        work = []
        for f in sorted(os.listdir(DESC_DIR)):
            if not f.endswith('.json'):
                continue
            img_id = int(f.replace('.json', ''))
            with open(os.path.join(DESC_DIR, f)) as fh:
                d = json.load(fh)
            for a in d.get('annotations', []):
                if isinstance(a, dict) and isinstance(a.get('vlm'), str):
                    work.append({"img_id": img_id, "ann_id": a["ann_id"],
                                 "cat_id": a.get("category_id")})
        print(f"Total annotations to fix: {len(work)}")
        print(f"Unique images: {len(set(w['img_id'] for w in work))}")
