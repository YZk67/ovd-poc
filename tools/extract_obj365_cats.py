#!/usr/bin/env python3
"""
Extract Objects365 v2 category metadata from val annotations.

After downloading zhiyuan_objv2_val.json, run this script to produce:
  - dataset/object365/annotations/obj_cats.json
      Used by detectron2's _get_obj365_instances_meta() at registration time.
  - dataset/metadata/obj_365_classes.json
      Ordered class-name list (used by transfer config).

Usage:
    python tools/extract_obj365_cats.py
    # or with explicit paths:
    python tools/extract_obj365_cats.py \
        --val-json dataset/object365/annotations/zhiyuan_objv2_val.json \
        --cats-out dataset/object365/annotations/obj_cats.json \
        --names-out dataset/metadata/obj_365_classes.json
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val-json",
        default="dataset/object365/annotations/zhiyuan_objv2_val.json",
    )
    parser.add_argument(
        "--cats-out",
        default="dataset/object365/annotations/obj_cats.json",
    )
    parser.add_argument(
        "--names-out",
        default="dataset/metadata/obj_365_classes.json",
    )
    args = parser.parse_args()

    print(f"Loading {args.val_json} ...")
    with open(args.val_json) as f:
        ann = json.load(f)

    cats = sorted(ann["categories"], key=lambda x: x["id"])
    assert len(cats) == 365, f"expected 365 categories, got {len(cats)}"

    obj_cats = [{"id": c["id"], "name": c["name"]} for c in cats]
    names = [c["name"] for c in cats]

    Path(args.cats_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.cats_out, "w") as f:
        json.dump(obj_cats, f, indent=2)

    Path(args.names_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.names_out, "w") as f:
        json.dump(names, f, indent=2)

    print(f"Saved {len(obj_cats)} categories -> {args.cats_out}")
    print(f"Saved {len(names)} names      -> {args.names_out}")
    print(f"\nFirst 5: {names[:5]}")
    print(f"Last  5: {names[-5:]}")


if __name__ == "__main__":
    main()
