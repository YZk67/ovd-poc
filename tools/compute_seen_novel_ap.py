#!/usr/bin/env python3
import argparse
import json
import re

import numpy as np


def load_list(path):
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def parse_ap_table(text):
    ap = {}
    for line in text.splitlines():
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if not parts or parts[0].lower() in {"category", "ap", ":-----------"}:
            continue
        # parts are like: [cat, ap, cat, ap, ...]
        for i in range(0, len(parts) - 1, 2):
            cat = parts[i]
            val = parts[i + 1]
            if not cat or not val:
                continue
            if not re.match(r"^-?\d+(\.\d+)?$", val):
                continue
            ap[cat] = float(val)
    return ap


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def ap_from_coco_results(gt_json, results_json, iou_type="bbox"):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(results_json)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()

    cat_ids = coco_gt.getCatIds()
    cats = coco_gt.loadCats(cat_ids)
    cat_names = [c["name"] for c in cats]

    precision = coco_eval.eval["precision"]  # T x R x K x A x M
    ap = {}
    for k, name in enumerate(cat_names):
        p = precision[:, :, k, 0, 2]  # area=all, maxDets=100
        p = p[p > -1]
        ap[name] = float(np.mean(p)) if p.size else 0.0
    return ap


def main():
    parser = argparse.ArgumentParser(description="Compute seen/novel mean AP.")
    parser.add_argument("--ap-file", help="Text file containing per-class AP table.")
    parser.add_argument("--coco-results", help="COCO detections json (e.g., coco_instances_results.json).")
    parser.add_argument("--gt-json", help="COCO ground truth json for evaluation.")
    parser.add_argument("--iou-type", default="bbox", choices=["bbox", "segm"], help="COCOeval iou type.")
    parser.add_argument("--seen-classes", required=True, help="JSON list of seen classes.")
    parser.add_argument("--all-classes", required=True, help="JSON list of all classes.")
    args = parser.parse_args()

    if args.ap_file:
        with open(args.ap_file, "r") as f:
            text = f.read()
        ap = parse_ap_table(text)
    else:
        if not args.coco_results or not args.gt_json:
            raise SystemExit("Provide --ap-file or both --coco-results and --gt-json")
        ap = ap_from_coco_results(args.gt_json, args.coco_results, args.iou_type)

    seen = load_list(args.seen_classes)
    all_classes = load_list(args.all_classes)
    novel = [c for c in all_classes if c not in seen]

    seen_vals = [ap[c] for c in seen if c in ap]
    novel_vals = [ap[c] for c in novel if c in ap]
    missing = [c for c in all_classes if c not in ap]

    print(f"seen mean AP:  {mean(seen_vals):.3f} ({len(seen_vals)}/{len(seen)})")
    print(f"novel mean AP: {mean(novel_vals):.3f} ({len(novel_vals)}/{len(novel)})")
    if missing:
        print(f"missing classes in table ({len(missing)}): {missing}")


if __name__ == "__main__":
    main()
