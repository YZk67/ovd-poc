"""
Offline COCO-style eval on Objects365 v2 val.

Reproduces the AP table that detectron2 swallowed due to the
`KeyError: 'info'` bug in pycocotools, without re-running inference.

Usage:
    python offline_eval_obj365.py \
        --ann   /path/to/object365/annotations/zhiyuan_objv2_val.json \
        --pred  /root/autodl-tmp/transfer_lvis_to_obj365/coco_instances_results.json
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def patch_ann_inplace(ann_path: str) -> str:
    """Make a patched copy of the annotation JSON next to the original
    with `info` / `licenses` filled in. Returns the patched path."""
    with open(ann_path, "r") as f:
        data = json.load(f)

    changed = False
    if "info" not in data:
        data["info"] = {"description": "Objects365 v2", "version": "2.0"}
        changed = True
    if "licenses" not in data:
        data["licenses"] = []
        changed = True

    if not changed:
        return ann_path

    fd, patched = tempfile.mkstemp(
        prefix="obj365_val_patched_", suffix=".json",
        dir=os.path.dirname(ann_path),
    )
    os.close(fd)
    with open(patched, "w") as f:
        json.dump(data, f)
    print(f"[patch] wrote patched annotation -> {patched}", flush=True)
    return patched


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ann", required=True, help="Objects365 val annotation json")
    p.add_argument("--pred", required=True, help="coco_instances_results.json from inference")
    p.add_argument("--iou-type", default="bbox", choices=["bbox", "segm"])
    p.add_argument("--max-dets", type=int, nargs=3, default=[100, 300, 1000],
                   help="maxDets used by COCOeval (default mimics LVIS-style 100/300/1000)")
    args = p.parse_args()

    if not os.path.isfile(args.ann):
        sys.exit(f"annotation not found: {args.ann}")
    if not os.path.isfile(args.pred):
        sys.exit(f"predictions not found: {args.pred}")

    ann_path = patch_ann_inplace(args.ann)

    print(f"[load] GT  : {ann_path}", flush=True)
    coco_gt = COCO(ann_path)

    print(f"[load] PRED: {args.pred}", flush=True)
    with open(args.pred) as f:
        preds = json.load(f)
    if len(preds) == 0:
        sys.exit("predictions file is empty")
    print(f"[load] {len(preds)} prediction entries", flush=True)
    coco_dt = coco_gt.loadRes(preds)

    coco_eval = COCOeval(coco_gt, coco_dt, args.iou_type)
    coco_eval.params.maxDets = list(args.max_dets)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    names = ["AP", "AP50", "AP75", "APs", "APm", "APl",
             f"AR@{args.max_dets[0]}", f"AR@{args.max_dets[1]}", f"AR@{args.max_dets[2]}",
             "ARs", "ARm", "ARl"]
    print("\n=== Final results ===")
    for n, v in zip(names, stats):
        print(f"  {n:>8s} = {v*100:6.2f}")


if __name__ == "__main__":
    main()
