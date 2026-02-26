#!/usr/bin/env python3
"""Generate pseudo-unknown annotations from external detector outputs.

Paper-aligned intent (OV-DQUO style):
1) Use an external detector (the paper uses OLN) to obtain unknown proposals (o_i, s_i).
2) Optionally use a foreground estimator score as pseudo weight (w_i).
3) Filter proposals (score threshold, NMS, overlap with known GT).
4) Merge pseudo annotations into COCO JSON with extra fields:
   - score        ~ s_i
   - is_pseudo    (bool/int flag)
   - pseudo_weight~ w_i (if available, else defaults to score)

This script does not run OLN itself. It consumes detector outputs already exported to JSON.
"""

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def area_xywh(box):
    return max(0.0, float(box[2])) * max(0.0, float(box[3]))


def iou_xywh(a, b):
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = area_xywh(a) + area_xywh(b) - inter
    return inter / max(union, 1e-12)


def nms_xywh(dets, iou_thr):
    # dets: list[dict] with "bbox", "score"
    dets = sorted(dets, key=lambda d: float(d.get("score", 0.0)), reverse=True)
    keep = []
    for det in dets:
        if all(iou_xywh(det["bbox"], k["bbox"]) <= iou_thr for k in keep):
            keep.append(det)
    return keep


def load_detector_results(path):
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        dets = data
    elif isinstance(data, dict) and "annotations" in data:
        dets = data["annotations"]
    else:
        raise ValueError("Unsupported detector json format. Expected list or dict with 'annotations'.")
    by_image = defaultdict(list)
    for det in dets:
        if "image_id" not in det or "bbox" not in det:
            continue
        by_image[int(det["image_id"])].append(det)
    return by_image


def parse_args():
    p = argparse.ArgumentParser(description="Generate pseudo unknown annotations from external detections.")
    p.add_argument("--gt-json", required=True, help="COCO train annotation json.")
    p.add_argument("--det-json", required=True, help="External detector outputs in COCO detections json.")
    p.add_argument("--out-json", required=True, help="Merged output json path.")
    p.add_argument(
        "--detector-name",
        default="oln",
        help="Metadata only. Paper uses 'oln'. Stored in output['info'].",
    )
    p.add_argument("--score-thr", type=float, default=0.3, help="Min detector score s_i.")
    p.add_argument("--nms-iou-thr", type=float, default=0.5, help="Class-agnostic NMS IoU threshold.")
    p.add_argument(
        "--known-gt-iou-thr",
        type=float,
        default=0.5,
        help="Reject proposal if overlap with any known GT exceeds this threshold.",
    )
    p.add_argument("--min-box-area", type=float, default=16.0, help="Reject tiny boxes by area in pixels.")
    p.add_argument("--max-per-image", type=int, default=50, help="Max pseudo proposals per image after filtering.")
    p.add_argument(
        "--pseudo-category-id",
        type=int,
        default=-1,
        help="Category id for class-agnostic proposals when detector result has no category_id.",
    )
    p.add_argument(
        "--keep-det-category",
        action="store_true",
        help="Use detector category_id if available (class-aware detector outputs).",
    )
    p.add_argument(
        "--weight-key",
        default="fg_score",
        help="Optional detector field to use as pseudo_weight w_i (fallback to score).",
    )
    p.add_argument(
        "--mark-existing-gt-score",
        action="store_true",
        help="Add score=1.0 to existing GT annotations for uniform downstream parsing.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    gt = json.loads(Path(args.gt_json).read_text())
    dets_by_image = load_detector_results(args.det_json)

    gt = copy.deepcopy(gt)
    anns = gt.get("annotations", [])
    gt_by_image = defaultdict(list)
    max_ann_id = 0
    for ann in anns:
        gt_by_image[int(ann["image_id"])].append(ann)
        max_ann_id = max(max_ann_id, int(ann.get("id", 0)))
        if args.mark_existing_gt_score and "score" not in ann:
            ann["score"] = 1.0
            ann["is_pseudo"] = 0

    next_ann_id = max_ann_id + 1
    added = 0
    kept_after_score = 0
    kept_after_nms = 0
    kept_after_gt_iou = 0

    for image in gt.get("images", []):
        image_id = int(image["id"])
        candidates = dets_by_image.get(image_id, [])
        if not candidates:
            continue

        # 1) score threshold on s_i
        cands = []
        for det in candidates:
            score = float(det.get("score", det.get("weight", 0.0)))
            bbox = det.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue
            if score < args.score_thr:
                continue
            if area_xywh(bbox) < args.min_box_area:
                continue
            cands.append(det)
        kept_after_score += len(cands)
        if not cands:
            continue

        # 2) class-agnostic NMS (OLN-style proposals are class-agnostic)
        cands = nms_xywh(cands, args.nms_iou_thr)
        kept_after_nms += len(cands)

        # 3) remove proposals that heavily overlap known GT boxes
        known_boxes = [ann["bbox"] for ann in gt_by_image.get(image_id, []) if not ann.get("is_pseudo", 0)]
        filtered = []
        for det in cands:
            max_iou = 0.0
            for gt_box in known_boxes:
                max_iou = max(max_iou, iou_xywh(det["bbox"], gt_box))
                if max_iou > args.known_gt_iou_thr:
                    break
            if max_iou <= args.known_gt_iou_thr:
                det = copy.deepcopy(det)
                det["_max_known_gt_iou"] = max_iou
                filtered.append(det)
        kept_after_gt_iou += len(filtered)

        # 4) top-k per image
        filtered = sorted(filtered, key=lambda d: float(d.get("score", 0.0)), reverse=True)[: args.max_per_image]

        for det in filtered:
            if args.keep_det_category and "category_id" in det:
                category_id = int(det["category_id"])
            else:
                category_id = int(det.get("category_id", args.pseudo_category_id))

            score = float(det.get("score", det.get("weight", 0.0)))  # s_i
            weight = float(det.get(args.weight_key, score))  # w_i (fallback to s_i)

            pseudo_ann = {
                "id": next_ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [float(x) for x in det["bbox"]],
                "area": area_xywh(det["bbox"]),
                "iscrowd": 0,
                "segmentation": [],
                "score": score,
                "pseudo_weight": weight,
                "is_pseudo": 1,
                "unknown": 1,
                "source_detector": args.detector_name,
                "max_known_gt_iou": float(det["_max_known_gt_iou"]),
            }
            anns.append(pseudo_ann)
            gt_by_image[image_id].append(pseudo_ann)
            next_ann_id += 1
            added += 1

    gt["annotations"] = anns
    info = gt.get("info", {})
    info["pseudo_unknown_generator"] = {
        "external_detector": args.detector_name,
        "score_thr": args.score_thr,
        "nms_iou_thr": args.nms_iou_thr,
        "known_gt_iou_thr": args.known_gt_iou_thr,
        "min_box_area": args.min_box_area,
        "max_per_image": args.max_per_image,
        "weight_key": args.weight_key,
        "paper_note": "OV-DQUO uses OLN as external detector and FE for foreground weighting.",
    }
    gt["info"] = info

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(gt))

    print(f"Saved: {out_path}")
    print(f"Added pseudo annotations: {added}")
    print(f"Candidates kept after score filter: {kept_after_score}")
    print(f"Candidates kept after NMS: {kept_after_nms}")
    print(f"Candidates kept after GT IoU filter (before top-k): {kept_after_gt_iou}")


if __name__ == "__main__":
    main()
