#!/usr/bin/env python3
"""Train a tiny score calibrator on held-out pseudo-novel LVIS classes."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def box_iou(box, boxes):
    if not boxes:
        return []
    x1, y1, x2, y2 = box
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    out = []
    for gx1, gy1, gx2, gy2 in boxes:
        ix1, iy1 = max(x1, gx1), max(y1, gy1)
        ix2, iy2 = min(x2, gx2), min(y2, gy2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        garea = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
        union = area + garea - inter
        out.append(inter / union if union > 0 else 0.0)
    return out


def build_category_maps(cat_info):
    cats = sorted(cat_info, key=lambda x: x["id"])
    cat_id_by_idx = {idx: cat["id"] for idx, cat in enumerate(cats)}
    freq_by_cat_id = {cat["id"]: float(cat.get("image_count", 0.0)) for cat in cats}
    bucket_by_cat_id = {cat["id"]: cat.get("frequency", "") for cat in cats}
    return cat_id_by_idx, freq_by_cat_id, bucket_by_cat_id


def build_gt(lvis_gt):
    images = {img["id"]: img for img in lvis_gt["images"]}
    gt_by_key = defaultdict(list)
    for ann in lvis_gt["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        key = (ann["image_id"], ann["category_id"])
        gt_by_key[key].append(xywh_to_xyxy(ann["bbox"]))
    return images, gt_by_key


def label_predictions(preds, train_cat_ids, gt_by_key, iou_thr):
    grouped = defaultdict(list)
    for idx, pred in enumerate(preds):
        if pred["category_id"] in train_cat_ids:
            grouped[(pred["image_id"], pred["category_id"])].append((idx, pred))

    labels = {}
    for key, items in grouped.items():
        gt_boxes = gt_by_key.get(key, [])
        used = [False] * len(gt_boxes)
        for idx, pred in sorted(items, key=lambda x: x[1]["score"], reverse=True):
            ious = box_iou(xywh_to_xyxy(pred["bbox"]), gt_boxes)
            best_iou = max(ious) if ious else 0.0
            best_idx = ious.index(best_iou) if ious else -1
            is_pos = best_iou >= iou_thr and best_idx >= 0 and not used[best_idx]
            if is_pos:
                used[best_idx] = True
            labels[idx] = 1.0 if is_pos else 0.0
    return labels


def feature_vector(pred, images, freq_by_cat_id):
    score = min(max(float(pred["score"]), 1e-6), 1.0 - 1e-6)
    logit_score = math.log(score / (1.0 - score))
    image = images.get(pred["image_id"], {})
    width = max(float(image.get("width", 1.0)), 1.0)
    height = max(float(image.get("height", 1.0)), 1.0)
    _, _, bw, bh = pred["bbox"]
    area_frac = max(float(bw) * float(bh), 0.0) / (width * height)
    class_freq = math.log1p(freq_by_cat_id.get(pred["category_id"], 0.0))
    return [logit_score, class_freq, math.sqrt(max(area_frac, 0.0))]


class Calibrator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gt", default="dataset/lvis/lvis_v1_val.json")
    parser.add_argument("--cat-info", default="dataset/lvis/lvis_v1_train_norare_cat_info.json")
    parser.add_argument("--pseudo-split", default="dataset/lvis/pseudo_novel_base100_seed42.json")
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--apply", choices=["all", "pseudo", "rare"], default="rare")
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    preds = load_json(args.predictions)
    lvis_gt = load_json(args.gt)
    cat_info = load_json(args.cat_info)
    pseudo_split = load_json(args.pseudo_split)

    cat_id_by_idx, freq_by_cat_id, bucket_by_cat_id = build_category_maps(cat_info)
    pseudo_cat_ids = {cat_id_by_idx[idx] for idx in pseudo_split["class_ids"]}
    rare_cat_ids = {cat_id for cat_id, bucket in bucket_by_cat_id.items() if bucket == "r"}
    apply_cat_ids = {
        "all": None,
        "pseudo": pseudo_cat_ids,
        "rare": rare_cat_ids,
    }[args.apply]

    images, gt_by_key = build_gt(lvis_gt)
    labels_by_pred = label_predictions(preds, pseudo_cat_ids, gt_by_key, args.iou_thr)
    train_indices = sorted(labels_by_pred)
    if not train_indices:
        raise RuntimeError("No pseudo-novel predictions found for calibrator training.")

    x = torch.tensor(
        [feature_vector(preds[idx], images, freq_by_cat_id) for idx in train_indices],
        dtype=torch.float32,
    )
    y = torch.tensor([labels_by_pred[idx] for idx in train_indices], dtype=torch.float32)
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-6)
    x = (x - mean) / std

    model = Calibrator(x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos_weight = ((y.numel() - y.sum()) / y.sum().clamp_min(1.0)).clamp_min(1.0)

    for epoch in range(args.epochs):
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch == 0 or (epoch + 1) % 50 == 0:
            print(f"epoch={epoch + 1:04d} loss={float(loss.item()):.6f}")

    calibrated = []
    model.eval()
    with torch.no_grad():
        for pred in preds:
            new_pred = dict(pred)
            if apply_cat_ids is None or pred["category_id"] in apply_cat_ids:
                feat = torch.tensor(feature_vector(pred, images, freq_by_cat_id), dtype=torch.float32)
                feat = (feat - mean) / std
                new_pred["score"] = float(torch.sigmoid(model(feat.unsqueeze(0))).item())
            calibrated.append(new_pred)

    Path(args.output_predictions).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_predictions, "w") as f:
        json.dump(calibrated, f)

    Path(args.output_model).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "feature_names": ["logit_score", "log_class_freq", "sqrt_box_area_frac"],
            "pseudo_cat_ids": sorted(pseudo_cat_ids),
            "apply": args.apply,
            "iou_thr": args.iou_thr,
        },
        args.output_model,
    )
    print(
        f"trained on {len(train_indices)} detections "
        f"(positives={int(y.sum().item())}); wrote {args.output_predictions}"
    )


if __name__ == "__main__":
    main()
