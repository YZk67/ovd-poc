#!/usr/bin/env python3
"""Offline rerank unknown proposals in an R3-style JSON with OpenCLIP.

Use CLIP as a box-level quality scorer:
  - s_obj : object-vs-background margin score
  - s_conf: multi-scale consistency score
  - w_new = clip(w_old * s_obj * s_conf, 0, 1)

Typical use:
python tools/rerank_unknowns_with_clip.py \
  --r3-json dataset/metadata/OW_COCO_R3.json \
  --coco-json dataset/coco/annotations/instances_train2017.json \
  --image-root dataset/coco/train2017 \
  --out-json dataset/metadata/OW_COCO_R3_vlm.json
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


POS_PROMPTS = [
    "a photo of an object",
    "a photo of a thing",
    "a photo of an item",
    "a photo of a foreground object",
]

NEG_PROMPTS = [
    "a photo of background",
    "a photo of texture",
    "a photo of pattern",
    "a photo of clutter",
]


def coco_xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return [float(x), float(y), float(x + w), float(y + h)]


def expand_xyxy(xyxy, scale, width, height):
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    half_w = bw * scale / 2.0
    half_h = bh * scale / 2.0
    nx1 = max(0.0, cx - half_w)
    ny1 = max(0.0, cy - half_h)
    nx2 = min(float(width - 1), cx + half_w)
    ny2 = min(float(height - 1), cy + half_h)
    if nx2 <= nx1 + 1:
        nx2 = min(float(width - 1), nx1 + 2)
    if ny2 <= ny1 + 1:
        ny2 = min(float(height - 1), ny1 + 2)
    return [nx1, ny1, nx2, ny2]


def crop_pil(img, xyxy):
    x1, y1, x2, y2 = map(int, map(round, xyxy))
    return img.crop((x1, y1, x2, y2))


class OpenCLIPScorer:
    def __init__(self, model_name, pretrained, device):
        try:
            import torch
            import torch.nn.functional as F
            import open_clip
        except ImportError as e:
            raise ImportError(
                "Missing dependencies. Install with: pip install open_clip_torch torch torchvision"
            ) from e

        self.torch = torch
        self.F = F
        self.open_clip = open_clip
        self.device = device if torch.cuda.is_available() and device != "cpu" else "cpu"

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=model_name, pretrained=pretrained
        )
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess

        tokenizer = open_clip.get_tokenizer(model_name)
        with torch.no_grad():
            pos_tokens = tokenizer(POS_PROMPTS).to(self.device)
            neg_tokens = tokenizer(NEG_PROMPTS).to(self.device)
            pos_text = self.model.encode_text(pos_tokens)
            neg_text = self.model.encode_text(neg_tokens)
            self.pos_text = F.normalize(pos_text, dim=-1)
            self.neg_text = F.normalize(neg_text, dim=-1)

    def score_crops(self, crops, tau=0.07, batch_size=64):
        torch = self.torch
        F = self.F
        if not crops:
            return np.zeros((0,), dtype=np.float32)

        outs = []
        with torch.no_grad():
            for i in range(0, len(crops), batch_size):
                batch = crops[i : i + batch_size]
                imgs = torch.stack([self.preprocess(c) for c in batch], dim=0).to(self.device)
                img_feat = self.model.encode_image(imgs)
                img_feat = F.normalize(img_feat, dim=-1)
                pos_sim = img_feat @ self.pos_text.T
                neg_sim = img_feat @ self.neg_text.T
                margin = pos_sim.max(dim=1).values - neg_sim.max(dim=1).values
                s_obj = torch.sigmoid(margin / tau)
                outs.append(s_obj.detach().cpu().numpy())
        return np.concatenate(outs, axis=0).astype(np.float32)


def compute_scores_for_boxes(
    img,
    boxes_xywh,
    scorer,
    scales,
    tau,
    gamma,
    batch_size,
):
    width, height = img.size
    sobj_per_scale = []
    for scale in scales:
        crops = []
        for box in boxes_xywh:
            xyxy = coco_xywh_to_xyxy(box)
            xyxy = expand_xyxy(xyxy, scale, width, height)
            crops.append(crop_pil(img, xyxy))
        sobj_per_scale.append(scorer.score_crops(crops, tau=tau, batch_size=batch_size))

    sobj_stack = np.stack(sobj_per_scale, axis=1)  # [N, K]
    s_obj = sobj_stack.max(axis=1)
    std = sobj_stack.std(axis=1)
    s_conf = np.exp(-std / max(gamma, 1e-8))
    return s_obj.astype(np.float32), s_conf.astype(np.float32)


def load_json_any(path):
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return data, None
    if isinstance(data, dict):
        if "annotations" in data and isinstance(data["annotations"], list):
            return data["annotations"], data
    raise ValueError("Unsupported JSON format. Expected list or dict with 'annotations'.")


def build_image_map(coco_json_path, image_root):
    coco = json.loads(Path(coco_json_path).read_text())
    root = Path(image_root)
    by_id = {}
    for img in coco.get("images", []):
        by_id[int(img["id"])] = root / img["file_name"]
    return by_id


def parse_scales(s):
    return tuple(float(x) for x in s.split(",") if x.strip())


def parse_args():
    p = argparse.ArgumentParser(description="Rerank unknown proposals in R3 JSON using OpenCLIP.")
    p.add_argument("--r3-json", required=True, help="Input R3 proposals JSON (list or dict with annotations).")
    p.add_argument("--coco-json", required=True, help="COCO json that provides image_id -> file_name mapping.")
    p.add_argument("--image-root", required=True, help="Directory containing the images (e.g. coco/train2017).")
    p.add_argument("--out-json", required=True, help="Output reranked JSON path.")
    p.add_argument("--model-name", default="ViT-L-14", help="OpenCLIP model name.")
    p.add_argument("--pretrained", default="openai", help="OpenCLIP pretrained tag.")
    p.add_argument("--device", default="cuda", help="cuda/cpu")
    p.add_argument("--batch-size", type=int, default=64, help="CLIP image batch size.")
    p.add_argument("--scales", default="1.0,1.3,1.6", help="Crop scales, comma-separated.")
    p.add_argument("--tau", type=float, default=0.07, help="Temperature for margin sigmoid.")
    p.add_argument("--gamma", type=float, default=0.10, help="Gamma for consistency score exp(-std/gamma).")
    p.add_argument("--weight-key", default="weight", help="Original weight field in R3 JSON.")
    p.add_argument("--fallback-weight", type=float, default=1.0, help="Fallback w_old if weight field missing.")
    p.add_argument("--obj-thr", type=float, default=0.0, help="Drop if s_obj < obj-thr.")
    p.add_argument("--weight-thr", type=float, default=0.0, help="Drop if w_new < weight-thr.")
    p.add_argument("--topk-per-image", type=int, default=0, help="Keep top-K per image after rerank (0 disables).")
    p.add_argument("--keep-all-fields", action="store_true", help="Keep all original fields (default true behavior).")
    p.add_argument("--skip-missing-images", action="store_true", help="Skip missing images instead of raising.")
    p.add_argument("--save-meta", action="store_true", help="Write rerank settings under output['info'].")
    return p.parse_args()


def main():
    args = parse_args()
    scales = parse_scales(args.scales)
    if not scales:
        raise ValueError("--scales must contain at least one value")

    anns, container = load_json_any(args.r3_json)
    image_map = build_image_map(args.coco_json, args.image_root)
    scorer = OpenCLIPScorer(args.model_name, args.pretrained, args.device)

    grouped = defaultdict(list)
    for idx, ann in enumerate(anns):
        if "image_id" not in ann or "bbox" not in ann:
            continue
        grouped[int(ann["image_id"])].append((idx, ann))

    kept_indices = set()
    stats = {
        "num_input": len(anns),
        "num_images_grouped": len(grouped),
        "num_missing_images": 0,
        "num_processed_images": 0,
        "num_kept": 0,
        "num_dropped_obj_thr": 0,
        "num_dropped_weight_thr": 0,
        "num_dropped_topk": 0,
    }

    for image_id, items in grouped.items():
        img_path = image_map.get(image_id)
        if img_path is None or not img_path.exists():
            stats["num_missing_images"] += 1
            if args.skip_missing_images:
                continue
            raise FileNotFoundError(f"Missing image for image_id={image_id}: {img_path}")

        img = Image.open(img_path).convert("RGB")
        boxes = [ann["bbox"] for _, ann in items]
        s_obj, s_conf = compute_scores_for_boxes(
            img=img,
            boxes_xywh=boxes,
            scorer=scorer,
            scales=scales,
            tau=args.tau,
            gamma=args.gamma,
            batch_size=args.batch_size,
        )
        stats["num_processed_images"] += 1

        reranked = []
        for local_i, (ann_idx, ann) in enumerate(items):
            w_old = float(ann.get(args.weight_key, args.fallback_weight))
            w_new = float(np.clip(w_old * float(s_obj[local_i]) * float(s_conf[local_i]), 0.0, 1.0))

            ann["weight_old"] = w_old
            ann["vlm_s_obj"] = float(s_obj[local_i])
            ann["vlm_s_conf"] = float(s_conf[local_i])
            ann["weight"] = w_new

            if ann.get("score") is not None:
                try:
                    ann["score_old"] = float(ann["score"])
                    ann["score"] = w_new
                except Exception:
                    pass

            drop = False
            if float(s_obj[local_i]) < args.obj_thr:
                stats["num_dropped_obj_thr"] += 1
                drop = True
            if w_new < args.weight_thr:
                stats["num_dropped_weight_thr"] += 1
                drop = True
            if not drop:
                reranked.append((ann_idx, w_new))

        if args.topk_per_image > 0 and len(reranked) > args.topk_per_image:
            reranked.sort(key=lambda x: x[1], reverse=True)
            keep_now = reranked[: args.topk_per_image]
            stats["num_dropped_topk"] += len(reranked) - len(keep_now)
            reranked = keep_now

        kept_indices.update(idx for idx, _ in reranked)

    out_anns = [ann for i, ann in enumerate(anns) if i in kept_indices]
    stats["num_kept"] = len(out_anns)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if container is None:
        out_obj = out_anns
    else:
        out_obj = dict(container)
        out_obj["annotations"] = out_anns
        if args.save_meta:
            info = dict(out_obj.get("info", {}))
            info["vlm_unknown_rerank"] = {
                "model_name": args.model_name,
                "pretrained": args.pretrained,
                "scales": list(scales),
                "tau": args.tau,
                "gamma": args.gamma,
                "obj_thr": args.obj_thr,
                "weight_thr": args.weight_thr,
                "topk_per_image": args.topk_per_image,
                "weight_key_in": args.weight_key,
                "weight_key_out": "weight",
            }
            out_obj["info"] = info

    out_path.write_text(json.dumps(out_obj))

    print(f"Saved: {out_path}")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
