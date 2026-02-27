#!/usr/bin/env python3
"""Visualize old/new pseudo weights on a sampled subset of images.

This script is designed for outputs like OW_COCO_R3_vlm.json where each annotation
contains both old and new weights (e.g. weight_old and weight).
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


def load_json_any(path):
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("annotations"), list):
        return data["annotations"]
    raise ValueError("Unsupported JSON format. Expected list or dict with 'annotations'.")


def build_image_map(coco_json_path, image_root):
    coco = json.loads(Path(coco_json_path).read_text())
    root = Path(image_root)
    by_id = {}
    for img in coco.get("images", []):
        file_name = img["file_name"]
        p1 = root / file_name
        p2 = root / Path(file_name).name
        by_id[int(img["id"])] = (p1, p2)
    return by_id


def color_from_weight(w):
    w = max(0.0, min(1.0, float(w)))
    r = int(255 * (1.0 - w))
    g = int(255 * w)
    b = 48
    return (r, g, b)


def draw_boxes(image, anns, weight_key, topk):
    draw = ImageDraw.Draw(image)
    anns = sorted(anns, key=lambda a: float(a.get(weight_key, 0.0)), reverse=True)[:topk]
    for ann in anns:
        x, y, w, h = ann["bbox"]
        score = float(ann.get(weight_key, 0.0))
        x1, y1, x2, y2 = x, y, x + w, y + h
        color = color_from_weight(score)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{score:.2f}"
        tx = max(0, int(x1))
        ty = max(0, int(y1) - 12)
        draw.rectangle([tx, ty, tx + 42, ty + 11], fill=(0, 0, 0))
        draw.text((tx + 2, ty), label, fill=color)
    return image


def image_stats(anns, old_key, new_key):
    old_vals = [float(a.get(old_key, 0.0)) for a in anns]
    new_vals = [float(a.get(new_key, 0.0)) for a in anns]
    if not old_vals:
        return {"old_mean": 0.0, "new_mean": 0.0, "delta_mean": 0.0, "improved": 0, "count": 0}
    improved = sum(1 for o, n in zip(old_vals, new_vals) if n > o)
    old_mean = sum(old_vals) / len(old_vals)
    new_mean = sum(new_vals) / len(new_vals)
    return {
        "old_mean": old_mean,
        "new_mean": new_mean,
        "delta_mean": new_mean - old_mean,
        "improved": improved,
        "count": len(old_vals),
    }


def make_canvas(old_img, new_img, image_id, stats):
    w, h = old_img.size
    title_h = 52
    gap = 12
    canvas = Image.new("RGB", (w * 2 + gap, h + title_h), color=(18, 18, 18))
    canvas.paste(old_img, (0, title_h))
    canvas.paste(new_img, (w + gap, title_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), f"image_id={image_id}  anns={stats['count']}  improved={stats['improved']}", fill=(230, 230, 230))
    draw.text((8, 24), f"OLD mean={stats['old_mean']:.4f}  delta={stats['delta_mean']:+.4f}", fill=(255, 180, 120))
    draw.text((w + gap + 8, 24), f"NEW mean={stats['new_mean']:.4f}", fill=(130, 240, 130))
    draw.text((8, title_h - 14), "OLD(top-k by old weight)", fill=(220, 220, 220))
    draw.text((w + gap + 8, title_h - 14), "NEW(top-k by new weight)", fill=(220, 220, 220))
    return canvas


def parse_args():
    p = argparse.ArgumentParser(description="Visualize pseudo weight changes (old vs new) on sampled images.")
    p.add_argument("--r3-json", required=True, help="R3 JSON after rerank (should contain old/new weights).")
    p.add_argument("--coco-json", required=True, help="COCO json for image_id -> file_name mapping.")
    p.add_argument("--image-root", required=True, help="Directory containing training images.")
    p.add_argument("--out-dir", required=True, help="Directory to write visualizations and summary.")
    p.add_argument("--old-weight-key", default="weight_old", help="Field name for old weight.")
    p.add_argument("--new-weight-key", default="weight", help="Field name for new weight.")
    p.add_argument("--topk-per-image", type=int, default=30, help="How many boxes to draw on each side.")
    p.add_argument("--max-images", type=int, default=20, help="How many images to visualize.")
    p.add_argument("--select-by", choices=["delta", "new", "random"], default="delta", help="Image sampling rule.")
    p.add_argument("--seed", type=int, default=3407, help="Random seed when select-by=random.")
    p.add_argument("--only-pseudo", action="store_true", help="Only use annotations with pseudo=1.")
    p.add_argument("--skip-missing-images", action="store_true", help="Skip missing images instead of failing.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    anns = load_json_any(args.r3_json)
    if args.only_pseudo:
        anns = [a for a in anns if int(a.get("pseudo", 0)) == 1]

    grouped = defaultdict(list)
    for ann in anns:
        if "image_id" not in ann or "bbox" not in ann:
            continue
        grouped[int(ann["image_id"])].append(ann)

    scored = []
    for image_id, image_anns in grouped.items():
        st = image_stats(image_anns, args.old_weight_key, args.new_weight_key)
        scored.append((image_id, st))

    if args.select_by == "delta":
        scored.sort(key=lambda x: x[1]["delta_mean"], reverse=True)
    elif args.select_by == "new":
        scored.sort(key=lambda x: x[1]["new_mean"], reverse=True)
    else:
        rng = random.Random(args.seed)
        rng.shuffle(scored)

    image_map = build_image_map(args.coco_json, args.image_root)

    summary = {
        "num_annotations": len(anns),
        "num_images_grouped": len(grouped),
        "num_images_selected": 0,
        "num_written": 0,
        "num_checked": 0,
        "num_missing_images": 0,
        "selected_by": args.select_by,
    }
    summary_rows = []

    rank = 0
    for image_id, st in scored:
        if summary["num_written"] >= args.max_images:
            break
        summary["num_checked"] += 1
        candidates = image_map.get(image_id)
        if candidates is None:
            summary["num_missing_images"] += 1
            if args.skip_missing_images:
                continue
            raise FileNotFoundError(f"Missing image mapping for image_id={image_id}")
        img_path = None
        for p in candidates:
            if p.exists():
                img_path = p
                break
        if img_path is None:
            summary["num_missing_images"] += 1
            if args.skip_missing_images:
                continue
            raise FileNotFoundError(f"Missing image for image_id={image_id}: tried {candidates[0]} and {candidates[1]}")

        base = Image.open(img_path).convert("RGB")
        image_anns = grouped[image_id]
        old_img = draw_boxes(base.copy(), image_anns, args.old_weight_key, args.topk_per_image)
        new_img = draw_boxes(base.copy(), image_anns, args.new_weight_key, args.topk_per_image)
        canvas = make_canvas(old_img, new_img, image_id, st)

        out_path = out_dir / f"{rank:03d}_{image_id}.jpg"
        canvas.save(out_path, quality=92)
        summary["num_written"] += 1
        summary["num_images_selected"] += 1
        summary_rows.append(
            {
                "rank": rank,
                "image_id": image_id,
                "old_mean": st["old_mean"],
                "new_mean": st["new_mean"],
                "delta_mean": st["delta_mean"],
                "improved": st["improved"],
                "count": st["count"],
                "file": out_path.name,
            }
        )
        rank += 1

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "rows.json").write_text(json.dumps(summary_rows, indent=2))

    print(f"Saved visualizations to: {out_dir}")
    print(json.dumps(summary, indent=2))
    if summary["num_written"] == 0:
        print("No visualization images were written. Check image-root/coco-json alignment and missing-image count.")


if __name__ == "__main__":
    main()
