"""Generate VLM soft targets using local Qwen2-VL-7B with crop mode.

Shares the same output format and descriptions directory as the OpenAI script,
so all backends (GPT-4o-mini, Qwen2-VL, Claude) can interleave freely.
Already-processed images (by ANY backend) are automatically skipped.

Usage:
    # Run all remaining images
    python tools/vlm_generate_crop_qwen.py \
        --image-dir /home/yi/detr-attention/data/coco/train2017 \
        --output-dir dataset/vlm_targets_crop

    # Run specific range
    python tools/vlm_generate_crop_qwen.py \
        --image-dir /home/yi/detr-attention/data/coco/train2017 \
        --output-dir dataset/vlm_targets_crop \
        --start 0 --num-images 1000

    # Use a different model size
    python tools/vlm_generate_crop_qwen.py \
        --image-dir /home/yi/detr-attention/data/coco/train2017 \
        --output-dir dataset/vlm_targets_crop \
        --model Qwen/Qwen2-VL-2B-Instruct

    # After ALL backends finish, convert to training tensors:
    python tools/vlm_generate_soft_targets.py \
        --mode convert \
        --descriptions-dir dataset/vlm_targets_crop/descriptions \
        --output-dir dataset/vlm_targets_crop \
        --clip-model convnext
"""

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageDraw

ALL_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "kite", "skateboard", "surfboard", "bottle", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "pizza", "donut", "cake", "chair", "couch",
    "bed", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors", "toothbrush"
]
CLASS_LIST_STR = ", ".join(ALL_CLASSES)
ALLOWED_CLASS_SET = set(ALL_CLASSES)

PROMPT_CROP = f"""What is the main object in this cropped image region?

Focus on the object this crop was made for, not the whole scene. Ignore larger surrounding objects if they are only context.

CRITICAL: You MUST only use class names from this exact list. Do NOT use synonyms or subcategories.
Use "person" (not child/woman/man), "handbag" (not bag/purse), "bowl" (not plate/dish), "book" (not notebook/magazine).

Allowed classes: {CLASS_LIST_STR}

Answer in JSON only:
{{"best_class":"<class from list or null>","confidence":<0-1>,"top5_similar":{{"<class1>":<score>,...}},"attributes":["<a1>",...],"description":"<one sentence>"}}

Rules: best_class and top5_similar keys MUST be from the allowed list. top5=5 classes, scores 0-1, and should describe the target object in the crop. attributes=color,shape,size,material,state."""

PROMPT_CROP_BOX = f"""Identify the object inside the single red target box in this crop.

The red box marks the target object. Ignore surrounding objects, people, and scene context unless they are inside the red box.
Classify the boxed object itself, not the most salient thing in the crop and not the whole scene.

CRITICAL: You MUST only use class names from this exact list. Do NOT use synonyms or subcategories.
Use "person" (not child/woman/man), "handbag" (not bag/purse), "bowl" (not plate/dish), "book" (not notebook/magazine).

Allowed classes: {CLASS_LIST_STR}

Answer in JSON only:
{{"best_class":"<class from list or null>","confidence":<0-1>,"top5_similar":{{"<class1>":<score>,...}},"attributes":["<a1>",...],"description":"<one sentence>"}}

Rules: best_class and top5_similar keys MUST be from the allowed list. top5=5 classes, scores 0-1, and should describe the boxed object only. attributes=color,shape,size,material,state."""

PROMPT_FULL_HEADER = f"""For each object identified by the numbered red box and bbox [x,y,x2,y2], answer in a JSON array.

CRITICAL: You MUST only use class names from this exact list. Do NOT use synonyms or subcategories.
Use "person" (not child/woman/man), "handbag" (not bag/purse), "bowl" (not plate/dish), "book" (not notebook/magazine).

Allowed classes: {CLASS_LIST_STR}

Each entry: {{"best_class":"<class from list or null>","confidence":<0-1>,"top5_similar":{{"<class1>":<score>,...}},"attributes":["<a1>",...],"description":"<one sentence>"}}
Rules: best_class and top5_similar keys MUST be from the allowed list. Return JSON array, one per object."""

SMALL_AREA_THRESH = 4096
CROP_MARGIN = 0.5
QWEN_DEFAULT_MAX_PIXELS = 28 * 28 * 1280
QWEN_DEFAULT_CROP_MAX_PIXELS = 28 * 28 * 768


def parse_args():
    parser = argparse.ArgumentParser(description="Generate VLM soft targets with Qwen2-VL-7B crop mode")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--annotation", default="/home/yi/ovd-poc/ovd_ins_train2017_b.json")
    parser.add_argument("--output-dir", default="dataset/vlm_targets_crop")
    parser.add_argument("--image-id", type=int, default=None,
                        help="Restrict processing to a single image ID")
    parser.add_argument("--ann-id", type=int, default=None,
                        help="Restrict processing to a single annotation ID")
    parser.add_argument("--start", type=int, default=0, help="Start image index")
    parser.add_argument("--num-images", type=int, default=0, help="0 = all images")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--small-area-thresh", type=int, default=SMALL_AREA_THRESH)
    parser.add_argument("--crop-margin", type=float, default=CROP_MARGIN)
    parser.add_argument("--max-full-objects", type=int, default=15,
                        help="Max objects per full-image call (local model has less capacity)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for crop inference (increase if GPU memory allows)")
    parser.add_argument("--crop-max-pixels", type=int, default=QWEN_DEFAULT_CROP_MAX_PIXELS,
                        help="Max pixel budget per crop image sent to Qwen")
    parser.add_argument("--full-max-pixels", type=int, default=QWEN_DEFAULT_MAX_PIXELS,
                        help="Max pixel budget per full image sent to Qwen")
    parser.add_argument("--full-min-pixels", type=int, default=None,
                        help="Optional min pixel budget for full images")
    parser.add_argument("--only", default="both", choices=["crop", "full", "both"],
                        help="Only process crop (small) or full (large) objects")
    parser.add_argument("--no-box-overlay", action="store_true",
                        help="Disable drawing numbered boxes for full-image mode")
    parser.add_argument("--crop-box-overlay", action="store_true",
                        help="Draw a thin red target box on crop images")
    parser.add_argument("--crop-box-prompt", action="store_true",
                        help="Use the stronger boxed-object crop prompt (intended to pair with --crop-box-overlay)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run selected image/annotation even if an existing valid result is present")
    parser.add_argument("--debug-save-failures", action="store_true",
                        help="Save failed crop responses, metadata, and crop images for debugging")
    parser.add_argument("--debug-dir", default=None,
                        help="Directory for failed crop debug artifacts (default: <output-dir>/debug_failures)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    return parser.parse_args()


def canonicalize_class_name(name):
    if name is None:
        return None
    normalized = str(name).strip().lower()
    return normalized if normalized in ALLOWED_CLASS_SET else None


def clamp_confidence(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return max(0.0, min(1.0, value))


def normalize_top5_similar(top5, best_class=None, confidence=0.0):
    scores = {}
    if isinstance(top5, dict):
        for cls_name, score in top5.items():
            cls_name = canonicalize_class_name(cls_name)
            if cls_name is None:
                continue
            try:
                score = float(score)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score) or score <= 0:
                continue
            scores[cls_name] = max(scores.get(cls_name, 0.0), score)

    if best_class is not None:
        if confidence > 0:
            scores[best_class] = max(scores.get(best_class, 0.0), confidence)
        elif not scores:
            scores[best_class] = 1.0

    if not scores:
        return {}

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:5]
    total = sum(score for _, score in ranked)
    if total <= 0:
        return {}
    return {cls_name: score / total for cls_name, score in ranked}


def sanitize_vlm_output(vlm):
    if not isinstance(vlm, dict):
        return None

    best_class = canonicalize_class_name(vlm.get("best_class"))
    confidence = clamp_confidence(vlm.get("confidence", 0.0))
    top5 = normalize_top5_similar(vlm.get("top5_similar", {}), best_class, confidence)

    attributes = vlm.get("attributes", [])
    if not isinstance(attributes, list):
        attributes = [attributes]
    attributes = [str(attr).strip() for attr in attributes if str(attr).strip()]

    description = str(vlm.get("description", "")).strip()

    cleaned = dict(vlm)
    cleaned["best_class"] = best_class
    cleaned["confidence"] = confidence
    cleaned["top5_similar"] = top5
    cleaned["attributes"] = attributes
    cleaned["description"] = description
    return cleaned


def compute_crop_window(image, bbox, margin_ratio):
    x, y, w, h = bbox
    img_w, img_h = image.size
    margin = max(w, h) * margin_ratio
    x1 = max(0, int(x - margin))
    y1 = max(0, int(y - margin))
    x2 = min(img_w, int(x + w + margin))
    y2 = min(img_h, int(y + h + margin))
    return x1, y1, x2, y2


def draw_crop_target_box(crop_image, rel_bbox):
    overlay = crop_image.copy()
    draw = ImageDraw.Draw(overlay)
    x1, y1, x2, y2 = [int(round(v)) for v in rel_bbox]
    x1 = max(0, min(overlay.size[0] - 1, x1))
    y1 = max(0, min(overlay.size[1] - 1, y1))
    x2 = max(x1 + 1, min(overlay.size[0], x2))
    y2 = max(y1 + 1, min(overlay.size[1], y2))

    width = max(1, round(min(overlay.size) / 96))
    color = (255, 64, 64)
    draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
    return overlay


def crop_region(image, bbox, margin_ratio, draw_box=True):
    x1, y1, x2, y2 = compute_crop_window(image, bbox, margin_ratio)
    crop = image.crop((x1, y1, x2, y2))
    if not draw_box:
        return crop

    x, y, w, h = bbox
    rel_bbox = (x - x1, y - y1, x + w - x1, y + h - y1)
    return draw_crop_target_box(crop, rel_bbox)


def parse_vlm_json(text):
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find JSON object or array
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        idx_start = text.find(start_char)
        idx_end = text.rfind(end_char)
        if idx_start != -1 and idx_end != -1 and idx_end > idx_start:
            candidate = text[idx_start:idx_end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    return json.loads(text)


def get_done_image_ids(desc_dir):
    """Get set of image IDs already processed by ANY backend."""
    done = set()
    if not os.path.exists(desc_dir):
        return done
    for f in os.listdir(desc_dir):
        if f.endswith(".json"):
            try:
                done.add(int(f.replace(".json", "")))
            except ValueError:
                pass
    return done


def is_valid_vlm(vlm):
    return isinstance(vlm, dict)


def ann_matches_mode(ann, mode, small_area_thresh):
    area = ann["bbox"][2] * ann["bbox"][3]
    is_small = area < small_area_thresh
    if mode == "crop":
        return is_small
    if mode == "full":
        return not is_small
    return True


def get_pending_annotations(anns, existing_output, mode, small_area_thresh, force=False):
    if force or existing_output is None:
        return [ann for ann in anns if ann_matches_mode(ann, mode, small_area_thresh)]

    existing_by_id = {a["ann_id"]: a for a in existing_output.get("annotations", [])}
    pending = []
    for ann in anns:
        if not ann_matches_mode(ann, mode, small_area_thresh):
            continue
        existing = existing_by_id.get(ann["id"])
        if existing is None or not is_valid_vlm(existing.get("vlm")):
            pending.append(ann)
    return pending


def _image_processor_kwargs(max_pixels=None, min_pixels=None):
    kwargs = {}
    if max_pixels is not None:
        kwargs["max_pixels"] = max_pixels
    if min_pixels is not None:
        kwargs["min_pixels"] = min_pixels
    return kwargs


def get_debug_dir(args):
    return Path(args.debug_dir) if args.debug_dir else Path(args.output_dir) / "debug_failures"


def save_failed_crop_debug(args, img_id, img_path, ann, crop_img, prompt, raw_text=None, error=None):
    if not args.debug_save_failures:
        return

    debug_dir = get_debug_dir(args)
    debug_dir.mkdir(parents=True, exist_ok=True)

    stem = f"img{img_id}_ann{ann['id']}"
    crop_path = debug_dir / f"{stem}.jpg"
    meta_path = debug_dir / f"{stem}.json"
    raw_path = debug_dir / f"{stem}.txt"

    crop_img.save(crop_path, format="JPEG", quality=95)
    if raw_text is not None:
        raw_path.write_text(raw_text if isinstance(raw_text, str) else repr(raw_text))

    meta = {
        "image_id": img_id,
        "image_path": img_path,
        "file_name": os.path.basename(img_path),
        "ann_id": ann["id"],
        "bbox": ann["bbox"],
        "category_id": ann.get("category_id"),
        "crop_path": str(crop_path),
        "prompt": prompt,
        "error": error,
    }
    if raw_text is not None:
        meta["raw_response_path"] = str(raw_path)
    meta_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2))


def draw_numbered_boxes(image, anns):
    """Overlay numbered boxes on a copy of the image to improve grounding."""
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    width = max(2, round(min(image.size) / 240))
    text_pad = max(2, width)
    palette = [
        (255, 64, 64),
        (64, 160, 255),
        (64, 200, 120),
        (255, 170, 64),
        (200, 80, 255),
        (255, 220, 64),
    ]

    for idx, ann in enumerate(anns):
        x, y, w, h = ann["bbox"]
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + w))
        y2 = int(round(y + h))
        color = palette[idx % len(palette)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)

        label = str(idx)
        if hasattr(draw, "textbbox"):
            l, t, r, b = draw.textbbox((0, 0), label)
            tw = r - l
            th = b - t
        else:
            tw, th = draw.textsize(label)

        tx1 = max(0, x1)
        ty1 = max(0, y1 - th - 2 * text_pad)
        tx2 = min(image.size[0], tx1 + tw + 2 * text_pad)
        ty2 = min(image.size[1], ty1 + th + 2 * text_pad)
        draw.rectangle((tx1, ty1, tx2, ty2), fill=color)
        draw.text((tx1 + text_pad, ty1 + text_pad), label, fill=(255, 255, 255))

    return overlay


def load_qwen_model(model_name, device, dtype_str):
    """Load Qwen2-VL model and processor."""
    from transformers import AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLModel
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration as QwenVLModel

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[dtype_str]

    print(f"Loading {model_name} ({dtype_str})...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
    if hasattr(processor, "padding_side"):
        processor.padding_side = "left"
    load_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if device == "cuda":
        load_kwargs["device_map"] = "auto"
    model = QwenVLModel.from_pretrained(model_name, **load_kwargs)
    if device != "cuda":
        model = model.to(device)
    model.eval()
    input_device = getattr(model, "device", None)
    if input_device is None:
        input_device = next(model.parameters()).device
    print(f"Model loaded on {input_device}")
    return model, processor, input_device


def qwen_infer_batch(
    model, processor, images, prompt, input_device, max_new_tokens=512, image_processor_kwargs=None
):
    """Run Qwen2-VL inference on a batch of images with the same prompt."""
    texts = []
    for image in images:
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}
        ]
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
        **(image_processor_kwargs or {}),
    ).to(input_device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_lengths = inputs.attention_mask.sum(dim=1).tolist()
    generated_ids = [output_ids[i, input_lengths[i]:] for i in range(len(images))]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)


def qwen_infer(
    model, processor, image, prompt, input_device, max_new_tokens=512, image_processor_kwargs=None
):
    """Run Qwen2-VL inference on a single image with prompt."""
    return qwen_infer_batch(
        model,
        processor,
        [image],
        prompt,
        input_device,
        max_new_tokens=max_new_tokens,
        image_processor_kwargs=image_processor_kwargs,
    )[0]


def process_image_qwen(model, processor, input_device, img_id, img_path, img_w, img_h, anns, args):
    """Process all objects in one image with Qwen2-VL."""
    small_anns = []
    large_anns = []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        if w * h < args.small_area_thresh:
            small_anns.append(ann)
        else:
            large_anns.append(ann)

    results = {}
    image = Image.open(img_path).convert("RGB")

    # --- Crop mode for small objects ---
    if args.only != "full":
        batch_size = max(1, args.batch_size)
        crop_processor_kwargs = _image_processor_kwargs(max_pixels=args.crop_max_pixels)
        crop_prompt = PROMPT_CROP_BOX if args.crop_box_prompt else PROMPT_CROP
        for batch_start in range(0, len(small_anns), batch_size):
            batch_anns = small_anns[batch_start:batch_start + batch_size]
            batch_images = [
                crop_region(
                    image,
                    ann["bbox"],
                    args.crop_margin,
                    draw_box=args.crop_box_overlay,
                )
                for ann in batch_anns
            ]
            try:
                texts = qwen_infer_batch(
                    model,
                    processor,
                    batch_images,
                    crop_prompt,
                    input_device,
                    max_new_tokens=300,
                    image_processor_kwargs=crop_processor_kwargs,
                )
            except Exception as e:
                texts = [None] * len(batch_anns)
                for ann, crop_img in zip(batch_anns, batch_images):
                    save_failed_crop_debug(
                        args,
                        img_id,
                        img_path,
                        ann,
                        crop_img,
                        crop_prompt,
                        raw_text=None,
                        error=f"batch_infer_error: {type(e).__name__}: {e}",
                    )

            for ann, crop_img, text in zip(batch_anns, batch_images, texts):
                error = None
                try:
                    if text is None:
                        raise ValueError("missing batch output")
                    vlm = parse_vlm_json(text)
                    if isinstance(vlm, list):
                        vlm = vlm[0] if vlm else None
                    vlm = sanitize_vlm_output(vlm)
                    if not is_valid_vlm(vlm):
                        error = "parsed_response_not_dict"
                        vlm = None
                except Exception as e:
                    error = f"parse_error: {type(e).__name__}: {e}"
                    vlm = None
                if vlm is None:
                    save_failed_crop_debug(
                        args,
                        img_id,
                        img_path,
                        ann,
                        crop_img,
                        crop_prompt,
                        raw_text=text,
                        error=error,
                    )
                results[ann["id"]] = (vlm, "crop")

    # --- Full-image mode for large objects (batched by max_full_objects) ---
    if large_anns and args.only != "crop":
        full_processor_kwargs = _image_processor_kwargs(
            max_pixels=args.full_max_pixels, min_pixels=args.full_min_pixels
        )
        for chunk_start in range(0, len(large_anns), args.max_full_objects):
            chunk = large_anns[chunk_start:chunk_start + args.max_full_objects]
            full_image = draw_numbered_boxes(image, chunk) if not args.no_box_overlay else image

            objects_desc = []
            for j, ann in enumerate(chunk):
                x, y, w, h = ann["bbox"]
                objects_desc.append(
                    f"Object {j}: numbered box {j}, bbox=[{x:.0f},{y:.0f},{x+w:.0f},{y+h:.0f}]"
                )

            prompt = PROMPT_FULL_HEADER + "\n\nObjects:\n" + "\n".join(objects_desc)

            try:
                text = qwen_infer(
                    model,
                    processor,
                    full_image,
                    prompt,
                    input_device,
                    max_new_tokens=200 * len(chunk),
                    image_processor_kwargs=full_processor_kwargs,
                )
                vlm_list = parse_vlm_json(text)
                if not isinstance(vlm_list, list):
                    vlm_list = [vlm_list]
            except Exception:
                vlm_list = [None] * len(chunk)

            for k, ann in enumerate(chunk):
                vlm = vlm_list[k] if k < len(vlm_list) else None
                vlm = sanitize_vlm_output(vlm)
                results[ann["id"]] = (vlm, "full")

    return results


def main():
    args = parse_args()

    # Load annotations
    print(f"Loading annotations from {args.annotation}...")
    with open(args.annotation) as f:
        data = json.load(f)

    img_map = {img["id"]: img for img in data["images"]}
    ann_by_img = defaultdict(list)
    ann_id_to_image = {}
    for ann in data["annotations"]:
        ann_by_img[ann["image_id"]].append(ann)
        ann_id_to_image[ann["id"]] = ann["image_id"]

    img_ids = sorted(ann_by_img.keys())
    if args.ann_id is not None:
        ann_img_id = ann_id_to_image.get(args.ann_id)
        if ann_img_id is None:
            raise ValueError(f"Annotation id {args.ann_id} not found in {args.annotation}")
        if args.image_id is not None and args.image_id != ann_img_id:
            raise ValueError(f"ann_id {args.ann_id} belongs to image_id {ann_img_id}, not {args.image_id}")
        args.image_id = ann_img_id
    if args.image_id is not None:
        if args.image_id not in img_map:
            raise ValueError(f"image_id {args.image_id} not found in {args.annotation}")
        img_ids = [args.image_id]
    desc_dir = os.path.join(args.output_dir, "descriptions")
    os.makedirs(desc_dir, exist_ok=True)

    pending_anns_by_img = {}
    filtered = []
    for img_id in img_ids:
        out_path = os.path.join(desc_dir, f"{img_id}.json")
        existing = None
        if os.path.exists(out_path):
            with open(out_path) as f:
                existing = json.load(f)
        selected_anns = ann_by_img[img_id]
        if args.ann_id is not None:
            selected_anns = [ann for ann in selected_anns if ann["id"] == args.ann_id]
        pending_anns = get_pending_annotations(
            selected_anns, existing, args.only, args.small_area_thresh, force=args.force
        )
        if pending_anns:
            filtered.append(img_id)
            pending_anns_by_img[img_id] = pending_anns
    img_ids = filtered

    # Apply start/num-images to remaining undone images
    img_ids = img_ids[args.start:]
    if args.num_images > 0:
        img_ids = img_ids[:args.num_images]

    # Count work
    total_small = 0
    total_large = 0
    for img_id in img_ids:
        for ann in pending_anns_by_img[img_id]:
            x, y, w, h = ann["bbox"]
            if w * h < args.small_area_thresh:
                total_small += 1
            else:
                total_large += 1

    print(f"Mode: {args.only} | Images to process: {len(img_ids)}")
    print(f"Objects: {total_small} small (crop) + {total_large} large (full) = {total_small + total_large}")

    if args.dry_run or len(img_ids) == 0:
        print("Nothing to do, exiting.")
        return

    # Load model
    model, processor, input_device = load_qwen_model(args.model, args.device, args.dtype)

    # Process
    t0 = time.time()
    processed = 0

    for i, img_id in enumerate(img_ids):
        img_info = img_map[img_id]
        img_path = os.path.join(args.image_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        out_path = os.path.join(desc_dir, f"{img_id}.json")

        all_anns = ann_by_img[img_id]
        anns = pending_anns_by_img[img_id]
        img_w, img_h = img_info["width"], img_info["height"]

        try:
            results = process_image_qwen(
                model, processor, input_device, img_id, img_path, img_w, img_h, anns, args
            )
        except Exception as e:
            print(f"  Error on image {img_id}: {e}")
            results = {}

        # Build output — merge with existing file if present
        if os.path.exists(out_path):
            with open(out_path) as f:
                output = json.load(f)
            # Index existing annotations by ann_id
            existing_by_id = {a["ann_id"]: a for a in output.get("annotations", [])}
            # Merge new results into existing
            for ann in all_anns:
                if ann["id"] in results:
                    vlm, mode = results[ann["id"]]
                    if ann["id"] in existing_by_id:
                        if is_valid_vlm(vlm) or not is_valid_vlm(existing_by_id[ann["id"]].get("vlm")):
                            existing_by_id[ann["id"]]["vlm"] = vlm
                            existing_by_id[ann["id"]]["crop_mode"] = mode
                    else:
                        existing_by_id[ann["id"]] = {
                            "ann_id": ann["id"],
                            "bbox": ann["bbox"],
                            "category_id": ann.get("category_id"),
                            "vlm": vlm,
                            "crop_mode": mode,
                        }
                elif ann["id"] not in existing_by_id:
                    existing_by_id[ann["id"]] = {
                        "ann_id": ann["id"],
                        "bbox": ann["bbox"],
                        "category_id": ann.get("category_id"),
                        "vlm": None,
                        "crop_mode": "missing",
                    }
            output["annotations"] = list(existing_by_id.values())
        else:
            output = {
                "image_id": img_id,
                "file_name": img_info["file_name"],
                "source": args.model,
                "annotations": [],
            }
            for ann in anns:
                entry = {
                    "ann_id": ann["id"],
                    "bbox": ann["bbox"],
                    "category_id": ann.get("category_id"),
                }
                if ann["id"] in results:
                    vlm, mode = results[ann["id"]]
                    entry["vlm"] = vlm
                    entry["crop_mode"] = mode
                else:
                    entry["vlm"] = None
                    entry["crop_mode"] = "missing"
                output["annotations"].append(entry)

        with open(out_path, "w") as f:
            json.dump(output, f)

        processed += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(img_ids):
            elapsed = time.time() - t0
            rate = processed / elapsed * 3600 if elapsed > 0 else 0
            eta = (len(img_ids) - i - 1) / (processed / elapsed) if elapsed > 0 and processed > 0 else 0
            print(f"  [{i+1}/{len(img_ids)}] {processed} images, "
                  f"{rate:.0f} img/hr, ETA {eta/3600:.1f}h")

    elapsed = time.time() - t0
    print(f"\nDone! {processed} images in {elapsed/3600:.1f}h")
    print(f"Results saved to {desc_dir}")


if __name__ == "__main__":
    main()
