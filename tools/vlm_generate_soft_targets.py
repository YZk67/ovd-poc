"""Generate VLM soft targets for OV-COCO training images.

Uses a VLM (LLaVA or OpenAI API) to produce structured descriptions
for each training image, then converts them to soft supervision signals:
  1. similarity_distribution: [65] soft class probabilities
  2. uncertainty_score: float, how uncertain the VLM is
  3. attributes: list of attribute strings per object

Usage:
    # Step 1a: Generate VLM descriptions - full image mode (original)
    python tools/vlm_generate_soft_targets.py \
        --mode generate \
        --image-dir dataset/coco/train2017 \
        --annotation dataset/coco/annotations/ovd_ins_train2017_b.json \
        --output-dir dataset/vlm_targets_crop \
        --num-images 100

    # Step 1b: Generate with crop mode (better for small objects)
    python tools/vlm_generate_soft_targets.py \
        --mode generate \
        --crop-mode crop \
        --crop-margin 0.5 \
        --image-dir dataset/coco/train2017 \
        --annotation dataset/coco/annotations/ovd_ins_train2017_b.json \
        --output-dir dataset/vlm_targets_crop

    # Step 1c: Generate with both mode (auto: crop small, full for large)
    python tools/vlm_generate_soft_targets.py \
        --mode generate \
        --crop-mode both \
        --small-area-thresh 4096 \
        --image-dir dataset/coco/train2017 \
        --annotation dataset/coco/annotations/ovd_ins_train2017_b.json \
        --output-dir dataset/vlm_targets_crop

    # Step 2: Convert descriptions to tensors
    python tools/vlm_generate_soft_targets.py \
        --mode convert \
        --descriptions-dir dataset/vlm_targets_crop/descriptions \
        --output-dir dataset/vlm_targets_crop \
        --clip-model convnext

    # Step 3 (optional): Analyze quality
    python tools/vlm_generate_soft_targets.py \
        --mode analyze \
        --output-dir dataset/vlm_targets_crop
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from pycocotools.coco import COCO

def _torch_load_compat(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


# All 65 classes in OVDCOCO65
ALL_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "kite", "skateboard", "surfboard", "bottle", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "pizza", "donut", "cake", "chair", "couch",
    "bed", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors", "toothbrush"
]

# 48 base (seen) classes
SEEN_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "train", "truck", "boat", "bench", "bird",
    "horse", "sheep", "bear", "zebra", "giraffe", "backpack", "handbag", "suitcase", "frisbee",
    "skis", "kite", "surfboard", "bottle", "fork", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "pizza", "donut", "chair", "bed", "toilet", "tv", "laptop",
    "mouse", "remote", "microwave", "oven", "toaster", "refrigerator", "book", "clock", "vase",
    "toothbrush"
]

# 17 novel classes
NOVEL_CLASSES = [c for c in ALL_CLASSES if c not in SEEN_CLASSES]

CLASS_TO_IDX = {name: i for i, name in enumerate(ALL_CLASSES)}

VLM_PROMPT_FULL = """Look at this image region (marked by the red box or described by coordinates).

For the object in this region, answer these questions in JSON format:
{
  "best_class": "<most likely class from the list below, or null if unknown>",
  "confidence": <0.0 to 1.0>,
  "top5_similar": {"<class1>": <score>, "<class2>": <score>, ...},
  "attributes": ["<attr1>", "<attr2>", ...],
  "description": "<one sentence describing the object>"
}

Available classes: """ + ", ".join(ALL_CLASSES) + """

Rules:
- top5_similar: give similarity scores (0-1) for the 5 most similar classes
- attributes: visual attributes like color, shape, size, material, state
- confidence: 1.0 = very sure, 0.0 = no idea what this is
- If the object doesn't match any class well, set best_class to null and confidence to 0
"""

VLM_PROMPT_CROP = """What is the main object in this cropped image region?

Answer in JSON format:
{
  "best_class": "<most likely class from the list below, or null if unknown>",
  "confidence": <0.0 to 1.0>,
  "top5_similar": {"<class1>": <score>, "<class2>": <score>, ...},
  "attributes": ["<attr1>", "<attr2>", ...],
  "description": "<one sentence describing the object>"
}

Available classes: """ + ", ".join(ALL_CLASSES) + """

Rules:
- top5_similar: give similarity scores (0-1) for the 5 most similar classes
- attributes: visual attributes like color, shape, size, material, state
- confidence: 1.0 = very sure, 0.0 = no idea what this is
- If the object doesn't match any class well, set best_class to null and confidence to 0
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generate", "convert", "analyze"], required=True)
    parser.add_argument("--image-dir", default="dataset/coco/train2017")
    parser.add_argument("--annotation", default="dataset/coco/annotations/ovd_ins_train2017_b.json")
    parser.add_argument("--output-dir", default="dataset/vlm_targets_crop")
    parser.add_argument("--num-images", type=int, default=0, help="0 = all images")
    parser.add_argument("--descriptions-dir", default=None)
    parser.add_argument("--clip-model", default="convnext", choices=["convnext", "vitl14"])
    # VLM backend
    parser.add_argument("--vlm", default="openai", choices=["openai", "llava"])
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--batch-size", type=int, default=1, help="Images per VLM call")
    # Crop mode for small objects
    parser.add_argument("--crop-mode", default="none", choices=["none", "crop", "both"],
                        help="none: full image + bbox coords (original). "
                             "crop: crop each object region separately. "
                             "both: full image for large objects, crop for small ones.")
    parser.add_argument("--crop-margin", type=float, default=0.5,
                        help="Context margin ratio around bbox for crop mode (default: 0.5)")
    parser.add_argument("--small-area-thresh", type=int, default=4096,
                        help="Area threshold (w*h) below which crop mode is used in 'both' mode (default: 4096 = ~64x64)")
    return parser.parse_args()


# ============================================================
# Mode 1: Generate VLM descriptions
# ============================================================

def generate_descriptions(args):
    """Run VLM on training images to produce per-object descriptions."""
    os.makedirs(os.path.join(args.output_dir, "descriptions"), exist_ok=True)

    # Load annotations
    coco = COCO(args.annotation)
    img_ids = sorted(coco.getImgIds())
    if args.num_images > 0:
        img_ids = img_ids[:args.num_images]

    print(f"Generating descriptions for {len(img_ids)} images...")

    # Check which images already have descriptions (for resume)
    done = set()
    desc_dir = os.path.join(args.output_dir, "descriptions")
    for f in os.listdir(desc_dir):
        if f.endswith(".json"):
            done.add(int(f.replace(".json", "")))
    img_ids = [i for i in img_ids if i not in done]
    print(f"Skipping {len(done)} already processed. Remaining: {len(img_ids)}")

    if args.vlm == "openai":
        generate_with_openai(args, coco, img_ids, desc_dir)
    elif args.vlm == "llava":
        generate_with_llava(args, coco, img_ids, desc_dir)


def _crop_region(image, bbox, margin_ratio, img_w, img_h):
    """Crop a region from PIL image with context margin.

    Args:
        image: PIL Image
        bbox: [x, y, w, h] in COCO format
        margin_ratio: context margin as ratio of max(w, h)
        img_w, img_h: full image dimensions

    Returns:
        Cropped PIL Image
    """
    x, y, w, h = bbox
    margin = max(w, h) * margin_ratio
    x1 = max(0, int(x - margin))
    y1 = max(0, int(y - margin))
    x2 = min(img_w, int(x + w + margin))
    y2 = min(img_h, int(y + h + margin))
    return image.crop((x1, y1, x2, y2))


def _should_crop(ann, crop_mode, small_area_thresh):
    """Decide whether to use crop mode for this annotation."""
    if crop_mode == "crop":
        return True
    if crop_mode == "both":
        _, _, w, h = ann["bbox"]
        return w * h < small_area_thresh
    return False


def _parse_vlm_json(result_text):
    """Parse JSON from VLM response, handling markdown code blocks."""
    if "```json" in result_text:
        result_text = result_text.split("```json")[1].split("```")[0]
    elif "```" in result_text:
        result_text = result_text.split("```")[1].split("```")[0]
    result = json.loads(result_text.strip())
    return result


def generate_with_openai(args, coco, img_ids, desc_dir):
    """Generate descriptions using OpenAI API (GPT-4o-mini).

    Supports three crop modes:
      - none: send full image with bbox coordinates (original behavior)
      - crop: crop each object region and send separately
      - both: crop small objects, full image for large ones
    """
    import base64
    from io import BytesIO
    try:
        from openai import OpenAI
        from PIL import Image
    except ImportError:
        print("pip install openai pillow")
        return

    client = OpenAI()  # Uses OPENAI_API_KEY env var

    for i, img_id in enumerate(img_ids):
        if i % 100 == 0:
            print(f"Processing {i}/{len(img_ids)}...")

        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(args.image_dir, img_info["file_name"])
        img_w, img_h = img_info["width"], img_info["height"]

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        if not anns:
            continue

        # Determine which annotations need cropping
        need_crop = [_should_crop(ann, args.crop_mode, args.small_area_thresh) for ann in anns]
        crop_anns = [ann for ann, nc in zip(anns, need_crop) if nc]
        full_anns = [ann for ann, nc in zip(anns, need_crop) if not nc]

        # Load image (needed for crop mode or if any annotation needs cropping)
        image = None
        if any(need_crop):
            image = Image.open(img_path).convert("RGB")

        # Encode full image for full-image mode annotations
        full_img_b64 = None
        if full_anns:
            with open(img_path, "rb") as f:
                full_img_b64 = base64.b64encode(f.read()).decode()

        results_by_idx = {}  # ann index -> vlm result

        # --- Process full-image annotations (original mode) ---
        if full_anns:
            full_indices = [j for j, nc in enumerate(need_crop) if not nc]
            objects_desc = []
            for k, ann in enumerate(full_anns):
                x, y, w, h = ann["bbox"]
                objects_desc.append(f"Object {k}: bbox=[{x:.0f},{y:.0f},{x+w:.0f},{y+h:.0f}]")

            full_prompt = VLM_PROMPT_FULL + "\n\nObjects to describe:\n" + "\n".join(objects_desc)
            full_prompt += "\n\nReturn a JSON array with one entry per object, in the same order."

            try:
                response = client.chat.completions.create(
                    model=args.openai_model,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": full_prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{full_img_b64}",
                            "detail": "low",
                        }},
                    ]}],
                    max_tokens=2000,
                    temperature=0.1,
                )
                result = _parse_vlm_json(response.choices[0].message.content)
                if not isinstance(result, list):
                    result = [result]
            except Exception as e:
                print(f"  Error on image {img_id} (full): {e}")
                result = []

            for k, idx in enumerate(full_indices):
                results_by_idx[idx] = result[k] if k < len(result) else None

        # --- Process cropped annotations (one API call per object) ---
        if crop_anns:
            crop_indices = [j for j, nc in enumerate(need_crop) if nc]
            for k, (ann, idx) in enumerate(zip(crop_anns, crop_indices)):
                x, y, w, h = ann["bbox"]
                crop_img = _crop_region(image, ann["bbox"], args.crop_margin, img_w, img_h)

                # Encode cropped region
                buf = BytesIO()
                crop_img.save(buf, format="JPEG", quality=95)
                crop_b64 = base64.b64encode(buf.getvalue()).decode()

                try:
                    response = client.chat.completions.create(
                        model=args.openai_model,
                        messages=[{"role": "user", "content": [
                            {"type": "text", "text": VLM_PROMPT_CROP},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{crop_b64}",
                                "detail": "auto",
                            }},
                        ]}],
                        max_tokens=500,
                        temperature=0.1,
                    )
                    result = _parse_vlm_json(response.choices[0].message.content)
                    if isinstance(result, list):
                        result = result[0] if result else None
                    results_by_idx[idx] = result
                except Exception as e:
                    print(f"  Error on image {img_id} object {idx} (crop, bbox=[{x:.0f},{y:.0f},{w:.0f},{h:.0f}]): {e}")
                    results_by_idx[idx] = None

        # Save per-image description
        output = {
            "image_id": img_id,
            "file_name": img_info["file_name"],
            "annotations": [],
        }
        for j, ann in enumerate(anns):
            entry = {
                "ann_id": ann["id"],
                "bbox": ann["bbox"],
                "category_id": ann.get("category_id"),
                "crop_mode": "crop" if need_crop[j] else "full",
            }
            entry["vlm"] = results_by_idx.get(j, None)
            output["annotations"].append(entry)

        with open(os.path.join(desc_dir, f"{img_id}.json"), "w") as f:
            json.dump(output, f)

    print(f"Done. Descriptions saved to {desc_dir}")


def generate_with_llava(args, coco, img_ids, desc_dir):
    """Generate descriptions using local LLaVA model.

    Supports crop modes like OpenAI backend.
    """
    try:
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        from PIL import Image
    except ImportError:
        print("pip install transformers pillow")
        return

    print("Loading LLaVA model...")
    model_name = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_name)
    llava_model = LlavaForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )

    def _llava_infer(pil_image, prompt_text):
        full_prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
        inputs = processor(text=full_prompt, images=pil_image, return_tensors="pt").to(llava_model.device)
        with torch.no_grad():
            output_ids = llava_model.generate(**inputs, max_new_tokens=2000, temperature=0.1, do_sample=False)
        result_text = processor.decode(output_ids[0], skip_special_tokens=True)
        if "ASSISTANT:" in result_text:
            result_text = result_text.split("ASSISTANT:")[-1].strip()
        return _parse_vlm_json(result_text)

    for i, img_id in enumerate(img_ids):
        if i % 100 == 0:
            print(f"Processing {i}/{len(img_ids)}...")

        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(args.image_dir, img_info["file_name"])
        img_w, img_h = img_info["width"], img_info["height"]
        image = Image.open(img_path).convert("RGB")

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        if not anns:
            continue

        need_crop = [_should_crop(ann, args.crop_mode, args.small_area_thresh) for ann in anns]
        full_anns = [ann for ann, nc in zip(anns, need_crop) if not nc]
        crop_anns = [ann for ann, nc in zip(anns, need_crop) if nc]

        results_by_idx = {}

        # --- Full-image mode ---
        if full_anns:
            full_indices = [j for j, nc in enumerate(need_crop) if not nc]
            objects_desc = []
            for k, ann in enumerate(full_anns):
                x, y, w, h = ann["bbox"]
                objects_desc.append(f"Object {k}: bbox=[{x:.0f},{y:.0f},{x+w:.0f},{y+h:.0f}]")

            prompt = VLM_PROMPT_FULL + "\n\nObjects to describe:\n" + "\n".join(objects_desc)
            prompt += "\n\nReturn a JSON array with one entry per object."

            try:
                result = _llava_infer(image, prompt)
                if not isinstance(result, list):
                    result = [result]
            except Exception:
                result = []

            for k, idx in enumerate(full_indices):
                results_by_idx[idx] = result[k] if k < len(result) else None

        # --- Crop mode ---
        if crop_anns:
            crop_indices = [j for j, nc in enumerate(need_crop) if nc]
            for k, (ann, idx) in enumerate(zip(crop_anns, crop_indices)):
                crop_img = _crop_region(image, ann["bbox"], args.crop_margin, img_w, img_h)
                try:
                    result = _llava_infer(crop_img, VLM_PROMPT_CROP)
                    if isinstance(result, list):
                        result = result[0] if result else None
                    results_by_idx[idx] = result
                except Exception:
                    results_by_idx[idx] = None

        # Save
        output = {
            "image_id": img_id,
            "file_name": img_info["file_name"],
            "annotations": [],
        }
        for j, ann in enumerate(anns):
            entry = {
                "ann_id": ann["id"],
                "bbox": ann["bbox"],
                "category_id": ann.get("category_id"),
                "crop_mode": "crop" if need_crop[j] else "full",
            }
            entry["vlm"] = results_by_idx.get(j, None)
            output["annotations"].append(entry)

        with open(os.path.join(desc_dir, f"{img_id}.json"), "w") as f:
            json.dump(output, f)

    print(f"Done. Descriptions saved to {desc_dir}")


# ============================================================
# Mode 2: Convert descriptions to training tensors
# ============================================================

def convert_descriptions(args):
    """Convert VLM JSON descriptions to soft target tensors."""
    desc_dir = args.descriptions_dir or os.path.join(args.output_dir, "descriptions")

    print(f"Loading descriptions from {desc_dir}...")
    desc_files = sorted(Path(desc_dir).glob("*.json"))
    print(f"Found {len(desc_files)} description files")

    all_targets = {}  # image_id -> list of per-object targets

    for f in desc_files:
        with open(f) as fp:
            data = json.load(fp)

        image_id = data["image_id"]
        targets = []

        for ann in data["annotations"]:
            vlm = ann.get("vlm")
            if vlm is None:
                targets.append(None)
                continue

            # 1. Similarity distribution [65]
            sim_dist = np.zeros(len(ALL_CLASSES), dtype=np.float32)
            top5 = vlm.get("top5_similar", {})
            normalized_top5 = {}
            if isinstance(top5, dict):
                for cls_name, score in top5.items():
                    cls_name_lower = cls_name.lower().strip()
                    if cls_name_lower not in CLASS_TO_IDX:
                        continue
                    try:
                        score = float(score)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(score) or score <= 0:
                        continue
                    normalized_top5[cls_name_lower] = max(normalized_top5.get(cls_name_lower, 0.0), score)
            if normalized_top5:
                total_score = sum(normalized_top5.values())
                for cls_name, score in normalized_top5.items():
                    sim_dist[CLASS_TO_IDX[cls_name]] = float(score) / total_score

            # Fall back to best_class only when needed, but do not overwrite an existing normalized distribution.
            best = vlm.get("best_class")
            if best and best.lower().strip() in CLASS_TO_IDX:
                best_idx = CLASS_TO_IDX[best.lower().strip()]
                conf = vlm.get("confidence", 0.5)
                if sim_dist.sum() == 0:
                    sim_dist[best_idx] = max(float(conf), 1e-6)
                elif sim_dist[best_idx] == 0:
                    sim_dist[best_idx] = max(float(conf), 1e-6)
                    sim_dist = sim_dist / sim_dist.sum()
            # Normalize to sum=1 (soft label)
            if sim_dist.sum() > 0:
                sim_dist = sim_dist / sim_dist.sum()

            # 2. Uncertainty score
            confidence = vlm.get("confidence", 0.5)
            uncertainty = 1.0 - float(confidence)

            # 3. Attributes (store as strings, encode to CLIP later if needed)
            attributes = vlm.get("attributes", [])

            targets.append({
                "ann_id": ann["ann_id"],
                "bbox": ann["bbox"],
                "similarity_distribution": sim_dist,
                "uncertainty": uncertainty,
                "attributes": attributes,
                "description": vlm.get("description", ""),
            })

        all_targets[image_id] = targets

    # Save as .pth
    output_path = os.path.join(args.output_dir, "vlm_soft_targets.pth")
    torch.save(all_targets, output_path)
    print(f"Saved {len(all_targets)} images' targets to {output_path}")

    # Also encode attributes to CLIP embeddings if requested
    if args.clip_model:
        encode_attributes_clip(args, all_targets)


def encode_attributes_clip(args, all_targets):
    """Encode attribute strings to CLIP text embeddings."""
    try:
        import open_clip
    except ImportError:
        print("Skipping CLIP encoding (pip install open_clip_torch)")
        return

    # Collect unique attributes
    all_attrs = set()
    for targets in all_targets.values():
        for t in targets:
            if t is not None:
                all_attrs.update(t.get("attributes", []))

    all_attrs = sorted(all_attrs)
    print(f"Encoding {len(all_attrs)} unique attributes with CLIP...")

    if args.clip_model == "convnext":
        model, _, preprocess = open_clip.create_model_and_transforms(
            "convnext_large_d_320", pretrained="laion2b_s29b_b131k_ft_soup"
        )
        tokenizer = open_clip.get_tokenizer("convnext_large_d_320")
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        tokenizer = open_clip.get_tokenizer("ViT-L-14")

    model.eval()

    # Encode in batches
    batch_size = 64
    all_embeddings = []
    for i in range(0, len(all_attrs), batch_size):
        batch = all_attrs[i:i+batch_size]
        tokens = tokenizer(batch)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embeddings.append(emb.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)  # [num_attrs, feat_dim]

    attr_data = {
        "attribute_names": all_attrs,
        "attribute_embeddings": all_embeddings,
    }
    output_path = os.path.join(args.output_dir, "vlm_attribute_embeddings.pth")
    torch.save(attr_data, output_path)
    print(f"Saved {len(all_attrs)} attribute embeddings to {output_path}")


# ============================================================
# Mode 3: Analyze quality
# ============================================================

def analyze_quality(args):
    """Analyze VLM description quality and coverage."""
    targets_path = os.path.join(args.output_dir, "vlm_soft_targets.pth")
    if not os.path.exists(targets_path):
        print(f"Run --mode convert first. {targets_path} not found.")
        return

    targets = _torch_load_compat(targets_path)

    total_objects = 0
    valid_objects = 0
    confidence_sum = 0
    class_coverage = defaultdict(int)  # how many times each class appears in top5
    uncertainty_hist = defaultdict(int)

    for image_id, objs in targets.items():
        for obj in objs:
            total_objects += 1
            if obj is None:
                continue
            valid_objects += 1

            unc = obj["uncertainty"]
            confidence_sum += (1 - unc)

            # Bin uncertainty
            bin_key = f"{int(unc * 10) / 10:.1f}"
            uncertainty_hist[bin_key] += 1

            sim = obj["similarity_distribution"]
            top5_idx = np.argsort(sim)[-5:]
            for idx in top5_idx:
                if sim[idx] > 0:
                    class_coverage[ALL_CLASSES[idx]] += 1

    print(f"Total objects: {total_objects}")
    print(f"Valid VLM descriptions: {valid_objects} ({valid_objects/max(total_objects,1)*100:.1f}%)")
    print(f"Average confidence: {confidence_sum/max(valid_objects,1):.3f}")

    print(f"\nUncertainty distribution:")
    for k in sorted(uncertainty_hist.keys()):
        bar = "#" * (uncertainty_hist[k] // 100)
        print(f"  {k}: {uncertainty_hist[k]:6d} {bar}")

    print(f"\nTop-20 classes in VLM descriptions:")
    for cls, count in sorted(class_coverage.items(), key=lambda x: -x[1])[:20]:
        is_novel = "NOVEL" if cls in NOVEL_CLASSES else "base"
        print(f"  {cls:20s}: {count:6d} ({is_novel})")

    # Check novel class coverage
    novel_coverage = sum(class_coverage.get(c, 0) for c in NOVEL_CLASSES)
    base_coverage = sum(class_coverage.get(c, 0) for c in SEEN_CLASSES)
    print(f"\nNovel class mentions: {novel_coverage}")
    print(f"Base class mentions: {base_coverage}")


def main():
    args = parse_args()

    if args.mode == "generate":
        generate_descriptions(args)
    elif args.mode == "convert":
        convert_descriptions(args)
    elif args.mode == "analyze":
        analyze_quality(args)


if __name__ == "__main__":
    main()
