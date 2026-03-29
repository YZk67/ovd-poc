"""Generate VLM soft targets for OV-COCO training images.

Uses a VLM (LLaVA or OpenAI API) to produce structured descriptions
for each training image, then converts them to soft supervision signals:
  1. similarity_distribution: [65] soft class probabilities
  2. uncertainty_score: float, how uncertain the VLM is
  3. attributes: list of attribute strings per object

Usage:
    # Step 1: Generate VLM descriptions (run on GPU server with LLaVA, or use OpenAI API)
    python tools/vlm_generate_soft_targets.py \
        --mode generate \
        --image-dir dataset/coco/train2017 \
        --annotation dataset/coco/annotations/ovd_ins_train2017_b.json \
        --output-dir dataset/vlm_targets \
        --num-images 100  # start small to test

    # Step 2: Convert descriptions to tensors
    python tools/vlm_generate_soft_targets.py \
        --mode convert \
        --descriptions-dir dataset/vlm_targets/descriptions \
        --output-dir dataset/vlm_targets \
        --clip-model convnext

    # Step 3 (optional): Analyze quality
    python tools/vlm_generate_soft_targets.py \
        --mode analyze \
        --output-dir dataset/vlm_targets
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

VLM_PROMPT = """Look at this image region (marked by the red box or described by coordinates).

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generate", "convert", "analyze"], required=True)
    parser.add_argument("--image-dir", default="dataset/coco/train2017")
    parser.add_argument("--annotation", default="dataset/coco/annotations/ovd_ins_train2017_b.json")
    parser.add_argument("--output-dir", default="dataset/vlm_targets")
    parser.add_argument("--num-images", type=int, default=0, help="0 = all images")
    parser.add_argument("--descriptions-dir", default=None)
    parser.add_argument("--clip-model", default="convnext", choices=["convnext", "vitl14"])
    # VLM backend
    parser.add_argument("--vlm", default="openai", choices=["openai", "llava"])
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--batch-size", type=int, default=1, help="Images per VLM call")
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


def generate_with_openai(args, coco, img_ids, desc_dir):
    """Generate descriptions using OpenAI API (GPT-4o-mini)."""
    import base64
    try:
        from openai import OpenAI
    except ImportError:
        print("pip install openai")
        return

    client = OpenAI()  # Uses OPENAI_API_KEY env var

    for i, img_id in enumerate(img_ids):
        if i % 100 == 0:
            print(f"Processing {i}/{len(img_ids)}...")

        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(args.image_dir, img_info["file_name"])

        # Get annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        if not anns:
            continue

        # Encode image
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        # Build prompt with object locations
        objects_desc = []
        for j, ann in enumerate(anns):
            x, y, w, h = ann["bbox"]
            objects_desc.append(f"Object {j}: bbox=[{x:.0f},{y:.0f},{x+w:.0f},{y+h:.0f}]")

        full_prompt = VLM_PROMPT + "\n\nObjects to describe:\n" + "\n".join(objects_desc)
        full_prompt += "\n\nReturn a JSON array with one entry per object, in the same order."

        try:
            response = client.chat.completions.create(
                model=args.openai_model,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": full_prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}",
                            "detail": "low",
                        }},
                    ]}
                ],
                max_tokens=2000,
                temperature=0.1,
            )
            result_text = response.choices[0].message.content

            # Parse JSON from response
            # Handle markdown code blocks
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            result = json.loads(result_text)
            if not isinstance(result, list):
                result = [result]

        except Exception as e:
            print(f"  Error on image {img_id}: {e}")
            result = []

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
            }
            if j < len(result):
                entry["vlm"] = result[j]
            else:
                entry["vlm"] = None
            output["annotations"].append(entry)

        with open(os.path.join(desc_dir, f"{img_id}.json"), "w") as f:
            json.dump(output, f)

    print(f"Done. Descriptions saved to {desc_dir}")


def generate_with_llava(args, coco, img_ids, desc_dir):
    """Generate descriptions using local LLaVA model."""
    try:
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        from PIL import Image
    except ImportError:
        print("pip install transformers pillow")
        return

    print("Loading LLaVA model...")
    model_name = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )

    for i, img_id in enumerate(img_ids):
        if i % 100 == 0:
            print(f"Processing {i}/{len(img_ids)}...")

        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(args.image_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        if not anns:
            continue

        objects_desc = []
        for j, ann in enumerate(anns):
            x, y, w, h = ann["bbox"]
            objects_desc.append(f"Object {j}: bbox=[{x:.0f},{y:.0f},{x+w:.0f},{y+h:.0f}]")

        full_prompt = "USER: <image>\n" + VLM_PROMPT
        full_prompt += "\n\nObjects to describe:\n" + "\n".join(objects_desc)
        full_prompt += "\n\nReturn a JSON array with one entry per object.\nASSISTANT:"

        inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=2000, temperature=0.1, do_sample=False)

        result_text = processor.decode(output_ids[0], skip_special_tokens=True)
        # Extract assistant response
        if "ASSISTANT:" in result_text:
            result_text = result_text.split("ASSISTANT:")[-1].strip()

        try:
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            result = json.loads(result_text)
            if not isinstance(result, list):
                result = [result]
        except Exception:
            result = []

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
            }
            entry["vlm"] = result[j] if j < len(result) else None
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
            for cls_name, score in top5.items():
                cls_name_lower = cls_name.lower().strip()
                if cls_name_lower in CLASS_TO_IDX:
                    sim_dist[CLASS_TO_IDX[cls_name_lower]] = float(score)
            # Also add best_class
            best = vlm.get("best_class")
            if best and best.lower().strip() in CLASS_TO_IDX:
                conf = vlm.get("confidence", 0.5)
                sim_dist[CLASS_TO_IDX[best.lower().strip()]] = max(
                    sim_dist[CLASS_TO_IDX[best.lower().strip()]], float(conf)
                )
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

    targets = torch.load(targets_path)

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
