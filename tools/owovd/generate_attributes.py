#!/usr/bin/env python3
"""Generate visual attribute descriptions for COCO classes using GPT-4.

For each class, generates candidate visual attributes like "has four legs",
"made of metal", etc. These are used by VSAS to select generalizable
attributes for unknown object detection.

Usage:
    python tools/owovd/generate_attributes.py --api-key $OPENAI_API_KEY
    python tools/owovd/generate_attributes.py --from-file tools/owovd/coco_attributes.json  # skip GPT
"""

import argparse
import json
import os
from pathlib import Path

# COCO 80 classes in standard order
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

SYSTEM_PROMPT = """You are a visual attribute expert. For each object class, generate visual attributes that describe its appearance.
Each attribute should be a short phrase describing a visual property (shape, color, texture, material, parts, size, etc.).
Generate attributes that are:
1. Visually distinctive and observable in images
2. Some specific to this class, some shared with other classes
3. Diverse (cover different visual aspects)
Output as a JSON object mapping class name to a list of attribute strings."""

USER_PROMPT_TEMPLATE = """Generate {n} visual attributes for the following object classes.
Each attribute should be a short descriptive phrase like "has four legs", "is metallic", "has round shape", "has wheels".

Classes: {classes}

Output format:
{{"class_name": ["attribute1", "attribute2", ...], ...}}"""


def generate_with_gpt(classes, n_attrs=40, api_key=None):
    """Call GPT-4 to generate attributes for classes."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai not found. Run: pip install openai")

    client = OpenAI(api_key=api_key)

    # Process in batches to avoid token limits
    batch_size = 20
    all_attributes = {}

    for i in range(0, len(classes), batch_size):
        batch = classes[i : i + batch_size]
        print(f"Generating attributes for batch {i // batch_size + 1}: {batch[:3]}...")

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT_TEMPLATE.format(
                        n=n_attrs, classes=", ".join(batch)
                    ),
                },
            ],
            temperature=0.7,
            max_tokens=4096,
        )

        text = response.choices[0].message.content
        # Extract JSON from response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            batch_attrs = json.loads(text[start:end])
            all_attributes.update(batch_attrs)
        else:
            print(f"  Warning: failed to parse response for batch {i // batch_size + 1}")

    return all_attributes


def main():
    parser = argparse.ArgumentParser(description="Generate visual attributes for COCO classes")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY env)")
    parser.add_argument("--from-file", default=None, help="Load pre-generated attributes from JSON instead of calling GPT")
    parser.add_argument("--n-attrs", type=int, default=40, help="Number of attributes per class")
    parser.add_argument("--output", default="tools/owovd/coco_attributes.json", help="Output JSON path")
    args = parser.parse_args()

    if args.from_file:
        print(f"Loading attributes from {args.from_file}")
        attributes = json.load(open(args.from_file))
    else:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Provide --api-key or set OPENAI_API_KEY")
        attributes = generate_with_gpt(COCO_CLASSES, n_attrs=args.n_attrs, api_key=api_key)

    # Flatten all attributes into a unique list
    all_attrs = []
    attr_to_classes = {}
    for cls, attrs in attributes.items():
        for attr in attrs:
            attr = attr.strip().lower()
            if attr not in attr_to_classes:
                attr_to_classes[attr] = []
                all_attrs.append(attr)
            attr_to_classes[attr].append(cls)

    print(f"\nTotal unique attributes: {len(all_attrs)}")
    print(f"Classes covered: {len(attributes)}")
    print(f"Attributes shared across 3+ classes: {sum(1 for v in attr_to_classes.values() if len(v) >= 3)}")

    output = {
        "attributes": attributes,
        "all_attributes": all_attrs,
        "attribute_to_classes": attr_to_classes,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
