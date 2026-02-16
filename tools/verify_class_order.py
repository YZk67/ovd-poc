#!/usr/bin/env python3
"""
Verify that class order is consistent across:
  - OVDCOCO65 (builtin metadata / thing_classes)
  - ovdcoco_prompts_list8_v2.npy (text embeddings)
  - ovcoco_seen_classes.json / ovcoco_all_classes.json

If any mismatch exists, visualization and predictions will show wrong class labels.
"""
import json
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Must import detectron2 to register datasets and get OVDCOCO65
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import builtin  # triggers register_all_coco
from detectron2.data.datasets.builtin import OVDCOCO65


def main():
    root = Path(__file__).parent.parent
    dataset_root = root / "dataset"

    print("=" * 60)
    print("OVD-COCO 65 Class Order Verification")
    print("=" * 60)

    # 1. OVDCOCO65 (canonical order)
    print(f"\n1. OVDCOCO65 (builtin): {len(OVDCOCO65)} classes")
    print(f"   First 10: {OVDCOCO65[:10]}")
    print(f"   Index 6 (train): '{OVDCOCO65[6]}'")

    # 2. Metadata thing_classes for ovdcoco65_2017_val_all
    try:
        meta = MetadataCatalog.get("ovdcoco65_2017_val_all")
        tc = getattr(meta, "thing_classes", None)
        if tc is None:
            print("\n2. MetadataCatalog thing_classes: NOT SET (dataset may not be registered)")
        else:
            print(f"\n2. MetadataCatalog thing_classes: {len(tc)} classes")
            match = all(tc[i] == OVDCOCO65[i] for i in range(min(len(tc), len(OVDCOCO65))))
            print(f"   Matches OVDCOCO65: {'YES' if match else 'NO'}")
            if not match:
                for i in range(min(len(tc), len(OVDCOCO65))):
                    if tc[i] != OVDCOCO65[i]:
                        print(f"   MISMATCH at {i}: meta='{tc[i]}' vs OVDCOCO65='{OVDCOCO65[i]}'")
    except Exception as e:
        print(f"\n2. MetadataCatalog: ERROR - {e}")

    # 3. Text embeddings
    embed_path = dataset_root / "metadata" / "ovdcoco_prompts_list8_v2.npy"
    if embed_path.exists():
        import numpy as np
        emb = np.load(str(embed_path))
        print(f"\n3. ovdcoco_prompts_list8_v2.npy: shape {emb.shape}")
        if emb.shape[0] != 65:
            print(f"   WARNING: First dim is {emb.shape[0]}, expected 65!")
    else:
        print(f"\n3. ovdcoco_prompts_list8_v2.npy: NOT FOUND at {embed_path}")

    # 4. seen_classes / all_classes JSON
    seen_path = dataset_root / "metadata" / "ovcoco_seen_classes.json"
    all_path = dataset_root / "metadata" / "ovcoco_all_classes.json"
    for name, path in [("ovcoco_seen_classes.json", seen_path), ("ovcoco_all_classes.json", all_path)]:
        if path.exists():
            with open(path) as f:
                classes = json.load(f)
            print(f"\n4. {name}: {len(classes)} classes")
            if len(classes) == 65:
                match = all(classes[i] == OVDCOCO65[i] for i in range(65))
                print(f"   Matches OVDCOCO65: {'YES' if match else 'NO'}")
                if not match:
                    for i in range(65):
                        if classes[i] != OVDCOCO65[i]:
                            print(f"   MISMATCH at {i}: json='{classes[i]}' vs OVDCOCO65='{OVDCOCO65[i]}'")
                            break
        else:
            print(f"\n4. {name}: NOT FOUND")

    # 5. Prompt JSON if exists (source of text embedding order)
    prompt_json = dataset_root / "metadata" / "ovdcoco_prompts_list8_rich_v2.json"
    if prompt_json.exists():
        with open(prompt_json) as f:
            prompts = json.load(f)
        if isinstance(prompts, dict):
            prompt_classes = list(prompts.keys())
        elif isinstance(prompts, list):
            prompt_classes = [p.get("class", p.get("name", "")) for p in prompts]
        else:
            prompt_classes = []
        print(f"\n5. Prompt JSON class order: {len(prompt_classes)} classes")
        if len(prompt_classes) >= 65:
            match = all(prompt_classes[i] == OVDCOCO65[i] for i in range(65))
            print(f"   Matches OVDCOCO65: {'YES' if match else 'NO'}")
            if not match:
                for i in range(min(65, len(prompt_classes))):
                    if prompt_classes[i] != OVDCOCO65[i]:
                        print(f"   MISMATCH at {i}: prompt='{prompt_classes[i]}' vs OVDCOCO65='{OVDCOCO65[i]}'")
                        break
    else:
        print(f"\n5. ovdcoco_prompts_list8_rich_v2.json: NOT FOUND (cannot verify text embed source order)")

    print("\n" + "=" * 60)
    print("Summary: If any MISMATCH above, predictions will show wrong labels.")
    print("Fix: Regenerate text embeddings with class order = OVDCOCO65")
    print("=" * 60)


if __name__ == "__main__":
    main()
