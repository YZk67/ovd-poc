#!/usr/bin/env python3
"""Create M-OWODB and S-OWODB dataset splits from COCO annotations.

M-OWODB: 4 tasks, 20 classes each, mixed across super-categories.
S-OWODB: 4 tasks, grouped by COCO super-categories.

For each task T_i:
  - Train JSON: only annotations for classes known up to T_i (classes from T_1..T_i)
  - Test JSON: all 80-class annotations (for computing U-Recall on unknowns)

Following OW-DETR / OW-OVD benchmark splits.

Usage:
    python tools/owovd/create_owodb_splits.py \
        --coco-dir dataset/coco \
        --output-dir dataset/coco/annotations/owodb
"""

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

# M-OWODB split: classes grouped into 4 tasks (mixed super-categories)
# Following OW-DETR standard split
M_OWODB_TASKS = {
    1: [
        "airplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "dining table", "dog", "horse", "motorcycle", "person",
        "potted plant", "sheep", "couch", "train", "tv",
    ],
    2: [
        "truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
        "bench", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase",
        "microwave", "oven", "toaster", "sink", "refrigerator",
    ],
    3: [
        "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake",
    ],
    4: [
        "bed", "toilet", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    ],
}

# S-OWODB split: grouped by COCO super-categories
S_OWODB_TASKS = {
    1: [  # outdoor, vehicle, animal
        "person", "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow",
    ],
    2: [  # more animals, accessory
        "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase",
        "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle",
    ],
    3: [  # kitchen, food
        "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut",
        "cake", "chair", "couch", "potted plant", "bed",
    ],
    4: [  # furniture, electronic, appliance, indoor
        "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock",
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    ],
}


def load_coco(ann_path):
    print(f"Loading {ann_path}...")
    with open(ann_path) as f:
        return json.load(f)


def get_cat_name_to_id(coco):
    return {cat["name"]: cat["id"] for cat in coco["categories"]}


def create_task_split(coco, task_classes_by_task, task_id, split_name, output_dir):
    """Create train/test JSONs for a specific task.

    Train: annotations only for known classes (tasks 1..task_id)
    Test: all annotations (for U-Recall/U-mAP computation)
    """
    name_to_id = get_cat_name_to_id(coco)

    # Known classes = union of tasks 1..task_id
    known_classes = []
    for t in range(1, task_id + 1):
        known_classes.extend(task_classes_by_task[t])
    known_cat_ids = set()
    for cls_name in known_classes:
        if cls_name in name_to_id:
            known_cat_ids.add(name_to_id[cls_name])
        else:
            print(f"  Warning: class '{cls_name}' not found in COCO categories")

    all_cat_ids = set(cat["id"] for cat in coco["categories"])
    unknown_cat_ids = all_cat_ids - known_cat_ids

    # Build annotation index
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    # --- Train JSON: only known class annotations ---
    train_anns = [
        ann for ann in coco["annotations"] if ann["category_id"] in known_cat_ids
    ]
    # Only include images that have at least one known annotation
    train_image_ids = set(ann["image_id"] for ann in train_anns)
    train_images = [img for img in coco["images"] if img["id"] in train_image_ids]

    train_json = {
        "images": train_images,
        "annotations": train_anns,
        "categories": coco["categories"],
        "info": {
            "split": split_name,
            "task_id": task_id,
            "known_classes": known_classes,
            "num_known_cats": len(known_cat_ids),
            "num_unknown_cats": len(unknown_cat_ids),
        },
    }

    train_path = output_dir / f"{split_name}_t{task_id}_train.json"
    with open(train_path, "w") as f:
        json.dump(train_json, f)
    print(f"  Train: {len(train_images)} images, {len(train_anns)} anns, "
          f"{len(known_cat_ids)} known cats -> {train_path}")

    # --- Test JSON: all annotations, with unknown flag ---
    test_anns = copy.deepcopy(coco["annotations"])
    for ann in test_anns:
        ann["is_known"] = int(ann["category_id"] in known_cat_ids)

    test_json = {
        "images": coco["images"],
        "annotations": test_anns,
        "categories": coco["categories"],
        "info": {
            "split": split_name,
            "task_id": task_id,
            "known_cat_ids": sorted(known_cat_ids),
            "unknown_cat_ids": sorted(unknown_cat_ids),
        },
    }

    test_path = output_dir / f"{split_name}_t{task_id}_test.json"
    with open(test_path, "w") as f:
        json.dump(test_json, f)
    print(f"  Test:  {len(coco['images'])} images, {len(test_anns)} anns -> {test_path}")

    return known_cat_ids, unknown_cat_ids


def main():
    parser = argparse.ArgumentParser(description="Create OWODB splits")
    parser.add_argument("--coco-dir", default="dataset/coco", help="COCO dataset root")
    parser.add_argument("--output-dir", default="dataset/coco/annotations/owodb", help="Output directory")
    parser.add_argument("--train-ann", default=None, help="Train annotation JSON (default: {coco-dir}/annotations/instances_train2017.json)")
    parser.add_argument("--val-ann", default=None, help="Val annotation JSON (default: {coco-dir}/annotations/instances_val2017.json)")
    args = parser.parse_args()

    coco_dir = Path(args.coco_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ann = args.train_ann or str(coco_dir / "annotations" / "instances_train2017.json")
    val_ann = args.val_ann or str(coco_dir / "annotations" / "instances_val2017.json")

    train_coco = load_coco(train_ann)
    val_coco = load_coco(val_ann)

    # Generate M-OWODB splits
    print("\n=== M-OWODB (Mixed) ===")
    for task_id in range(1, 5):
        print(f"\nTask {task_id}: +{len(M_OWODB_TASKS[task_id])} classes")
        create_task_split(train_coco, M_OWODB_TASKS, task_id, "mowodb", output_dir)
        create_task_split(val_coco, M_OWODB_TASKS, task_id, "mowodb_val", output_dir)

    # Generate S-OWODB splits
    print("\n=== S-OWODB (Super-category) ===")
    for task_id in range(1, 5):
        print(f"\nTask {task_id}: +{len(S_OWODB_TASKS[task_id])} classes")
        create_task_split(train_coco, S_OWODB_TASKS, task_id, "sowodb", output_dir)
        create_task_split(val_coco, S_OWODB_TASKS, task_id, "sowodb_val", output_dir)

    # Save task class lists for reference
    meta = {
        "mowodb": {str(k): v for k, v in M_OWODB_TASKS.items()},
        "sowodb": {str(k): v for k, v in S_OWODB_TASKS.items()},
    }
    meta_path = output_dir / "owodb_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved metadata to {meta_path}")


if __name__ == "__main__":
    main()
