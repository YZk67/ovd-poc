# Inference files checklist (Linux)

Use this list to confirm all required **data/asset files** are present on the Linux machine before running inference with:

- **Config**: `lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py`
- **Test set**: `ovdcoco65_2017_val_all` (COCO val2017, 65 classes)

Paths are relative to the **project root** (or to `DETECTRON2_DATASETS` if you set that env var). Run the verify script from the project root.

---

## 1. Checkpoint (model weights)

| Path | Description |
|------|-------------|
| `pretrained_models/model_0028399.pth` | Trained model for inference (you pass this as `train.init_checkpoint=...`) |

---

## 2. Metadata (model loads these at init)

| Path | Type | Description |
|------|------|-------------|
| `dataset/metadata/ovdcoco_vlm_query_convnextl.npy` | .npy | VLM query embeddings |
| `dataset/metadata/ovcoco_seen_classes.json` | .json | Seen (base) class names |
| `dataset/metadata/ovcoco_all_classes.json` | .json | All 65 class names |
| `dataset/metadata/ovdcoco_prompts_list8_v2.npy` | .npy | Text prompts embeddings (query + TPA) |
| `dataset/metadata/ovd_ins_train2017_all_cat_info.json` | .json | Category frequency info |
| `dataset/cluster/ovd_cluster_128.npy` | .npy | Cluster labels (Fed Loss; loaded at init) |

---

## 3. Test dataset (val2017)

| Path | Type | Description |
|------|------|-------------|
| `dataset/coco/annotations/ovd_ins_val2017_all.json` | .json | Val annotations (65 classes) |
| `dataset/coco/val2017/*.jpg` | images | Val images (COCO val2017, typically 5000) |

Annotation file must list each image’s `file_name` under `dataset/coco/val2017/` (or relative to dataset root).

---

## 4. Optional (only for training)

Not needed for inference; listed so you know they are not required for eval:

- `pretrained_models/clip_convnext_large_trans.pth` — init for training only
- `dataset/coco/annotations/ovd_ins_train2017_b.json` — train set
- `dataset/coco/train2017/*.jpg` — train images

---

## 5. Verify script (run on Linux from project root)

Save as `scripts/check_inference_files.sh` and run: `bash scripts/check_inference_files.sh`

```bash
#!/bin/bash
# Run from project root. Uses DETECTRON2_DATASETS or "dataset" as dataset root.
ROOT="${DETECTRON2_DATASETS:-.}"
if [ "$ROOT" = "." ]; then
  ROOT="dataset"
fi

missing=0

check() {
  if [ -e "$1" ]; then
    echo "  OK   $1"
  else
    echo "  MISS $1"
    missing=1
  fi
}

echo "=== Checkpoint ==="
check "pretrained_models/model_0028399.pth"

echo ""
echo "=== Metadata ==="
check "$ROOT/metadata/ovdcoco_vlm_query_convnextl.npy"
check "$ROOT/metadata/ovcoco_seen_classes.json"
check "$ROOT/metadata/ovcoco_all_classes.json"
check "$ROOT/metadata/ovdcoco_prompts_list8_v2.npy"
check "$ROOT/metadata/ovd_ins_train2017_all_cat_info.json"
check "$ROOT/cluster/ovd_cluster_128.npy"

echo ""
echo "=== Test set ==="
check "$ROOT/coco/annotations/ovd_ins_val2017_all.json"
if [ -d "$ROOT/coco/val2017" ]; then
  n=$(find "$ROOT/coco/val2017" -type f | wc -l)
  echo "  OK   $ROOT/coco/val2017/ ($n files)"
else
  echo "  MISS $ROOT/coco/val2017/"
  missing=1
fi

echo ""
if [ $missing -eq 0 ]; then
  echo "All required files present."
else
  echo "Some files are missing."
  exit 1
fi
```

---

## 6. Plain list (for copy-paste / rsync)

```
pretrained_models/model_0028399.pth
dataset/metadata/ovdcoco_vlm_query_convnextl.npy
dataset/metadata/ovcoco_seen_classes.json
dataset/metadata/ovcoco_all_classes.json
dataset/metadata/ovdcoco_prompts_list8_v2.npy
dataset/metadata/ovd_ins_train2017_all_cat_info.json
dataset/cluster/ovd_cluster_128.npy
dataset/coco/annotations/ovd_ins_val2017_all.json
dataset/coco/val2017/
```

Ensure `dataset/coco/val2017/` contains the actual image files (e.g. 5000 `.jpg` for COCO val2017).
