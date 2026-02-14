#!/bin/bash
# Check all files required for inference (dino_convnext_large_4scale_12ep_lvis.py).
# Run from project root:  bash scripts/check_inference_files.sh

set -e
ROOT="${DETECTRON2_DATASETS:-dataset}"

check() {
  if [ -e "$1" ]; then
    echo "  OK   $1"
  else
    echo "  MISS $1"
    return 1
  fi
}

missing=0

echo "=== Checkpoint ==="
check "pretrained_models/model_0028399.pth" || missing=1

echo ""
echo "=== Metadata ==="
check "$ROOT/metadata/ovdcoco_vlm_query_convnextl.npy" || missing=1
check "$ROOT/metadata/ovcoco_seen_classes.json" || missing=1
check "$ROOT/metadata/ovcoco_all_classes.json" || missing=1
check "$ROOT/metadata/ovdcoco_prompts_list8_v2.npy" || missing=1
check "$ROOT/metadata/ovd_ins_train2017_all_cat_info.json" || missing=1
check "$ROOT/cluster/ovd_cluster_128.npy" || missing=1

echo ""
echo "=== Test set ==="
check "$ROOT/coco/annotations/ovd_ins_val2017_all.json" || missing=1
if [ -d "$ROOT/coco/val2017" ]; then
  n=$(find "$ROOT/coco/val2017" -type f 2>/dev/null | wc -l)
  echo "  OK   $ROOT/coco/val2017/ ($n files)"
else
  echo "  MISS $ROOT/coco/val2017/"
  missing=1
fi

echo ""
if [ $missing -eq 0 ]; then
  echo "All required inference files are present."
else
  echo "Some required files are missing."
  exit 1
fi
