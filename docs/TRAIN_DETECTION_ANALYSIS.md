# Train Detection Issue - Analysis & Debugging Guide

## Problem
When running `vis_airplane.py` on train images, bounding boxes for "train" are not identified correctly.

## Verification Results

Run the class order verification:
```bash
python tools/verify_class_order.py
```

If all components match OVDCOCO65, the mapping is correct. Current check shows:
- MetadataCatalog.thing_classes ✓
- ovcoco_all_classes.json ✓
- ovdcoco_prompts_list8_v2.npy shape (65, 8, 768) ✓

## Possible Causes

### 1. **Low score threshold → noisy predictions**
Config has `model.test_score_thresh = 0.01` (very low). Many low-confidence boxes may be wrong.

**Fix**: Use a higher threshold when visualizing:
```bash
python tools/vis_airplane.py \
  --config lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py \
  --weights /root/autodl-tmp/pretrained_models/model_0028399.pth \
  --input dataset/coco/train2017 \
  --output ./vis_output \
  --score-thresh 0.3 \
  --class-name train \
  --debug
```

### 2. **Confusion with similar classes**
"train", "bus", "airplane" can be visually similar. Model might predict bus/airplane for trains.

From your metrics: train AP=68.6, bus AP=17.9, airplane AP=9.2. Train is detected well overall, but confusions can occur on specific images.

**Debug**: Run with `--debug` to see raw pred_classes:
```bash
python tools/vis_airplane.py ... --debug
```
Output: `top_classes=[6, 2, 4, ...]` — index 6 = train, 2 = car, 4 = airplane, etc.

### 3. **Input images: train vs val**
- **Training set** (`dataset/coco/train2017`): Only 48 base classes. "train" is base.
- **Val set** (`dataset/coco/val2017`): All 65 classes.

Ensure you're using images that actually contain trains. Try a few val images that have train annotations:
```bash
# Example: visualize on val images
python tools/vis_airplane.py \
  --config ... --weights ... \
  --input dataset/coco/val2017 \
  --output ./vis_val \
  --score-thresh 0.25
```

### 4. **Text embedding order (if verification fails)**
If `verify_class_order.py` shows MISMATCH for the prompts file, regenerate:
```bash
# Ensure prompts JSON has OVDCOCO65 order, then:
python tools/generate_text_embeddings.py \
  --prompt-json dataset/metadata/ovdcoco_prompts_list8.json \
  --clip-model ... \
  --output dataset/metadata/ovdcoco_prompts_list8_v2.npy
```

### 5. **Visualize all classes for one image**
To see what the model predicts without filtering:
```bash
python tools/vis_airplane.py ... --input path/to/single_image.jpg --output ./out
# Omit --class-name to see all detections
```

## Recommended Debug Commands

```bash
cd /root/ovd-poc
conda activate lami

# 1. Verify class order
python tools/verify_class_order.py

# 2. Visualize with higher threshold + debug
python tools/vis_airplane.py \
  --config lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py \
  --weights /root/autodl-tmp/pretrained_models/model_0028399.pth \
  --input dataset/coco/val2017 \
  --output ./vis_debug \
  --score-thresh 0.25 \
  --class-name train \
  --limit 10 \
  --debug
```

Check the debug output: `top_classes` values. Index 6 = train in OVDCOCO65.
