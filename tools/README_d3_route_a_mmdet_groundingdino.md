# D3 Route A: MMDetection GroundingDINO-B

This is the Route A bridge from the current ConvNeXt-L LaMI-DETR ablations to a
Swin-B/GDINO-style detector platform. The goal is not to claim that vanilla
GroundingDINO-B is SOTA. The goal is to establish a clean strong-detector
baseline whose backbone and model family are aligned with GDINO, mm-GDINO, and
Real-LOD.

Current status:

- ConvNeXt-L mechanism ablation: `9.3301 -> 20.4842` AP on the strict split.
- Route A platform target: MMDetection `GroundingDINO-B / Swin-B`.
- First Route A milestone: reproduce the D3 GroundingDINO-B baseline before
  adding query/DN adaptation.

OpenMMLab's GroundingDINO README reports D3/DOD intra-scenario results for
GroundingDINO-B:

```text
FULL concat:   20.2
FULL parallel: 25.0
PRES parallel: 23.7
ABS parallel:  28.8
```

Those are the sanity targets for the baseline environment.

GroundingDINO note: `bbox_head.num_classes` is the text-token classification
width, not the D3 phrase count. Keep it at the OpenMMLab default `256`; the
422 D3 phrases enter through `DODDataset` text/positive maps.

Evaluator note: D3 must use `DODCocoMetric`, not plain `CocoMetric`.
`DODDataset` predicts labels as image-local phrase indices, and the DOD metric
maps them back to global D3 sentence ids before COCO-style bbox evaluation.
Plain `CocoMetric` evaluates the wrong category ids and can produce invalid
near-zero AP.

## Setup

On AutoDL, keep MMDetection outside this repo:

```bash
export MMDET_DIR=/root/autodl-tmp/mmdetection
git clone https://github.com/open-mmlab/mmdetection.git "$MMDET_DIR"
cd "$MMDET_DIR"
pip install -U openmim
mim install "mmcv>=2.0.0"
pip install -r requirements/multimodal.txt
pip install "transformers<5" "huggingface-hub<1"
pip install -v -e .
pip install ddd-dataset
```

If HuggingFace access is unavailable, download `bert-base-uncased` elsewhere and
pass `--bert-path /path/to/bert-base-uncased` to the preparation script below.

If editable install fails with a PEP 660 error, use non-editable install:

```bash
pip install -v . --no-build-isolation
```

Keep NumPy below 2 for PyTorch/mmcv compatibility:

```bash
pip install "numpy<2" "opencv-python<4.10" --force-reinstall
pip install "numpy<2" --force-reinstall
```

## Generate The Baseline Config

Run this from the LaMI-DETR repo:

```bash
export MMDET_DIR=/root/autodl-tmp/mmdetection
export APE_DIR=/root/autodl-tmp/LaMI-DETR-output/d3_ape_b_full
export D3_IMAGE_ROOT=/root/autodl-tmp/APE_datasets/D3/d3_images
export D3_PKL_ROOT=/root/autodl-tmp/APE_datasets/D3/d3_pkl
export D3_INTRA_FULL_JSON=/root/autodl-tmp/dataset/d3/annotations/d3_intra_full.json

python tools/prepare_mmdet_groundingdino_d3.py \
  --mmdet-root "$MMDET_DIR" \
  --d3-ann-file "$D3_INTRA_FULL_JSON" \
  --d3-image-root "$D3_IMAGE_ROOT" \
  --d3-pkl-root "$D3_PKL_ROOT" \
  --output-dir "$APE_DIR/route_a_mmdet_gdino_b_d3_full" \
  --work-dir "$APE_DIR/route_a_mmdet_gdino_b_d3_full/work_dir"
```

Then run the generated evaluator:

```bash
bash "$APE_DIR/route_a_mmdet_gdino_b_d3_full/run_eval.sh"
```

If this does not land near the OpenMMLab D3 reference range, fix the environment
or D3 paths before touching model code.

## Adapter Step After Baseline

After baseline reproduction, patch MMDetection rather than this ConvNeXt-L repo:

1. Add a small text/query adapter in the GroundingDINO path that maps phrase text
   features into the query-selection / decoder-conditioning space.
2. Freeze backbone, BERT, image neck, and box regression at first. Train only the
   adapter, matching the stable ConvNeXt-L result where `content_adapter` avoids
   full-detector drift.
3. Add alias-mean and alias-random description conditioning as the first
   ablation. Keep the baseline prompt source and only change the adapter/DN
   training mechanism.
4. Add wrong-phrase same-region negatives after the adapter is stable.

Do not use APE-B, OWLv2, or post-hoc proposal stacking as the main Route A claim.
Those can remain appendix or upper-bound experiments.
