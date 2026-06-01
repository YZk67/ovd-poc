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

Inference-mode note: OpenMMLab reports two D3 modes. `concat` is the default and
concatenates all image-local sub-sentences into one prompt. `parallel` runs
sub-sentences in a loop. The generated parallel config sets
`model.test_cfg.chunked_size=1`, which exercises GroundingDINO's chunked-text
prediction path.

Evaluator note: D3 must use `DODCocoMetric`, not plain `CocoMetric`.
`DODDataset` predicts labels as image-local phrase indices, and the DOD metric
maps them back to global D3 sentence ids before COCO-style bbox evaluation.
Plain `CocoMetric` evaluates the wrong category ids and can produce invalid
near-zero AP. `DODCocoMetric` also has a smaller constructor than
`CocoMetric`; do not pass `metric='bbox'` or `format_only=False`. The generated
config uses `_delete_=True` so those keys do not leak in from the base config.
Keep `sent_ids` in `PackDetInputs.meta_keys`; the metric uses it to map predicted
image-local phrase labels back to global D3 sentence ids.

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
  --work-dir "$APE_DIR/route_a_mmdet_gdino_b_d3_full/work_dir" \
  --inference-mode concat
```

Then run the generated evaluator:

```bash
bash "$APE_DIR/route_a_mmdet_gdino_b_d3_full/run_eval.sh"
```

To reproduce the stronger official parallel baseline, generate a separate
output directory:

```bash
python tools/prepare_mmdet_groundingdino_d3.py \
  --mmdet-root "$MMDET_DIR" \
  --d3-ann-file "$D3_INTRA_FULL_JSON" \
  --d3-image-root "$D3_IMAGE_ROOT" \
  --d3-pkl-root "$D3_PKL_ROOT" \
  --output-dir "$APE_DIR/route_a_mmdet_gdino_b_d3_full_parallel" \
  --work-dir "$APE_DIR/route_a_mmdet_gdino_b_d3_full_parallel/work_dir" \
  --bert-path /root/autodl-tmp/huggingface_models/bert-base-uncased \
  --inference-mode parallel

grep -n "chunked_size\\|DODCocoMetric\\|sent_ids" \
  "$APE_DIR/route_a_mmdet_gdino_b_d3_full_parallel/grounding_dino_swin-b_d3_dod_eval.py"

bash "$APE_DIR/route_a_mmdet_gdino_b_d3_full_parallel/run_eval.sh"
```

If this does not land near the OpenMMLab D3 reference range, fix the environment
or D3 paths before touching model code.

For strict-split method development, also generate a matched strict-val
parallel baseline:

```bash
export STRICT_DIR="$APE_DIR/strict_d3_splits_seed42"
export STRICT_BASE="$APE_DIR/route_a_mmdet_gdino_b_d3_strict_val_parallel"

python tools/prepare_mmdet_groundingdino_d3.py \
  --mmdet-root "$MMDET_DIR" \
  --d3-ann-file "$STRICT_DIR/d3_strict_val_annotations.json" \
  --d3-image-root "$D3_IMAGE_ROOT" \
  --d3-pkl-root "$D3_PKL_ROOT" \
  --output-dir "$STRICT_BASE" \
  --work-dir "$STRICT_BASE/work_dir" \
  --bert-path /root/autodl-tmp/huggingface_models/bert-base-uncased \
  --inference-mode parallel \
  --use-subset-dataset

bash "$STRICT_BASE/run_eval.sh"
```

## Adapter Training Step

The first Route-A method step is a conservative description-conditioned query
adapter on the GroundingDINO-B platform. It keeps the official checkpoint and
freezes the visual backbone, BERT, neck, decoder, and box heads. The trainable
surface is:

```text
text_query_adapter
dn_query_generator.label_embedding
```

`text_query_adapter` is a zero-initialized residual MLP applied to projected
text features before GroundingDINO's multimodal encoder. Because it is
zero-initialized, the starting point is the reproduced `25.0` parallel baseline.

Generate a strict train/val adapter config:

```bash
export STRICT_DIR="$APE_DIR/strict_d3_splits_seed42"
export ADAPT_OUT="$APE_DIR/route_a_mmdet_gdino_b_d3_strict_text_adapter"

python tools/prepare_mmdet_groundingdino_d3.py \
  --mmdet-root "$MMDET_DIR" \
  --d3-ann-file "$STRICT_DIR/d3_strict_val_annotations.json" \
  --train-ann-file "$STRICT_DIR/d3_strict_train_annotations.json" \
  --val-ann-file "$STRICT_DIR/d3_strict_val_annotations.json" \
  --d3-image-root "$D3_IMAGE_ROOT" \
  --d3-pkl-root "$D3_PKL_ROOT" \
  --output-dir "$ADAPT_OUT" \
  --work-dir "$ADAPT_OUT/work_dir" \
  --bert-path /root/autodl-tmp/huggingface_models/bert-base-uncased \
  --inference-mode parallel \
  --max-iter 1000 \
  --val-interval 500 \
  --checkpoint-interval 500 \
  --train-lr 0.0001

grep -n "GroundingDINOTextQueryAdapter\|text_query_adapter\|TrainableParamFreezeHook\|chunked_size" \
  "$ADAPT_OUT/grounding_dino_swin-b_d3_dod_adapter_train.py"
```

Run a short smoke first by replacing `--max-iter 1000` with `--max-iter 20`
and `--val-interval 20`. Once that passes:

```bash
bash "$ADAPT_OUT/run_train.sh"
```

Evaluate the saved adapter checkpoint on strict val:

```bash
cd "$MMDET_DIR"
export PYTHONPATH=/root/LaMI-DETR:${PYTHONPATH:-}

python tools/test.py \
  "$ADAPT_OUT/grounding_dino_swin-b_d3_dod_adapter_train.py" \
  "$ADAPT_OUT/work_dir/iter_1000.pth" \
  --work-dir "$ADAPT_OUT/eval_iter1000"
```

This strict split run is for method development and should beat the matched
strict GroundingDINO-B parallel baseline before running larger final comparisons.

## Adapter Roadmap

After baseline reproduction, patch MMDetection rather than this ConvNeXt-L repo:

1. Train the zero-initialized text/query adapter above and confirm it does not
   regress the `25.0` parallel baseline.
2. Add alias-mean and alias-random description conditioning as the first
   ablation. Keep the baseline prompt source and only change the adapter/DN
   training mechanism.
3. Add wrong-phrase same-region negatives after the adapter is stable.

Do not use APE-B, OWLv2, or post-hoc proposal stacking as the main Route A claim.
Those can remain appendix or upper-bound experiments.
