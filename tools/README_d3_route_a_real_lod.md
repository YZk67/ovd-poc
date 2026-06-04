# D3 Route A: Real-LOD / Real-Model

This is the stronger Route-A bridge after the MMDetection GroundingDINO-B
sandbox. The goal is:

```text
1. reproduce the public Real-Model D3 result around 34.1 AP
2. apply the same detector-internal text/query adapter on top of Real-Model
3. test whether the adapter is complementary to Real-LOD
```

Real-LOD is a public MMDetection-style codebase with Real-Model configs,
training scripts, and checkpoints:

- https://github.com/FishAndWasabi/Real-LOD
- https://www.fishworld.site/projects/reallod/index.html

## Setup

On AutoDL, keep Real-LOD outside this repo:

```bash
export REAL_LOD_DIR=/root/autodl-tmp/Real-LOD
git clone https://github.com/FishAndWasabi/Real-LOD.git "$REAL_LOD_DIR"
cd "$REAL_LOD_DIR"
bash install.sh
```

If the install script is too heavy for the current environment, follow
`docs/install.md` in the Real-LOD repo. Their tested stack is Python 3.11,
PyTorch 2.0.1, MMCV 2.0.0rc4, MMEngine 0.7.1, and MMDetection 3.3.0.

Download the Real-Model base checkpoint from the Real-LOD HuggingFace dataset
or use the direct URL supported by the preparation script:

```text
real-model-ckpts/real-model_b-357a96d2.pth
```

## Reproduce Real-Model D3

Run from the LaMI-DETR / ovd-poc repo:

```bash
export REAL_LOD_DIR=/root/autodl-tmp/Real-LOD
export APE_DIR=/root/autodl-tmp/LaMI-DETR-output/d3_ape_b_full
export D3_IMAGE_ROOT=/root/autodl-tmp/APE_datasets/D3/d3_images
export D3_PKL_ROOT=/root/autodl-tmp/APE_datasets/D3/d3_pkl
export D3_FULL_JSON=/root/autodl-tmp/dataset/d3/d3_json/d3_full_annotations.json
export REAL_MODEL_B_CKPT=/root/autodl-tmp/Real-LOD-Data/real-model-ckpts/real-model_b-357a96d2.pth
export REAL_OUT="$APE_DIR/route_a_real_lod_real_model_b_d3_full"

python tools/prepare_real_lod_d3.py \
  --real-lod-root "$REAL_LOD_DIR" \
  --d3-ann-file "$D3_FULL_JSON" \
  --d3-image-root "$D3_IMAGE_ROOT" \
  --d3-pkl-root "$D3_PKL_ROOT" \
  --output-dir "$REAL_OUT" \
  --work-dir "$REAL_OUT/work_dir" \
  --checkpoint "$REAL_MODEL_B_CKPT" \
  --bert-path /root/autodl-tmp/huggingface_models/bert-base-uncased \
  --inference-mode parallel

grep -n "DODCocoMetric\|chunked_size\|test_dataloader" \
  "$REAL_OUT/real_model_swin-b_d3_eval.py"

bash "$REAL_OUT/run_eval.sh"
```

Expected reference:

```text
Real-Model-B D3 full parallel: about 34.1 AP
```

Do not adapt the model until this baseline is reproduced.

## Training Protocol Guard

D3/D-cube is an evaluation benchmark, not an official training split. Any D3
adaptation run must keep training and validation images disjoint. A protocol
audit on 2026-06-04 found that the old Stage-2 full fine-tune commands passed
`$D3_FULL_JSON` to `--d3-ann-file`, `--train-ann-file`, and `--val-ann-file`.
The generated configs therefore used the same file for training and validation:

```text
d3_ann_file = .../d3_full_annotations.json
d3_train_ann_file = .../d3_full_annotations.json
d3_val_ann_file = .../d3_full_annotations.json
```

Those runs are train-on-test and are not paper-facing results. In particular,
discard the old full-D3 Stage-2 full fine-tune numbers such as:

```text
baseline full-ft lr=1e-5: 40.1 AP @500, 41.9 AP @1000
text_negdn full-ft lr=1e-5: 40.0 AP @500, 41.3 AP @1000
```

`tools/prepare_real_lod_d3.py` now refuses training configs whose resolved
`--train-ann-file` and `--val-ann-file` are the same path. Use
`--allow-train-val-same` only for an intentional smoke/debug run.

An image-level random split is also not paper-facing for description-conditioned
generalization. A follow-up audit on 2026-06-04 found that the deterministic
`image_seed42_val20` split had 376 validation positive phrases, of which 374
were already positive in train (`99.47%`). That split is useful only as an
in-domain "new image, mostly old description" diagnostic. Treat numbers such as
the split-val baseline full fine-tune `47.2 AP @500 / 48.9 AP @1000` and
`text_negdn` `47.2 AP @500 / 47.3 AP @1000` as diagnostic, not paper-facing.

For a scenario-held-out Stage-2 protocol, split whole D3 groups from
`groups.pkl`, then filter the COCO JSON by the resulting image ids. Because some
D3 groups include borrowed outer sentences from other groups, record both the
printed `positive_phrase_overlap` and `prompt_level_phrase_overlap` summaries
with every split. The headline novel-description result should use the
`val_prompt_novel` subset; the full group-held-out val is a scenario-adaptation
auxiliary result.

```bash
export D3_SPLIT_DIR=/root/autodl-tmp/dataset/d3/d3_splits/group_seed42_val20
export D3_TRAIN_JSON="$D3_SPLIT_DIR/d3_train_annotations.json"
export D3_VAL_JSON="$D3_SPLIT_DIR/d3_val_annotations.json"
export D3_VAL_NOVEL_JSON="$D3_SPLIT_DIR/d3_val_prompt_novel_annotations.json"

python tools/prepare_d3_group_splits.py \
  --annotation "$D3_FULL_JSON" \
  --pkl-root "$D3_PKL_ROOT" \
  --output-dir "$D3_SPLIT_DIR" \
  --val-ratio 0.2 \
  --test-ratio 0.0 \
  --seed 42

python tools/filter_coco_annotations_by_image_ids.py \
  --annotation "$D3_FULL_JSON" \
  --image-id-jsonl "$D3_SPLIT_DIR/train.jsonl" \
  --output "$D3_TRAIN_JSON"

python tools/filter_coco_annotations_by_image_ids.py \
  --annotation "$D3_FULL_JSON" \
  --image-id-jsonl "$D3_SPLIT_DIR/val.jsonl" \
  --output "$D3_VAL_JSON"

python tools/filter_coco_annotations_by_category_ids.py \
  --annotation "$D3_VAL_JSON" \
  --category-id-jsonl "$D3_SPLIT_DIR/val_prompt_novel_categories.jsonl" \
  --output "$D3_VAL_NOVEL_JSON"
```

All D3 adaptation baselines and ablations must compare on the same held-out
JSON. Report `$D3_VAL_NOVEL_JSON` as the clean novel-description subset and
`$D3_VAL_JSON` as the full group-held-out scenario auxiliary. These numbers are
not directly comparable to the Real-LOD zero-shot full-D3 `34.1 AP` reference.

For `group_seed42_val20`, the positive side is clean but prompt exposure is not:

```text
positive_phrase_overlap.train_val = 0
prompt_level_phrase_overlap.val_positive_seen_in_train_prompt = 50 / 84
prompt_level_phrase_overlap.val_positive_prompt_novel = 34 / 84
```

## Adapter Smoke

Once the Real-Model baseline is stable, run a short adapter smoke. This uses the
same frozen-parameter protocol as the GroundingDINO-B sandbox:

```text
trainable:
  text_query_adapter
  dn_query_generator.label_embedding

frozen:
  everything else
```

```bash
export REAL_ADAPT_OUT="$APE_DIR/route_a_real_lod_real_model_b_d3_full_text_adapter_smoke"

python tools/prepare_real_lod_d3.py \
  --real-lod-root "$REAL_LOD_DIR" \
  --d3-ann-file "$D3_FULL_JSON" \
  --train-ann-file "$D3_TRAIN_JSON" \
  --val-ann-file "$D3_VAL_JSON" \
  --d3-image-root "$D3_IMAGE_ROOT" \
  --d3-pkl-root "$D3_PKL_ROOT" \
  --output-dir "$REAL_ADAPT_OUT" \
  --work-dir "$REAL_ADAPT_OUT/work_dir" \
  --checkpoint "$REAL_MODEL_B_CKPT" \
  --bert-path /root/autodl-tmp/huggingface_models/bert-base-uncased \
  --inference-mode parallel \
  --max-iter 20 \
  --val-interval 20 \
  --checkpoint-interval 20 \
  --train-lr 0.0001

grep -n "RealModelTextQueryAdapter\|TrainableParamFreezeHook\|HungarianAssigner" \
  "$REAL_ADAPT_OUT/real_model_swin-b_d3_adapter_train.py"

bash "$REAL_ADAPT_OUT/run_train.sh"
```

If the smoke reaches validation without errors, scale to `750` iterations and
compare against the reproduced Real-Model baseline:

```bash
export REAL_ADAPT_OUT="$APE_DIR/route_a_real_lod_real_model_b_d3_full_text_adapter_lr1e4_750"

python tools/prepare_real_lod_d3.py \
  --real-lod-root "$REAL_LOD_DIR" \
  --d3-ann-file "$D3_FULL_JSON" \
  --train-ann-file "$D3_TRAIN_JSON" \
  --val-ann-file "$D3_VAL_JSON" \
  --d3-image-root "$D3_IMAGE_ROOT" \
  --d3-pkl-root "$D3_PKL_ROOT" \
  --output-dir "$REAL_ADAPT_OUT" \
  --work-dir "$REAL_ADAPT_OUT/work_dir" \
  --checkpoint "$REAL_MODEL_B_CKPT" \
  --bert-path /root/autodl-tmp/huggingface_models/bert-base-uncased \
  --inference-mode parallel \
  --max-iter 750 \
  --val-interval 250 \
  --checkpoint-interval 250 \
  --train-lr 0.0001

bash "$REAL_ADAPT_OUT/run_train.sh"
```

## E: Text Adapter + Alias-Aware Wrong-Phrase DN

After the DN-label-only control stays at the `34.1` baseline, run E with the
same schedule as the text-adapter run. This keeps the text adapter trainable,
but replaces the denoising negative half with near-same-region boxes paired with
wrong D3 phrases from a different image-local group when available.

```bash
export REAL_NEGDN_OUT="$APE_DIR/route_a_real_lod_real_model_b_d3_full_text_negdn_lr1e4_750"

python tools/prepare_real_lod_d3.py \
  --real-lod-root "$REAL_LOD_DIR" \
  --d3-ann-file "$D3_FULL_JSON" \
  --train-ann-file "$D3_TRAIN_JSON" \
  --val-ann-file "$D3_VAL_JSON" \
  --d3-image-root "$D3_IMAGE_ROOT" \
  --d3-pkl-root "$D3_PKL_ROOT" \
  --output-dir "$REAL_NEGDN_OUT" \
  --work-dir "$REAL_NEGDN_OUT/work_dir" \
  --checkpoint "$REAL_MODEL_B_CKPT" \
  --bert-path /root/autodl-tmp/huggingface_models/bert-base-uncased \
  --inference-mode parallel \
  --max-iter 750 \
  --val-interval 250 \
  --checkpoint-interval 250 \
  --train-lr 0.0001 \
  --adapter-mode text_negdn

grep -n "RealModelTextQueryAdapterNegDN\|negative_dn\|sent_group_ids\|trainable_prefixes\|auto_scale_lr\|optimizer=dict" \
  "$REAL_NEGDN_OUT/real_model_swin-b_d3_adapter_train.py"

bash "$REAL_NEGDN_OUT/run_train.sh"
```

Interpret E against B and C, not only against the public baseline:

```text
A: Real-Model checkpoint eval
B: text adapter + stock DN
C: DN-label-only control
E: text adapter + alias-aware wrong-phrase DN
```

The paper-relevant result is not `28.0` vs `34.1`. It is:

```text
Real-Model-B official checkpoint
Real-Model-B + our detector-internal adapter
```
