# Gradient accumulation: conditional objective audit

This is the next **diagnostic**, not a training fix. It does not change the
training code/config, save model weights, create an optimizer or evaluate AP.
In particular, it does not claim that changing accumulation will improve APr.

## Bounded server command

Use the lami Python and four GPUs, matching the current physical batch layout.
The default uses **two independent windows** from the same full no-radius 8ep
checkpoint. Each window has 32 mapped training images and runs two category
policies: 64 image exposures per window, 128 total. It performs two parameter
gradient probes per forward. This is not 500 updates and not a validation pass.
There is no local GPU validation in the development environment; test the
script on the server and preserve any failing output rather than changing the
training configuration to bypass a validation error.

```bash
cd ~/LaMI-DETR
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1,2,3 /root/miniconda3/envs/lami/bin/python -u \
  tools/audit_accumulation_objective.py \
  --checkpoint /root/autodl-tmp/instructdet_k5_no_radius_bs32_4ep_seed42/model_0056799.pth \
  --num-gpus 4 --windows 2 \
  --output-dir /root/autodl-tmp/no_radius_accumulation_objective \
  2>&1 | tee /root/autodl-tmp/no_radius_accumulation_objective.log
```

Use a new empty output directory. The log is deliberately **outside** it.
No automatic overwrite, training continuation or fallback inference occurs.
Upload `no_radius_accumulation_objective/report.json` after `COMPLETE.json`
appears. Window receipts are saved incrementally; incomplete output is not a
completed audit. `tail -F /root/autodl-tmp/no_radius_accumulation_objective.log`
shows progress, including rank, window and policy.

## The four conditional objectives

| Label | FedLoss categories | Detection GT normalization |
|---|---|---|
| native_micro | Native sample per physical microbatch | Native per microbatch |
| native_pooled | Same native forwards/matching | Pooled over 32 images |
| shared_micro | One sampled vocabulary containing the full window GT union | Native per microbatch |
| shared_pooled | Same shared-category forwards/matching | Pooled over 32 images |

For global microbatch GT counts `G1,G2`, four current ranks, and a declared
eight-rank reference, native effective denominators after DDP averaging are
`max(Gm,4)`. The pooled reference denominator is `max(G1+G2,8)`.
Since accumulation already divides loss by two, multiply each microbatch's
detection loss by `2*max(Gm,4)/max(G1+G2,8)`. The clamp matters for empty or
very sparse batches; do not replace this with a formula assuming positive GT.

The rescaling covers final/auxiliary/encoder/DN classification and L1/GIoU.
DN keeps its native local group count; only the GT denominator changes.
APR and RPSA are **not** multiplied by the GT correction. The audit checks the
actual criterion denominators against this algebra before accepting output.

Shared FedLoss is sampled once from all 32 images' GT union, using the native
frequency weights and class budget. Every physical forward still consumes its
native sampler's RNG first, in both policies; only then are the selected IDs
and global-to-local labels replaced. Inputs are deep-copied because DINO remaps
labels in place. The audit verifies pairing of all four ranks' mapped inputs,
native draws, downstream RNG states and DN shapes, and observes Hungarian
assignment changes. If the GT union exceeds the category budget it stops.

## How to interpret the report

`normalization` records GT counts, denominators and exact loss multipliers.
`comparisons` contains gradient norms, relative L2 differences and cosine for
all trainable parameters and each module group, before APR routing/clipping.
Zero-gradient cosine is undefined (`null`), not perfect agreement.

Read the two one-factor comparisons first:

1. `normalization_with_native_categories`: identical forward tensors and
   assignments, only scalar normalization changed. This isolates the local
   objective's GT-weighting effect.
2. `categories_with_micro_normalization`: paired images/RNG and unchanged
   normalization, but the full forward is recomputed with shared categories.
   This includes query initialization, TPA, APR/RPSA, DN label semantics and
   matching, not merely the final classification loss's negative mask.

The other two comparisons give the reverse order; `both_changes_vs_native`
must not be called the sum of independent effects. Do not infer benefit from a
large gradient change, or declare equivalence from two small sampled windows.

## What this does not reproduce

- No physical eight-GPU forward, AMP scaling, optimizer moments, APR conflict
  projection, gradient clipping, update or LR step is performed. Gradients are
  FP32, manually rank-averaged; they are not claims of bitwise native DDP parity.
- DN grouping and image padding remain local to four images, compatible with
  viewing two four-rank microbatches as eight *virtual* four-image ranks, but
  this does not recreate historical dropout streams, collectives or data order.
- Active training BatchNorm is rejected rather than silently using different
  statistics or turning it off. Parameter/persistent-buffer state and source
  checkpoint identities must remain unchanged. Diagnostic caches may change.
- This is one checkpoint and a small sample of train_norare batches. It does
  not establish a rare-class benefit, late-training causation or the old
  checkpoint's unknown training provenance. Validation GT is never used.
- Shared TPA long-term drift remains a separate question. No follow-up
  training or additional sweep is launched automatically.
