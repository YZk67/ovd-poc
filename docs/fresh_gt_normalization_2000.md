# Fresh-start 2,000-update GT-normalization screen

This is a separate experiment from the 8ep continuation. Both arms begin at
iteration zero with seed 42 random detector/TPA weights and load only the same
frozen CLIP ConvNeXt-L backbone. No trained detector, optimizer moments, LR
position or data-loader cursor is inherited.

Run on the server after syncing the commit:

```bash
cd ~/LaMI-DETR
set -o pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3 /root/miniconda3/envs/lami/bin/python -u \
  tools/run_fresh_gt_normalization_trial.py \
  --output-dir /root/autodl-tmp/fresh_gt_normalization_ab_2000 \
  2>&1 | tee /root/autodl-tmp/fresh_gt_normalization_ab_2000.log
```

Do not create the output directory first or tee into it. Monitor with:

```bash
tail -n 30 -F /root/autodl-tmp/fresh_gt_normalization_ab_2000.log
```

The runner executes A and B sequentially, then performs two complete LVIS
evaluations. A uses native per-microbatch GT normalization. B changes only the
detection class/L1/GIoU final, auxiliary, encoder and DN normalization to the
pooled count across the two accumulated microbatches. APR, RPSA, K5 no-radius,
slot prior, conflict routing, clipping and inference remain unchanged.

The script verifies before the first update that A and B have identical model,
empty AdamW, scheduler, AMP scaler, LR-group layout and trainable inventory.
It also verifies mapped inputs, augmentations, global GT counts, FedLoss sampled
classes, forward RNG, LR and AMP state on every rank/microbatch. Full-bank
all/rare rank is checked initially, every 50 updates and at the endpoint.

The LR scheduler retains its original 85,200-update horizon and starts at
iteration zero, including the original short linear warmup. This is not a
resume and does not begin at the 8ep high-LR state.

The final report is
`/root/autodl-tmp/fresh_gt_normalization_ab_2000/summary.json`. Two thousand
updates are only about 0.28 epoch. Treat the result as an early screening signal,
not evidence of final 4ep or 12ep AP/APr. A positive result would still require
a longer matched validation; a negative result should stop this direction.
