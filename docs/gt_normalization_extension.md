# Extend the paired GT-normalization trial

This extends the completed 500-update A/B to **2,000 total optimizer updates
per arm**, adding 1,500 updates to each arm. It does not restart from 8ep or
automatically train to 12ep. A uses native per-microbatch GT normalization;
B uses the same pooled-GT normalization tested in the first 500 updates.

## Server command

After syncing the new code, use the same `lami` environment, assets and four GPUs
as the completed trial. Retain its whole output directory, including both full
checkpoints, all-rank pairing logs, completion receipts and evaluation JSONs.
Do **not** pre-create the new output directory or redirect a log inside it.

```bash
cd ~/LaMI-DETR
set -o pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3 /root/miniconda3/envs/lami/bin/python -u \
  tools/extend_gt_normalization_trial.py \
  --parent-dir /root/autodl-tmp/no_radius_gt_normalization_ab_500 \
  --output-dir /root/autodl-tmp/no_radius_gt_normalization_ab_2000 \
  --total-updates 2000 \
  2>&1 | tee /root/autodl-tmp/no_radius_gt_normalization_ab_2000.log
```

In a second terminal:

```bash
tail -n 30 -F /root/autodl-tmp/no_radius_gt_normalization_ab_2000.log
```

The runner verifies the parent experiment, continues A, continues B, and runs
two full LVIS evaluations with the original inference settings. It writes
`summary.json` in the new directory with endpoint metrics, B-minus-A, changes
from the 500-update endpoints, and all/rare prototype health. Existing results
are not overwritten. No parameter sweep is performed.

## Continuation contract

- Each arm restores **its own** 500-update model, AdamW moments, scheduler and
  AMP scaler, with exact state checks; A's state is not substituted for B's.
- Training resumes at iteration 57300 and stops after iteration 58799. The
  original LR horizon stays **85200**. This changes the stopping point, not
  the LR schedule, and does not restart warmup.
- The original physical batch16, accumulation2, per-microbatch FedLoss
  sampling, no-radius K5, slot prior, APR/RPSA, conflict projection, clipping
  and calibrated inference settings remain unchanged. TPA remains trainable.
- The original loader did not checkpoint its cursor. Each arm first replays
  **only the first 500 data windows**, with the original mapping seeds, to
  reconstruct sampler/aspect-ratio grouping state. It verifies every mapped
  image/GT hash and global GT count against that arm's saved transcript.
  This costs data loading/augmentation time, but performs **no model forward,
  backward or optimizer update**. A mismatch aborts before new updates.
- New forwards use the original cumulative seed timeline. A/B input hashes,
  sampled category lists, RNG records, LR and AMP scales are checked as in the
  original trial. Model outputs/matching may naturally diverge after updates.
- The full-bank all/rare rank guard runs on the restored endpoint, every 50
  updates and at the final endpoint. A guard failure or skipped optimizer step
  aborts rather than declaring the run complete. These engineering checks
  monitor geometry; they do not guarantee every class has useful prototypes.
- This is a controlled continuation, **not a promise of bitwise CUDA identity**
  with an uninterrupted run. Tests verify uninterrupted-vs-resumed small-model
  training, including four-rank DDP; the server's custom CUDA kernels still
  have their original determinism limitations.

For validation only, append `--prepare-only` (no training/evaluation launched).
Then repeat the same command with `--execute-prepared`. That flag can also skip
intact, verified completed stages. An interrupted **partial** training or
evaluation directory is rejected: inspect and retain it, then use a fresh
output directory. This tool does not silently resume a partial arm with an
unknown data cursor or overwrite existing evidence.

## Interpretation

Compare B against A at the same 2,000-update endpoint, including AP, APr,
APc/APf and all/rare rank. Compare each arm with its own 500-update endpoint
separately. The original +0.9639 APr / -0.0826 AP result is a screening result,
not a guaranteed long-term gain. This continuation does not establish the
cause of the historical old/new model gap or the 8-to-12ep APr change.
