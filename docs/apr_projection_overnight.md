# APR / conflict-projection: unattended two-round experiment

This runs **three arms, each for 2,000 optimizer updates**, from the **same full
no-radius 8ep checkpoint** (iteration 56799). It does not train one arm from
another arm's endpoint.

| Arm | Directional barrier coefficient | Usage balance coefficient | Conflict projection |
| --- | ---: | ---: | --- |
| A | 0.1 | 0.03 | On |
| P | 0.1 | 0.03 | Off |
| R | 0 | 0.03 | Off |

Round 1 is **P minus A**: disabling projection with the full APR loss retained.
Round 2 is **R minus P**: disabling only the barrier, conditional on projection
already being disabled. P is shared, so four training runs are not needed.
This is not a full 2x2 interaction experiment or a reconstruction of the old
model's training history.

## Locked protocol and safety

- Restore the same model, AdamW moments, scheduler and AMP scaler in all arms.
  Existing optimizer momentum is retained, including its historical APR effects.
- Four GPUs, physical global batch 16, accumulation 2, effective batch 32.
  Use native per-microbatch GT normalization, **not pooled-GT normalization**.
- Original LR horizon 85200; continuation runs from iteration 56800 to 58799.
  No warmup restart or shorter-horizon decay.
- Keep TPA trainable, K=5, tau=0.004375, slot prior=0.2, no radius, and the same
  calibrated training/inference formulas, prompts, FedLoss and RPSA settings.
- Native routing, separate gradient clipping and AdamW are used. Each arm uses
  its own resulting clip coefficients; these are not forcibly equalized.
- Verify image/augmentation hashes, sampled categories, RNG consumption, LR and
  AMP scale across arms. Before any update, non-APR losses and both unweighted
  APR terms must agree. Later matching/features can naturally diverge.
- This is a new paired data stream, not a historical sampler replay. CUDA
  custom kernels are not promised to be bitwise deterministic.
- Check the full prompt bank at update 0, every 50 updates and the endpoint:
  both all-class and rare-class mean rank >=4, rank p10 >=3 and mean cosine
  <=0.8. Also report minimum rank and counts below 2. These engineering guards
  **do not guarantee that every individual class stays noncollapsed**.
- A rank-guard failure, nonfinite value, skipped optimizer update, pairing
  mismatch or incomplete evaluation stops the pipeline. No automatic retry,
  threshold adjustment, arm substitution or adoption of a new training recipe.
- Train/evaluate A, then train/evaluate P and save round 1, then train/evaluate R.
  Round 1 remains available if R fails. Preserve checkpoints, prediction JSONs,
  per-rank pairing logs, optimizer-update logs and rank checks.
- Save an intermediate checkpoint at update 500 (`model_0057299.pth`) and the
  2,000-update `model_final.pth`. Only the endpoint is evaluated automatically;
  the earlier +APr-at-500 / -APr-at-2000 reversal is why 500 is not the verdict.

The reference directory below supplies only its verified original 8ep inputs
and protocol, **not its A/B 500-update endpoints or normalization intervention**.
Keep that directory's `manifest.json`, its original source checkpoint, and the
referenced assets. Source hashes are checked; do not edit model/training code
while this experiment runs.

## Launch once on the server

Use the `lami` environment's explicit Python path. Pull the committed scripts
first. Leave the four selected GPUs free. The output directory must be NEW;
do not run `mkdir` on it or redirect the outer log inside it.

```bash
cd ~/LaMI-DETR

nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /root/miniconda3/envs/lami/bin/python -u tools/run_apr_projection_trial.py \
  --reference-trial /root/autodl-tmp/no_radius_gt_normalization_ab_500 \
  --output-dir /root/autodl-tmp/no_radius_apr_projection_2000 \
  --updates 2000 \
  > /root/autodl-tmp/no_radius_apr_projection_2000.log 2>&1 < /dev/null &

echo "PID=$!"
```

This is 6,000 optimizer updates plus **three full LVIS evaluations**, checkpoint
I/O and verification. Completion by morning cannot be guaranteed. Disk preflight
requires room for six full checkpoints plus 6 GiB for predictions/logs; source
files are never deleted. Nondefault CPU-thread settings must match the reference.

Watch progress:

```bash
tail -n 40 -F /root/autodl-tmp/no_radius_apr_projection_2000.log
```

In the morning:

```bash
cat /root/autodl-tmp/no_radius_apr_projection_2000/STATUS.json
cat /root/autodl-tmp/no_radius_apr_projection_2000/results.txt
```

`STATUS.json` reports `RUNNING`, `COMPLETE` or `FAILED` with a phase such as
`train_P` / `evaluate_R`. `results.txt` contains AP/APr, rank and the two contrasts.
`summary.json` retains detailed metrics and evidence. If preparation failed before
creating the directory, read the outer `.log`. Abrupt machine termination can
leave a stale `RUNNING` status; check the PID too.

Optional preflight: add `--prepare-only` to validate/create the manifest without
GPU work, then rerun with the same options plus `--execute-prepared` (without
`--prepare-only`). `--execute-prepared` can reuse only complete, verified stages;
it refuses to overwrite or resume an incomplete training/evaluation stage.

## Interpretation and checkpoint policy

Compare official AP/APr and rank, not local gradient angles or training loss.
A single seed and one continuation window are a screen, not proof of a durable
gain or the cause of the historical 8ep-to-12ep APr change. A failed rank guard
means that intervention is unsafe under this protocol, not an AP result.

The R arm changes `lambda_orth_base` at runtime before any update. It is a Python
loss coefficient, not a state-dict tensor; the policy is explicitly recorded in
the manifest, receipts, pairing logs and checkpoint trainer metadata. Do **not**
resume R with ordinary `train_net.py` and assume it retains this coefficient.
Evaluation is safe because APR coefficients are not part of inference. The
pipeline does not launch any follow-up training automatically.
