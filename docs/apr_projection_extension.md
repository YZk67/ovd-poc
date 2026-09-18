# Extend A/P to 4,000 cumulative updates

This continues only A and P from the completed 2,000-update APR experiment.
Each arm performs **2,000 new optimizer updates**, reaching **4,000 total** since
the shared 8ep starting checkpoint. R is not resumed or evaluated.

| Arm | Barrier | Balance | Conflict projection |
| --- | ---: | ---: | --- |
| A | 0.1 | 0.03 | On |
| P | 0.1 | 0.03 | Off |

The existing result (P-A: +0.8167 APr, +0.1249 AP) is a screening result, not a
promise that the advantage persists. The runner reads the actual parent metrics
instead of hardcoding them, then reports P-A at both horizons and each arm's
change from its own 2,000-update endpoint.

## One server command

Sync the committed code first. Use the same `lami` environment, assets and four
GPUs as before. Retain the original trial directory, including `manifest.json`,
`summary.json`, A/P full checkpoints, all-rank pairing/update logs, completion
receipts and evaluation outputs. The original runner's verified reference
manifest must still be present. Do not edit model/training code during the run.

The NEW output directory must not exist. Do not pre-create it or put the outer
log inside it. Existing parent checkpoints/results are never overwritten.

```bash
cd ~/LaMI-DETR

nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /root/miniconda3/envs/lami/bin/python -u tools/extend_apr_projection_trial.py \
  --parent-dir /root/autodl-tmp/no_radius_apr_projection_2000 \
  --output-dir /root/autodl-tmp/no_radius_apr_projection_4000 \
  --total-updates 4000 \
  > /root/autodl-tmp/no_radius_apr_projection_4000.log 2>&1 < /dev/null &

echo "PID=$!"
```

Monitor:

```bash
tail -n 40 -F /root/autodl-tmp/no_radius_apr_projection_4000.log
```

Read status and results:

```bash
cat /root/autodl-tmp/no_radius_apr_projection_4000/STATUS.json
cat /root/autodl-tmp/no_radius_apr_projection_4000/results.txt
```

Upload the new `summary.json` after completion. `results.txt` shows both the
2,000- and 4,000-update results. `STATUS.json` distinguishes `RUNNING`, `COMPLETE`
and `FAILED` (with phase/error). Preparation failures before directory creation
are in the outer log; a power failure may leave a stale `RUNNING` status.

## Continuation and pairing guarantees

- A restores **A2000**, P restores **P2000**, including each arm's model, AdamW
  moments, scheduler and AMP scaler. Their post-training states should no longer
  be identical; each must exactly match its own certified parent state.
- Resume at iteration **58800**, stop after **60799**. LR horizon stays **85200**;
  no warmup restart or shortened LR schedule. Do not override LR or accumulation.
- Preserve physical global batch16, accumulation2, calibrated formulas, no-radius
  K5, slot prior, full APR, FedLoss, RPSA, native microbatch GT normalization,
  per-arm clipping and trainable parameter scope. No pooled-GT intervention.
- The original loader did not checkpoint its cursor. Before new updates, read
  and hash-check the first **2,000 DATA windows** with their original seeds, so
  private sampler and aspect-ratio grouping state reach the next window. This
  costs decoding/augmentation time but performs **zero model forwards, backward
  passes or optimizer updates**. It is not retraining on old batches.
- Check model/optimizer/scheduler/scaler digests before and after data-only replay.
  Subsequent data/forward seeds use the cumulative update index; A/P input hashes,
  FedLoss selections, RNG consumption, LR and AMP scale remain paired. Naturally
  divergent model outputs and matching are not forced equal.
- Run the unchanged full-bank all/rare rank guard at restored update2000, every50
  updates and at update4000. Initial geometry must match the saved parent check.
  Guard thresholds remain mean rank >=4, p10 rank >=3 and mean cosine <=0.8 for
  both all and rare; also log minimum rank/counts below2. These are engineering
  checks, not a guarantee that every prototype has useful semantics.
- Fail closed on replay mismatch, nonfinite values, skipped updates, wrong state,
  guard failure or incomplete evaluation. No automatic retry/retuning.
- Train/evaluate A, then train/evaluate P. Preserve partial A results if P fails.
  Retain each new final checkpoint and both full LVIS prediction JSONs. This is
  4,000 NEW updates across two arms, two evaluations, replay and verification I/O;
  completion time depends on the server. No further training is launched.
- Small-model tests compare uninterrupted versus resumed native updates (including
  four-rank DDP). They do not guarantee bitwise determinism of the server's custom
  CUDA kernels or statistical reproducibility across independent training runs.

For preflight only, add `--prepare-only`; then use the same command with
`--execute-prepared` instead. That flag reuses only intact, verified completed
stages. Incomplete training/evaluation stages are rejected, not silently
overwritten or resumed without a known cursor.

This remains a single-seed continuation. Positive P-A at 4,000 would support
persistence over a longer window, not prove the complete cause of the historical
old-model gap or justify a new 12ep run automatically.
