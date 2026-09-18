# Paired GT-normalization continuation

This is a bounded **training experiment**, not another zero-update audit. It
creates new A/B checkpoints; it never changes the source checkpoint or the
production training defaults. It does not automatically adopt B or start a
longer run.

## Locked comparison

- Start: the same complete no-radius 8ep `model_0056799.pth` identified by the
  completed optimizer-aware audit, including optimizer moments, scheduler and
  AMP scaler. Weights-only initialization is rejected.
- A: existing per-microbatch GT normalization.
- B: pool GT counts over both microbatches. Only the detection classification,
  L1 and GIoU losses (final/aux/encoder/DN) are reweighted. APR/RPSA stay unchanged.
- Budget: default 500 optimizer updates per arm, iterations 56800..57299.
  The scheduler horizon stays **85200**, not 57300; no warmup restart or early
  LR decay. Physical global batch16, accumulation2, four GPUs, seed42.
- **TPA is trainable in both arms**, unlike the previous formula screen. Keep
  K5, tau=.004375, cls_tau=.07, slot prior=.2, no radius, calibrated formula,
  native APR/conflict projection/separate clipping and all optimizer LRs.
- Each arm computes its own clipping coefficients using the original trainer;
  they are not forcibly equalized, since clipping is part of the treatment's
  effect. All moments and weight decay advance normally.
- FedLoss is still independently sampled per microbatch. No shared vocabulary,
  added rare labels, teacher loss, validation-GT training or other intervention.

For global GT totals `n0, n1`, production DDP/criterion clamp produces effective
denominators `di=max(ni,4)`. The audited eight-rank reference denominator is
`D=max(n0+n1,8)`. B multiplies microbatch detection losses by `2*di/D` **before**
the native trainer's accumulation division by two. Empty cases are included.
This changes the normalizer only; it does NOT reproduce all aspects of physical
eight-GPU batch32 (FedLoss, padding and DN grouping remain microbatch-local).

## Pairing and safety checks

Both arms prefetch the two mapped CPU batches before forward so that their
global GT totals are known. Each forward graph is released through the normal
native backward; two GPU graphs are not held simultaneously. Independent RNG
contexts pair mapping and forward seeds without letting model RNG alter the
data stream. All ranks verify image/box/label hashes, FedLoss class lists,
post-forward RNG, LR and AMP scale. Initial **unweighted** losses must agree.
Later features, matching and losses can diverge through learning, as intended.
This is a new paired stream, not a replay of historical sampler state. CUDA
custom kernels may still be nondeterministic.

The full prompt bank is reconstructed on CPU, without changing model mode,
caches, buffers or RNG, at update0, every50 updates, and at the endpoint.
Both all-class and rare-class groups must have mean rank >=4, rank10th percentile
>=3 and mean pairwise cosine <=.8. These are predeclared engineering stop rules,
not performance guarantees and not proof that every individual class avoids
collapse. Minimum rank and number of classes below rank2 are also recorded.
Any failure stops the trial without a success receipt or official evaluation.
No gradient modification or forced projection is added to satisfy the guard.

Skipped AMP optimizer steps, incomplete resume, pairing mismatch, nonfinite
gradients or incomplete evaluation fail closed. New output directories are
required; existing evidence is not overwritten. Every updated model is saved
only under its own trial arm. Source checkpoint, code, assets and reports are
fingerprinted. The validation annotation identity is also recorded.

## Server command

Pull the committed changes using your existing remote/branch workflow first.
Use the `lami` Python interpreter (with compiled detrex). Do **not** pre-create
the trial output directory. The outer log is deliberately outside that directory.

```bash
cd ~/LaMI-DETR
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1,2,3 /root/miniconda3/envs/lami/bin/python -u \
  tools/run_gt_normalization_trial.py \
  --update-audit /root/autodl-tmp/no_radius_accumulation_updates/report.json \
  --output-dir /root/autodl-tmp/no_radius_gt_normalization_ab_500 \
  2>&1 | tee /root/autodl-tmp/no_radius_gt_normalization_ab_500.log
```

This trains A and B sequentially, then runs **two full LVIS evaluations** using
native predicted boxes/CLIP ROI/top300, alpha0/beta.3/novel_scale3, calibrated
Eq.2, mode_scale1. Evaluation can take longer than the short continuations.
Retain both prediction JSONs for any needed paired PR analysis; no further
diagnostic or sweep is launched automatically.

Optional `--prepare-only` checks identities and prepares output without starting
GPU work. To continue that exact prepared run, use the same command with
`--execute-prepared`. This also reuses fully verified completed stages, not a
half-finished training arm whose RNG/data cursor is missing. After failure,
inspect logs and use a new output directory for training; do not delete receipts
to force reuse. `--updates` accepts 1..500 for an explicit smoke test; all flags
must match when executing a prepared run.

Watch progress in another terminal:

```bash
tail -n 30 -F /root/autodl-tmp/no_radius_gt_normalization_ab_500.log
```

## Outputs and decision

- `manifest.json`: exact inputs, restored state digests and locked settings.
- `A/`, `B/`: full checkpoints, per-rank pairing/update logs, completion receipts.
- `A/rank_health.jsonl`, `B/rank_health.jsonl`: all/rare full-bank geometry checks.
- `eval_A/`, `eval_B/`: separate official logs and all-class prediction JSONs.
- `summary.json`: nine AP metrics, B-minus-A, final ranks and scope caveats.

Judge B against **this run's A**, not a previous 500-update baseline with a
different training stream or frozen parameters. Noncollapse is a necessary
screen, not evidence of AP improvement. Compare APr alongside overall AP and
APc/APf; a small one-seed short-run gain alone is not proof of a lasting benefit,
the old model's provenance, or the cause of the 8-to-12ep decline.

Local tests use the real native Trainer methods with a CPU test shell, real
AdamW/AMP state, a four-rank Gloo A/B integration test, plus algebra/pairing/guard/
receipt tests. The regression suite has 143 passing tests, including existing
small CUDA optimizer checks. This environment lacks `fvcore`, so direct full-D2
import tests could not run; neither these tests nor the CPU shell certify server
CUDA custom kernels or full LVIS accuracy before the server run.
