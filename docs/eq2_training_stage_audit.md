# 8ep / 12ep Eq.2 training-formula audit

This is a bounded diagnostic, **not a training experiment or an AP estimate**.
It compares the no-radius K5 checkpoints at iterations 56799 and 85199.
The recorded no-radius APr values, 42.8843 and 42.4229, differ by -0.4614;
this audit does not assume that the classification formula caused that change.

## Run on the server

```bash
cd ~/LaMI-DETR
set -o pipefail
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/lami/bin/python -u \
  tools/audit_eq2_training_stages.py \
  --checkpoint-8ep /root/autodl-tmp/instructdet_k5_no_radius_bs32_4ep_seed42/model_0056799.pth \
  --checkpoint-12ep /root/autodl-tmp/instructdet_k5_no_radius_bs32_4ep_seed42/model_final.pth \
  --output-dir /root/autodl-tmp/no_radius_eq2_8ep_vs_12ep \
  2>&1 | tee /root/autodl-tmp/no_radius_eq2_8ep_vs_12ep.log
```

One GPU, 4 batches of 2 images **per endpoint**, at most 16 image exposures
by default. Both stages use the same sampled training images, augmentations,
FedLoss categories and seeded stochastic forwards. They are not a replay of
the historical batches or the distributed effective-batch-32 optimizer.
All sampled classes and all their negative query/category cells are included;
no classes are selected using validation performance.

The two checkpoint files and the lami runtime/assets must be available. Use a
separate output directory. Feature caches may occupy several hundred MB or
more, depending on DN padding; preserve them to avoid repeating GPU capture.
Each completed stage has a hashed receipt. An interrupted, incomplete stage
may be recaptured, but a completed, unchanged stage is reused. Changed source
files/checkpoints or budget require a fresh output directory.

After both captures finish, `--analyze-only` with the same arguments reruns just
the detached-feature analysis on CPU, without constructing a detector. It fails
if caches are missing; it never silently starts GPU inference. `--capture-only`
is available to separate capture from analysis. Do not change `--device` when
reusing a manifest; `--analysis-device cpu` controls only the replay device.

## Controlled quantities and formulas

Within **each** endpoint, freeze its own native training-mode query features
(after the classifier's linear projection), raw shared TPA output slots, boxes,
sampled categories and target assignments. Do not transplant features between
endpoints or recompute the decoder/query initialization under an alternate formula.

For cosine similarities `u_k`, classification scale `s`, temperature `tau`:

- `calibrated`: `s*tau*(logsumexp(u/tau)-log(K))` — current native training.
- `calibrated_plus_logK`: calibrated plus `log(K)` — algebraic bias control.
- `legacy`: `logsumexp(s*u)` — historical formula, not historical training replay.

The third formula changes both the additive offset and aggregation sensitivity
when modes differ. With identical modes, the difference reduces to `log(K)`.
This audit never changes the training/inference implementation or saved weights.

## Outputs and limits

`report.json` contains per batch/head/category results and stage summaries:

- Positive/negative focal loss, logit gradients, partial input-feature gradients,
  and raw prototype-output gradients, including per-category slot allocation.
- Final, auxiliary, encoder and DN heads kept separate. Slot allocation is the
  allocation of **head-output partial derivatives**, not gradients of TPA weights.
- Primary comparison with fixed native assignment; secondary comparison with
  Hungarian rematching, holding boxes fixed. DN retains its known assignments.
- Native logits, weighted focal losses and Hungarian matches must reconstruct;
  numeric/tie mismatches stop the audit rather than being called formula effects.
- Formula effects at each stage and their stage differences. Summary averages
  are equally weighted batch/head statistics, **not** joint parameter-gradient norms.

“Negative” here means the training label is zero. It does **not** mean an LVIS
false positive; `train_norare` provides no direct rare-positive supervision.
Neither frozen feature gradients nor their stage differences prove why rare AP
fell, identify the actual historical update responsible, or predict that switching
formulas will improve AP. APR, RPSA, optimizer moments, gradient conflict
projection, clipping, and formula-dependent query/box changes are outside this
partial-derivative experiment. There is no validation GT use or automatic next
training/evaluation step. A null/inconsistent result is a reason to stop this
formula hypothesis, not to launch another parameter sweep.
