# GT-normalization: optimizer-aware, zero-live-update audit

This tests **only** microbatch GT normalization versus pooled-window GT
normalization. It is not a fix, performance evaluation or continuation run.
FedLoss still draws its native independent vocabulary per physical microbatch.
No change to training/model/config source files is required.

## Run after synchronizing the new scripts

Use the completed **v2** objective report. Its checkpoint, code, assets, dataset,
windows, mapper inputs, seeds, sampled categories, matching, losses and optimizer
identity are checked, not silently substituted. Use the same four-GPU lami
environment and an empty output directory. The log is outside that directory.

```bash
cd ~/LaMI-DETR
set -o pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3 /root/miniconda3/envs/lami/bin/python -u \
  tools/audit_accumulation_updates.py \
  --objective-audit /root/autodl-tmp/no_radius_accumulation_objective_v2/report.json \
  --num-gpus 4 \
  --output-dir /root/autodl-tmp/no_radius_accumulation_updates \
  2>&1 | tee /root/autodl-tmp/no_radius_accumulation_updates.log
```

In another terminal:

```bash
tail -F /root/autodl-tmp/no_radius_accumulation_updates.log
```

The two-window reference causes **64 training-image forwards total** (not 128):
each microbatch is forwarded once and differentiated for two total objectives
and an APR reference. Rank zero additionally runs four one-step AdamW probes on
temporary parameter/optimizer copies. All ranks reproduce native APR routing
and separate clipping. This is not 500 optimizer updates or LVIS inference.
No GPU/runtime duration is promised; temporary same-device copies require
additional GPU memory after the forward graphs are released.

If a preflight/pairing/nonfinite check fails, preserve the error and use a new
output directory after diagnosis. Do not remove checks, change training knobs,
or reuse a partially populated directory. Existing code/config fingerprints
must match the earlier audit; these new scripts add files only.

Upload `no_radius_accumulation_updates/report.json` after `COMPLETE.json`
appears. Partial per-window JSONs are receipts, not a completed audit.

## What is held fixed

- Full 8ep endpoint at iteration 56799, live parameters and persistent buffers.
- Restored native AdamW moments, per-parameter steps, groups, learning rates,
  betas, epsilon and weight decay. No LR schedule advancement/restart.
- Input mapping, FedLoss, dropout, DN, forward outputs and matching. Both
  objectives use **the same graph**, with no second candidate forward.
- APR/RPSA weights and APR gradient. Detection class/L1/GIoU (including final,
  auxiliary, encoder and DN) alone receive the existing pooled-GT multipliers.
- AMP scale restored from the checkpoint. The model forward is FP32 as in the
  native trainer; total gradients are scaled, accumulated, rank-averaged and
  unscaled with the native GradScaler. APR is differentiated unscaled.

The script calls the actual `Trainer._route_tpa_gradients` and
`Trainer.clip_model_grads` methods. Each arm recomputes its own detector/TPA
clipping coefficients; using A's coefficient for B would be a different
intervention. APR synchronization occurs exactly once in the native routing
method. Global gradient presence is retained: `grad=None` must not be replaced
by a zero gradient, which would incorrectly activate AdamW momentum/decay.

The loaded optimizer **never steps**. Each arm creates disjoint parameter and
moment copies on the same device, restores the same state, and calls an actual
`torch.optim.AdamW.step` once. It measures the resulting dtype-rounded weight
delta. The copies are discarded, never saved or forwarded through a model.
Every window is independent at the same endpoint, not a second training step.

## Report and limits

Each window reports per-module relative L2 differences and direction cosines
at four stages: `raw`, `routed`, `clipped`, `adamw_update`. Routing conflict
statistics, independent clipping coefficients, native-forward receipts and
shadow-step identities are included. The live model/optimizer and source files
are checked for mutation before success is declared.

Small update differences can result from clipping or restored optimizer
history; they do not prove long-term equivalence. Large differences do not say
which objective improves rare AP. All trainable parameters are included rather
than three post-selected validation categories. No validation annotations are
used to compute gradients, tune loss weights or choose a winner.

Manual gradient reduction is not bitwise DDP bucket replay. This does not
recreate physical eight-GPU training, establish the old model's provenance or
explain an 8ep-to-12ep performance change from a single endpoint. Keeping live
prototypes unchanged also does not guarantee non-collapse during future
training. There is no automatic follow-up training, evaluation or parameter
sweep.

Development tests cover the normalization/AMP algebra, native routing/clipping
methods, AdamW copy isolation and `None` semantics. Full CUDA/Detectron2 replay
must still be verified on the server; CPU unit tests are not that verification.
