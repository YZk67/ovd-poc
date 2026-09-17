# Optimizer-aware query/bank loss-source screen

This is the bounded screening step between the completed 8ep/10ep endpoint
diagnostics and any new paired training intervention.  It does **not** train a
model or estimate AP.

The audit starts from the complete no-radius 8ep checkpoint (iteration 56799)
and restores its AdamW moments.  On two native four-rank effective batches it
separates these weighted training losses:

- final decoder classification;
- auxiliary decoder classification;
- denoising classification;
- encoder classification;
- all L1/GIoU box losses;
- APR;
- RPSA.

It measures their finite optimizer-step contribution to two disjoint parameter
targets:

- `query`: `query_content + decoder_core + final_projection`;
- `bank`: the shared TPA parameters used to form the terminal prototype bank.

For each source/target pair the script analytically removes that direct gradient,
then recomputes the affected APR conflict routing, separate L2 clipping
coefficient, and AdamW update with the restored moments and weight decay.  The
result is compared with the actual 8ep-to-10ep parameter displacement.  This is
an optimizer-aware association screen, not proof that a loss caused APr to fall.

## Run

Use the same four-GPU topology as training.  The default is two windows, or 64
training-image exposures in total.  There are no validation forwards, LVIS
evaluation passes, optimizer steps, checkpoint writes, or parameter updates.

```bash
cd ~/LaMI-DETR
set -o pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3 /root/miniconda3/envs/lami/bin/python -u \
  tools/audit_pairing_optimizer_sources.py \
  --query-audit /root/autodl-tmp/no_radius_8ep_vs_10ep_query_updates/report.json \
  --output-dir /root/autodl-tmp/no_radius_pairing_optimizer_sources \
  --num-gpus 4 \
  --windows 2 \
  2>&1 | tee /root/autodl-tmp/no_radius_pairing_optimizer_sources.log
```

Follow progress from another terminal:

```bash
tail -n 40 -F /root/autodl-tmp/no_radius_pairing_optimizer_sources.log
```

The output directory must be new.  A partial failed run is preserved and is not
silently resumed; use another new directory after fixing the cause.  Completion
requires both `report.json` and `COMPLETE.json`.

Upload:

```text
/root/autodl-tmp/no_radius_pairing_optimizer_sources/report.json
```

## Reading the report

`screening_ranking` orders source/target rows by the mean cosine between the
source's finite AdamW update contribution and the observed 8ep-to-10ep endpoint
delta.  Positive alignment only means the source tends to drive parameters in
the same direction as that historical displacement.  It does not say that the
direction is harmful to AP.

Before selecting a paired intervention, require:

1. a nonzero direct connection in both windows;
2. the same alignment sign in both windows;
3. successful loss-gradient reconstruction (`relative_l2 <= 2e-3`);
4. no anomalous clipping or routing discontinuity that alone explains the row;
5. consistency with the already completed endpoint/readout evidence.

At most one remaining loss source should proceed to a query/bank 2x2 paired
training experiment.  Auxiliary classification has already failed its direct
decoder intervention and should not be selected again merely because a local
screening cosine is positive.

## Scope limits

- `query` is the terminal semantic query path, not the complete backbone and
  encoder feature extractor.
- The original AMP gradients are finite.  The audit operates on their unscaled
  equivalent and validates, but does not advance, the stored GradScaler.
- Removing a target gradient changes the corresponding clipping coefficient;
  the report records this.  It does not summarize clipping-mediated AdamW
  changes to every non-target parameter.
- AdamW moments and weight decay remain present in every counterfactual.
- The endpoint delta contains thousands of historical updates and optimizer
  interactions.  A one-step alignment is only a shortlist signal.
- A causal statement still requires a paired continuation from the same 8ep
  state and an official full-LVIS comparison.

## Local regression

```bash
PYTHONPATH=. python -m pytest -q --rootdir=tests --confcutdir=tests \
  tests/test_pairing_optimizer_sources.py \
  tests/test_gradient_accumulation.py \
  tests/test_prototype_ops.py
```
