# No-radius 8ep → 12ep rare performance retrospective

The measured endpoints are APr **42.8843 → 42.4229**, a change of
**−0.4614 AP points**. This is not the earlier 8ep → 10ep comparison.
This tool accounts for the performance difference using existing predictions;
it does **not** identify the historical training loss or update that caused it.

## Run on the server

Sync the code first. Both original **all-class, full-validation** prediction
JSONs must exist. Do not use a later overwritten prediction JSON for 8ep.
The 12ep training-directory prediction must still belong to `model_final.pth`
(iteration 85199), not an intervening evaluation.

```bash
cd ~/LaMI-DETR
set -o pipefail
/root/miniconda3/envs/lami/bin/python -u tools/run_rare_stage_comparison.py \
  --old-predictions /root/autodl-tmp/eval_k5_no_radius_bs32_8ep/lvis_instances_results.json \
  --new-predictions /root/autodl-tmp/instructdet_k5_no_radius_bs32_4ep_seed42/lvis_instances_results.json \
  --expected-old-apr 42.8843 \
  --expected-new-apr 42.4229 \
  --old-label 8ep --new-label 12ep \
  --full-iou \
  --output-dir /root/autodl-tmp/no_radius_8ep_vs_12ep_full_pr \
  2>&1 | tee /root/autodl-tmp/no_radius_8ep_vs_12ep_full_pr.log
```

Two sequential **CPU LVIS evaluator passes**, then report-only comparisons.
GPU devices are hidden from subprocesses. No model inference, training,
checkpoint loading, formula changes or threshold selection. The original
all-class image top-300 selection and official LVIS ignore rules are retained.

From another terminal:

```bash
tail -F /root/autodl-tmp/no_radius_8ep_vs_12ep_full_pr.log
```

Missing inputs, incompatible curves, or APr mismatch (tolerance 0.0002 AP
points) stop the pipeline. Use a separate, new output directory; existing
reports are never silently overwritten. Source predictions, annotations,
reused reports and diagnostic code are hashed before/after the pipeline.
Do not move, overwrite or re-evaluate into the source prediction paths while
it runs. A valid `COMPLETE.json` confirms success and unchanged source files.

### Optional report reuse

If complete reports already exist, add `--old-report PATH --new-report PATH`
(either side may be reused). They must include every valid rare category and
all ten IoUs, not just AP50/AP75. Both reusable reports are validated **before**
any evaluator pass. Incomplete supplied reports cause an error, not an
automatic regeneration. Source prediction/annotation paths, size and expected
APr must match. Older reports lack an original prediction hash: those checks
cannot independently authenticate their historical provenance.

If both CPU passes completed but the comparison was interrupted, reuse their
`old_report.json` / `new_report.json` in a fresh output directory, or run
`tools/compare_rare_stage_full_pr.py` directly with `--old-report`,
`--new-report`, both `--expected-*-apr` values, and a new `--output` file. The
direct command is standard-library-only and never evaluates predictions.

## Read the results

Upload **`no_radius_8ep_vs_12ep_full_pr/report.json`**. Other files:

- `old_report.json`, `new_report.json`: official per-class AP and complete raw
  and interpolated PR curves (unless explicitly reused from another location).
- `comparison.json`: the existing AP50/AP75 and top-decline comparison.
- `inputs.json`, `COMPLETE.json`: provenance and completion records.

`report.json` covers **all valid rare classes and IoUs 0.50:0.05:0.95**, with
no inherited focus on the previous auxiliary-loss experiment's three classes.
For each class/IoU, it decomposes the official 101-point AP difference into:

1. Lost recall support: old precision on recall grid points no longer reached.
2. Gained recall support: new precision on newly reached recall grid points.
3. Shared-recall precision change: new minus old precision where both have
   recall support.

A zero-TP curve has **no support**, including the recall=0 grid point. Losing
the last TP is counted entirely as lost support, not partly as a ranking
effect. This corrects an edge case in the shared PR comparison helper; it
does not change official AP values. Each class's ten-IoU sum and the overall
macro sum must close to the official ΔAPr, otherwise analysis fails.

Also recorded: TP/FP counts, recall, FP before the first TP and equal-recall
TP ordinals, GT-count strata, positive/negative class contributions, and
exhaustive leave-one-category-out sensitivity. Outcome-selected gain/loss/
unchanged examples are for display only; all classes remain in official APr.

## Interpretation limits

- Saved-detection recall loss does not prove missing raw proposals: top-300,
  classification, localization and matching can all change attained recall.
- Shared precision loss is not exclusively “more high-score FP”: TP may move
  down, FP up, or both. Equal-recall TP ordinals are not paired GT identities.
- Leave-one-category-out sensitivity is **not** a confidence interval,
  significance test, seed replication or permission to exclude classes.
- Expected APr and file hashes guard input mix-ups/changes; without original
  evaluation receipts they do not prove checkpoint or config identity.
- PR accounting does not isolate DN, auxiliary classification, APR, RPSA,
  optimizer moments, LR or feature drift as the training cause. The tool
  makes no automatic recommendation to train or change any of them.
