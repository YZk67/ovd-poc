# CPU-only fixed-candidate query / terminal bank / bias audit

Input is the **completed** `trace_rare_gt_queries.py` report for the no-radius
8ep/12ep endpoints and scarecrow GT 137708 in image 218917. All needed file paths
and SHA256 digests are in that report. No checkpoint, image, annotation index,
GPU, optimizer, model construction, or cache regeneration is needed.

```bash
cd ~/LaMI-DETR
mkdir -p /root/autodl-tmp/no_radius_scarecrow_readout
set -o pipefail
CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/lami/bin/python -u \
  tools/analyze_rare_query_readout.py \
  --source-json /root/autodl-tmp/no_radius_scarecrow_queries_v2/report.json \
  --output /root/autodl-tmp/no_radius_scarecrow_readout/report.json \
  2>&1 | tee /root/autodl-tmp/no_radius_scarecrow_readout/run.log
```

Preserve both earlier `pairing_cache` directories and the two
`*_all_query_scores.pt` files. Missing or modified inputs fail before producing
a result; there is no fallback inference. Upload the new `report.json`.

## What is fixed and what is exchanged

For each endpoint, keep **every native query with GT IoU >= 0.5**. The primary
candidate is the highest native fused true-class score among eligible queries,
chosen before any exchange. For the supplied report these are old q251 and new
q234; the IDs are **not** treated as corresponding cross-model slots.

Each fixed candidate gets both terminal prototype banks and both scalar biases.
Its box, CLIP log probability, and reference native cutoff stay unchanged. Full
1203-way logits are computed for that candidate to report true-class rank and
margin, not just the true-class scalar. Each native corner must reproduce its
cached full-vocabulary detector vector and fused score. Hidden attention queries
are 256D; cached classifier features and output prototypes are **768D**.

This is a terminal-classifier exchange only. It does not rerun upstream query
fusion with another bank. Projected query differences include the classifier
linear projection and all preceding paths, including upstream TPA interactions;
they cannot be called a decoder-only update effect. Different endpoints retain
their own GT-anchored boxes/CLIP values, not an artificially identical ROI.

## Interpretation

- The primary 2 x 2 x 2 table separates projected feature source, terminal bank
  source, and scalar bias source. Other eligible queries are in JSON as controls.
- Pre-bias conditional effects show both exchange orders and the query-bank
  interaction. The logit delta is also split into the symmetric query effect,
  symmetric bank effect, and additive scalar bias change.
- Because sigmoid is nonlinear, weighted log-detector contributions use the
  exact average over all six exchange orders (three-factor Shapley allocation).
  The native CLIP log change is added separately. Subtracting the native cutoff
  log change closes the native score/cutoff margin delta. Closure is checked.
- These allocations describe endpoint readout arithmetic, **not** historical
  loss/gradient causes or separate causal contributions of training modules.
- A scalar bias cannot change within-query category ranking, but it can change
  sigmoid scores and fused scoring. Temperature, logit scale, class mapping,
  prompt identity, CLIP text bank and novel mask must agree across endpoints.
- `frozen_cutoff_ratio` compares against the ORIGINAL cutoff. Counterfactual
  image ranks, TP/FP and AP are deliberately absent: replacing an entire bank
  could also change competing query/category scores. Do not interpret crossing
  the frozen cutoff as a verified rescue or an improved APr.
- This selected single example cannot establish a general rare-class mechanism.

The script modifies only its output report/manifest and never the source caches.
Use a new output name if inputs or analysis code change; output provenance locks
are retained.
