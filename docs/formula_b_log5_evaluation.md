# Formula B: one evaluation with log(5) restored

Use the completed `run_tpa_formula_screen.py` directory on the training server.
The runner reads its manifest, summary and verified A/B evaluation receipts,
checks B's checkpoint identity, and reuses B's recorded inference protocol.
The model override changed is `model.classifier.tpa_eval_logit_bias`, from zero
to `1.6094379124341003`. Aggregation remains calibrated; this is not legacy Eq.2.

```bash
cd ~/LaMI-DETR
set -o pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3 /root/miniconda3/envs/lami/bin/python -u \
  tools/evaluate_formula_b_log5.py \
  --screen-dir /root/autodl-tmp/no_radius_tpa_formula_screen_500 \
  --num-gpus 4 \
  2>&1 | tee /root/autodl-tmp/no_radius_tpa_formula_B_log5_driver.log
```

The new output directory is `SCREEN/eval_B_plus_log5`. Do not pre-create it.
Use `--output-dir` for another new directory. `--dry-run` validates inputs and
prints the command without creating outputs or launching inference.

The runner performs exactly one full LVIS evaluation, with zero training
updates and no checkpoint rewrite. It checks the original B checkpoint hash
again after evaluation. Previously verified TPA geometry is therefore preserved.
Native inference recomputes boxes, CLIP ROI scores and top-300 selections; this
is not a fixed-candidate rescoring experiment.

`eval_B_plus_log5/summary.json` contains:

- `A_calibrated`, `B_calibrated`, `B_plus_log5`: official metrics.
- `delta_B_plus_log5_minus_B_calibrated`: the same-checkpoint inference effect.
- `delta_B_plus_log5_minus_A_calibrated`: the joint training/inference difference.

The second delta cannot be attributed to training alone. The existing screen
is left intact; the script neither retrains an arm nor evaluates C.

```bash
tail -F /root/autodl-tmp/no_radius_tpa_formula_B_log5_driver.log
```
