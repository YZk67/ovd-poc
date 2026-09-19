# A/P 4,000-update rare AP concentration (CPU only)

Uses the completed extension's **existing all-class prediction JSONs**. A is
full APR with conflict projection; P is the same APR with projection disabled.
Does not load checkpoints, run inference, change scores/top-300, or train.
Both official LVIS CPU evaluations receive the full prediction JSON before
selecting rare category precision tensors. All valid rare classes are included
in AP attribution, even though the reused reporter prints only ten PR examples.

## Run on the server

Use the environment that already provides NumPy, LVIS and pycocotools. No GPU is
required. Evaluation is sequential so the two LVIS evaluators do not coexist in
memory. The two ~937 MB input files are hashed against the completed summary;
APr must reproduce the recorded A=41.4781 and P=42.1191 (tolerance 0.0002 points).
The manifest identifies the exact validation annotation file. The script reads
expected metrics from the summary rather than hard-coding the observed delta.

```bash
cd ~/LaMI-DETR
nohup /root/miniconda3/envs/lami/bin/python -u tools/analyze_apr_projection_rare_ap.py \
  --trial-dir /root/autodl-tmp/no_radius_apr_projection_4000 \
  --output-dir /root/autodl-tmp/no_radius_apr_projection_4000_rare_ap \
  > /root/autodl-tmp/no_radius_apr_projection_4000_rare_ap.log 2>&1 &
```

```bash
tail -F /root/autodl-tmp/no_radius_apr_projection_4000_rare_ap.log
```

If interrupted, run the same command with `--resume`. Completed CPU arm reports
are reused only after input/code/receipt and report SHA256 checks. Changed inputs
or code require a new output directory. No GPU fallback is available or attempted.
Missing predictions cause failure, not reconstruction from a log/APr value.

## Outputs and interpretation

- `STATUS.json`: RUNNING, FAILED, or COMPLETE; inspect this before using results.
- `report.json`, `results.txt`: official P−A APr; positive/negative contributions;
  all five validation-GT strata (1, 2–4, 5–9, 10–19, 20+); sign counts and medians.
- `per_class.csv`: **all valid** rare classes' A/P AP, AP50, AP75, deltas and contributions.
- Single-GT net, positive and negative contributions, positive-gain share, and
  remaining means for GT≥2/5/10. Contributions always divide by the original
  valid rare class count; undefined AP classes are excluded, not counted as zero.
- Sensitivity after removing the largest 1/2/5 gains, largest 1/2/5 losses, or
  both tails. Removed classes and GT counts are explicit. The remaining mean
  uses the remaining denominator; remaining contribution uses the original one.
  Additional single-GT-only tail removals show whether just a few one-instance
  categories account for the positive direction, with both signs also checked.
- `A_report.json`, `P_report.json`, receipts and `inputs.json`: reusable official
  CPU reports and provenance. Source predictions and trial records are untouched.

The official APr stays unchanged. Removing extreme classes is a post-hoc
concentration diagnostic, **not a corrected performance number**. Positive and
negative contributions may cancel, so a share of net gain is not a probability.
One paired seed/trajectory does not establish significance or generalization.
This report cannot identify a causal module/loss, authorize class-dependent
tuning, or automatically adopt P/start another training run.

Upload `report.json` and `results.txt` after STATUS is COMPLETE. No need to upload
the two original large prediction JSONs.
