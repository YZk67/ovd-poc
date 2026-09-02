# InstructDet Paper Experiment Lock

This file is the source of truth for the first paper-aligned OV-LVIS run. Do not
change one item mid-run and compare the resulting checkpoint with another row.

## Initialization and training

- Framework: DINO with ConvNeXt-L.
- Fresh-run initialization: `clip_convnext_large_trans.pth`, filtered to
  backbone keys only. Detector, transformer, TPA, APR, and RPSA start fresh.
- Backbone scope: ConvNeXt downsampling/stage trunk frozen; detection output
  norms (`norm1`/`norm2`/`norm3`) trainable at 0.1x LR.
- Optimizer: AdamW, base LR `1e-4`, weight decay `1e-4`, betas `(0.9, 0.999)`.
- TPA LR: 10x base LR (`1e-3`). This must be disclosed in the paper.
- Hardware/batch: 4 GPUs, total batch size 16, AMP enabled.
- Schedule: 12 epochs, 7,100 iterations/epoch, 85,200 total iterations.
- Seed: 42. Dataset: `lvis_v1_train_norare`; federated focal loss over 100
  sampled categories per step. TPA/APR/RPSA use that same synchronized category
  subset during training and the full vocabulary at evaluation.

## Paper equations

- Eq. (1): `d_h=256`, `tau_p=0.004375`, hence
  `sqrt(d_h) * tau_p = 0.07`.
- Eq. (2): calibrated log-mean-exp, `tau_cls=0.07`, shared scale `s=50`.
- Eqs. (3)-(4): top-R soft category fusion with `R=3`, `tau_s=1.0`,
  `tau_q=0.15`, and stop-gradient on the encoder seed.
- Eq. (5): 5 prototypes; directional diversity weight 0.10 and usage-balance
  weight 0.03, cosine warm-up over the first 5% of training. Global APR weight
  is 0.1.
- Eq. (6): 8 visual centers, `tau_r=0.06`, soft-k-means sigma 1.0. RPSA uses
  the top 1,024 distinct valid encoder tokens, excludes padding, and applies the
  stricter of fixed background threshold 0.05 and per-image 60th percentile.
- Eq. (7): global RPSA weight 0.05; zero through iteration 20,000 and linearly
  ramped to full strength over the following 8,000 iterations.
- LVIS evaluation uses the frozen CLIP score-ensemble branch for every ablation
  with `alpha=0.0`, `beta=0.4`, and novel-class scale 5.0.

## Launch gates

1. `python tools/preflight_instructdet.py --hash` passes on the training host.
2. Fresh-run log says `Loaded CLIP backbone only`, with at least 95% parameter
   coverage, and validates `output_norm_only` backbone scope.
3. A 200-500 iteration smoke run has finite task/APR/RPSA losses and observes
   active RPSA batches with non-zero valid clusters after its shortened warm-up.
   An empty filtered set for an individual batch is an observable zero-loss skip
   (`loss_rpsa_active=0`); shape errors and non-finite values remain fail-fast.
4. Record `tpa/proto_pairwise_cos`, `tpa/proto_effective_rank`,
   `query_fusion/category_max_weight`, `query_fusion/prototype_max_weight`,
   `loss_rpsa_bg_ratio`, `loss_rpsa_valid_clusters`, `loss_rpsa_active`, and
   `loss_rpsa_empty_image_ratio`.
5. Every ablation uses a distinct stable output directory and the same seed,
   initialization, batch size, and iteration budget.

## Manuscript details that must match this lock

The revised paper must disclose the TPA 10x LR, exact temperatures and auxiliary
loss weights, the ConvNeXt output-norm-only fine-tuning scope, RPSA token and
background filtering, and the exact APR/RPSA schedules above.
