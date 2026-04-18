"""Ablation: raise novel-mass threshold to 0.1 to kill noisy base-GT distill signal.

Hypothesis: variance driver is that ~37% of distill hits come from base-class GTs
where VLM has small-but-nonzero novel mass (ambiguous crops). Requiring >=10%
novel mass should leave only confident novel-pseudo targets, reducing variance.

Short validation run: 2500 iter, eval every 250 → 10 evals, ~1.5h on 4x A100.
"""

from .dino_convnext_large_4scale_12ep_lvis_distill import (
    model,
    dataloader,
    optimizer,
    lr_multiplier,
    train,
)

# The only knob changed vs. baseline distill run.
model.vlm_distill_novel_mass_threshold = 0.1

# Shorter run for variance ablation.
train.output_dir = "./output/distill_thr01_from_8999"
train.max_iter = 2500
train.eval_period = 250
train.checkpointer.period = 2500

dataloader.evaluator.output_dir = train.output_dir
