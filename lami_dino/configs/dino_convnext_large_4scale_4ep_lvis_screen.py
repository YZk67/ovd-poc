"""Matched four-epoch screening protocol for InstructDet on OV-LVIS.

The run stops after four epochs but deliberately retains the 12-epoch LR
horizon. Its checkpoint is therefore directly comparable to iteration 28,399
of a formal 12-epoch run; no LR decay is compressed into the screening window.
"""

from .dino_convnext_large_4scale_12ep_lvis import (
    dataloader,
    iterations_per_epoch,
    lr_multiplier,
    model,
    optimizer,
    train,
)


train.max_iter = 4 * iterations_per_epoch
train.lr_scheduler_max_iter = 12 * iterations_per_epoch
train.eval_period = 4 * iterations_per_epoch
train.checkpointer.period = 2 * iterations_per_epoch
train.output_dir = "./output/instructdet_lvis_4ep_screen"
dataloader.evaluator.output_dir = train.output_dir

# Locked inference protocol: never retune these per candidate.
model.alpha = 0.0
model.beta = 0.3
model.novel_scale = 3.0
model.classifier.tpa_tau = 0.004375
model.classifier.tpa_cls_tau = 0.07
model.soft_category_topk = 3
