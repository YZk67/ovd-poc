"""OV-COCO 65 + VLM KL distillation.

Finetune the current best novel checkpoint with a novel-class-only KL
divergence loss against pre-computed VLM soft targets.
"""

from .dino_convnext_large_4scale_12ep_lvis import (
    model,
    dataloader,
    optimizer,
    lr_multiplier,
    train,
)

# VLM distillation settings
model.vlm_distill_enabled = True
model.vlm_distill_weight = 0.5
model.vlm_distill_temperature = 1.0
train.vlm_targets_path = "dataset/vlm_targets_crop/vlm_soft_targets.pth"

# Finetune from the iter-8999 peak produced by the no_ema_baseline run.
train.init_checkpoint = "./output/no_ema_baseline/model_best_novel.pth"
train.output_dir = "./output/distill_from_8999"

# Short finetune — the base is already near-peak.
train.max_iter = 5000
train.eval_period = 500
train.checkpointer.period = 5000

# Lower LR: 1/10 of the original, to avoid disturbing the 28.38 baseline.
optimizer.lr = 1e-5

dataloader.evaluator.output_dir = train.output_dir
