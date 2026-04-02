"""OV-COCO + VLM Distillation on LaMI-DETR.

Finetune from OV-COCO baseline (epoch 2) with VLM soft target distillation.
Adds KL divergence loss between model predictions and VLM similarity distributions.
"""

from .dino_convnext_large_ovcoco import model, dataloader, optimizer, lr_multiplier, train

# === VLM distillation settings ===
model.vlm_distill_enabled = True
model.vlm_distill_weight = 1.0
model.vlm_distill_temperature = 2.0
train.vlm_targets_path = "dataset/vlm_targets_crop/vlm_soft_targets.pth"

# === Resume from epoch 2 baseline ===
train.init_checkpoint = "./output/dino_convnext_large_ovcoco/model_0014199.pth"
train.output_dir = "./output/dino_convnext_large_ovcoco_vlm_distill"

# Train 3 more epochs from epoch 2
train.max_iter = 21300  # 3 epochs * 7100
train.eval_period = 7100
train.checkpointer.period = 7100

dataloader.evaluator.output_dir = train.output_dir
