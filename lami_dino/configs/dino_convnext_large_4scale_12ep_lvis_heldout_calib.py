from .dino_convnext_large_4scale_12ep_lvis import dataloader, lr_multiplier, model, optimizer, train


# Held-out-base calibration probe:
# - sample pseudo novel class ids with tools/make_lvis_pseudo_novel_split.py
# - mask their GT during detector training
# - ignore unmatched predictions overlapping those masked boxes in classification loss
model.heldout_class_ids_path = "dataset/lvis/pseudo_novel_base100_seed42.json"
model.heldout_ignore_iou = 0.5

# Keep this short first; the goal is to produce checkpoints for calibration diagnostics.
train.max_iter = 21300
train.eval_period = 7100
train.checkpointer.period = 7100
