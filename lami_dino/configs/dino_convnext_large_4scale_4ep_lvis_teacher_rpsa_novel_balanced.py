"""Four-epoch candidate: GT-anchored, novel-balanced teacher RPSA."""

from .dino_convnext_large_4scale_4ep_lvis_teacher_rpsa import (
    dataloader,
    iterations_per_epoch,
    lr_multiplier,
    model,
    optimizer,
    train,
)


train.output_dir = "./output/instructdet_lvis_teacher_rpsa_novel_balanced_4ep"
dataloader.evaluator.output_dir = train.output_dir

# Preserve the original teacher-RPSA config as a reproducible ablation. This
# flag alone switches proposal routing to one anchor per GT, novel-only pseudo
# labels, and separately normalized GT/novel loss groups.
model.teacher_rpsa_novel_balanced = True
model.teacher_rpsa_novel_weight = 1.5
model.criterion.weight_dict["loss_rpsa"] = 0.1
