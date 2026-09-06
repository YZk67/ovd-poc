"""Four-epoch candidate: full-vocabulary CLIP teacher-routed mode alignment."""

from .dino_convnext_large_4scale_4ep_lvis_screen import (
    dataloader,
    iterations_per_epoch,
    lr_multiplier,
    model,
    optimizer,
    train,
)


train.output_dir = "./output/instructdet_lvis_teacher_rpsa_4ep"
dataloader.evaluator.output_dir = train.output_dir

# Isolate the new routing objective from the legacy self-routed, FedLoss-subset
# RPSA. APR, collapse prevention, query fusion, and inference stay unchanged.
model.transformer.use_rpsa = False
model.teacher_rpsa = True
model.teacher_rpsa_num_proposals = 64
model.teacher_rpsa_category_topk = 3
model.teacher_rpsa_confidence_threshold = 0.25
model.teacher_rpsa_margin_threshold = 0.05
model.teacher_rpsa_gt_iou_threshold = 0.5
model.teacher_rpsa_mode_temperature = 0.07
model.teacher_rpsa_novel_weight = 1.5
model.teacher_rpsa_warmup_start = iterations_per_epoch
model.teacher_rpsa_warmup_iters = iterations_per_epoch
model.criterion.weight_dict["loss_rpsa"] = 0.05
