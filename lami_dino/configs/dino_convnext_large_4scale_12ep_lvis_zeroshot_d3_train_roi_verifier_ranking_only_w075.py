from .dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_train_roi_verifier_only_w075 import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


# Ranking-loss variant of verifier-only warm fine-tuning. The detector remains
# frozen; only the region-description verifier is updated.
model.region_verifier_train_loss_type = "pairwise_rank"
model.region_verifier_ranking_margin = 0.0
model.region_verifier_topk_per_image = 50

train.output_dir = "./output/lami_convnext_large_12ep_lvis_zeroshot_d3_train_roi_verifier_ranking_only_w075"
dataloader.evaluator.output_dir = train.output_dir
