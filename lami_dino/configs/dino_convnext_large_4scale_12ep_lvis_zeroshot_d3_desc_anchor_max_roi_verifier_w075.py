from .dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_desc_anchor_logsumexp_roi_verifier_w075 import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


model.classifier.static_multi_prototype_agg = "max"

train.output_dir = "./output/lami_convnext_large_12ep_lvis_zeroshot_d3_desc_anchor_max_roi_verifier_w075"
dataloader.evaluator.output_dir = train.output_dir
