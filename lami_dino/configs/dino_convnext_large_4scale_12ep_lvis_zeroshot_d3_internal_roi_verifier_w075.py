from .dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_internal_verifier_w075 import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


# Same internal fusion path as the crop-trained verifier config, but the
# checkpoint is expected to be trained from detector ROI-feature caches exported
# by tools/export_d3_roi_verifier_features.py.
model.region_verifier_checkpoint = "output/d3_roi_verifier_w075_no_score/verifier_best.pt"

train.output_dir = "./output/lami_convnext_large_12ep_lvis_zeroshot_d3_internal_roi_verifier_w075"
dataloader.evaluator.output_dir = train.output_dir
