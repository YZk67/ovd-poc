from .dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_train_roi_verifier_w075 import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


# Keep the detector fixed and update only the description-region verifier.
# This is the right setting for warm-starting from an offline ROI verifier:
# the detector already gives the best known D3 baseline, while naive joint
# fine-tuning quickly damages the ranking.
optimizer.params.lr_factor_func = lambda module_name: 1.0 if module_name.startswith("region_verifier.") else 0.0

train.output_dir = "./output/lami_convnext_large_12ep_lvis_zeroshot_d3_train_roi_verifier_only_w075"
dataloader.evaluator.output_dir = train.output_dir
