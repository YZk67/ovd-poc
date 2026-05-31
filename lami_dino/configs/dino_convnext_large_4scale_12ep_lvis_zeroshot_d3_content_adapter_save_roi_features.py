from .dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_train_alias_dn_mean_content_adapter_w075 import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


# Save ROI features from a content-adapter checkpoint. These features are used
# to train a matched region-description verifier on the same detector/proposal
# distribution, avoiding the old w075 verifier distribution mismatch.
model.region_verifier_enabled = False
model.region_verifier_train_enabled = False
model.criterion.weight_dict["loss_region_verifier"] = 0.0

model.save_dir = "output/d3_roi_features_alias_dn_content_adapter/pth"
model.save_roi_features_only = True
model.save_roi_features_fp16 = True

train.output_dir = "./output/lami_convnext_large_12ep_lvis_zeroshot_d3_content_adapter_save_roi_features"
dataloader.evaluator.output_dir = train.output_dir
