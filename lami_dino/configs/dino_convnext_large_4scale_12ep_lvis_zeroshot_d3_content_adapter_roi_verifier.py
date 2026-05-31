from .dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_train_alias_dn_mean_content_adapter_w075 import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


# Eval a content-adapter checkpoint with a verifier trained from its own ROI
# feature/proposal distribution. Override model.region_verifier_checkpoint with
# the matched verifier path produced by tools/train_d3_crop_verifier.py.
model.region_verifier_enabled = True
model.region_verifier_train_enabled = False
model.criterion.weight_dict["loss_region_verifier"] = 0.0
model.region_verifier_checkpoint = "output/d3_roi_verifier_alias_dn_content_adapter/verifier_best.pt"
model.region_verifier_text_path = "dataset/metadata/d3_description_anchor_target_w100_bank_convnextl.npy"
model.region_verifier_text_index = None
model.region_verifier_fusion = "logit_add"
model.region_verifier_fusion_weight = 0.25
model.region_verifier_topk_per_image = 50

train.output_dir = "./output/lami_convnext_large_12ep_lvis_zeroshot_d3_content_adapter_roi_verifier"
dataloader.evaluator.output_dir = train.output_dir
