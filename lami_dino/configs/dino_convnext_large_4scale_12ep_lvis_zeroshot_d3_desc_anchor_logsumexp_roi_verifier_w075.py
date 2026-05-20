from .dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_internal_roi_verifier_w075 import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


# First-stage classifier now uses the multi-description alias bank and aggregates
# aliases per phrase with logsumexp. The frozen ROI verifier remains the stable
# region-description consistency module from the D3 verifier experiments.
d3_description_anchor_bank_path = "dataset/metadata/d3_description_anchor_bank_convnextl.npy"

model.classifier.zs_weight_path = d3_description_anchor_bank_path
model.classifier.eval_zs_weight_path = d3_description_anchor_bank_path
model.classifier.text_embed_path = d3_description_anchor_bank_path
model.classifier.eval_text_embed_path = d3_description_anchor_bank_path
model.classifier.use_tpa = False
model.classifier.static_multi_prototype_agg = "logsumexp"
model.region_verifier_topk_per_image = 50

train.output_dir = "./output/lami_convnext_large_12ep_lvis_zeroshot_d3_desc_anchor_logsumexp_roi_verifier_w075"
dataloader.evaluator.output_dir = train.output_dir
