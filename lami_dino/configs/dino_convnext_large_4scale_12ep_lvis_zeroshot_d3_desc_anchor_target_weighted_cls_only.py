from .dino_convnext_large_4scale_12ep_lvis_zeroshot_d3 import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


d3_description_anchor_target_weighted_bank_path = (
    "dataset/metadata/d3_description_anchor_target_w050_bank_convnextl.npy"
)

# Weighted banks are already collapsed to one prototype per class. Keep query
# initialization and the score-ensemble VLM branch on the original D3 phrase bank.
model.vlm_query_path = "dataset/metadata/d3_clip_convnextl_sentences.npy"
model.classifier.zs_weight_path = d3_description_anchor_target_weighted_bank_path
model.classifier.eval_zs_weight_path = d3_description_anchor_target_weighted_bank_path
model.classifier.text_embed_path = d3_description_anchor_target_weighted_bank_path
model.classifier.eval_text_embed_path = d3_description_anchor_target_weighted_bank_path
model.classifier.use_tpa = False

train.output_dir = "./output/lami_convnext_large_12ep_lvis_zeroshot_d3_desc_anchor_target_w050_cls_only"
dataloader.evaluator.output_dir = train.output_dir
