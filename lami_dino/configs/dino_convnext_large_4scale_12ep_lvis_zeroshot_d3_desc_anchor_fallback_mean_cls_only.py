from .dino_convnext_large_4scale_12ep_lvis_zeroshot_d3 import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


d3_description_anchor_fallback_bank_path = (
    "dataset/metadata/d3_description_anchor_fallback_bank_convnextl.npy"
)

# Keep query initialization and the score-ensemble VLM branch on the original
# D3 phrase bank. Only the detector classifier sees the fallback anchor bank.
model.vlm_query_path = "dataset/metadata/d3_clip_convnextl_sentences.npy"
model.classifier.zs_weight_path = d3_description_anchor_fallback_bank_path
model.classifier.eval_zs_weight_path = d3_description_anchor_fallback_bank_path
model.classifier.text_embed_path = d3_description_anchor_fallback_bank_path
model.classifier.eval_text_embed_path = d3_description_anchor_fallback_bank_path
model.classifier.use_tpa = False
model.classifier.static_multi_prototype_agg = "mean"

train.output_dir = "./output/lami_convnext_large_12ep_lvis_zeroshot_d3_desc_anchor_fallback_mean_cls_only"
dataloader.evaluator.output_dir = train.output_dir
