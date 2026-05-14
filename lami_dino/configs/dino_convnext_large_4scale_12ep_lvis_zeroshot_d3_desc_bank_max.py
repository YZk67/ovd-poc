from .dino_convnext_large_4scale_12ep_lvis_zeroshot_d3 import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


d3_description_bank_path = "dataset/metadata/d3_description_bank_convnextl.npy"

model.query_path = d3_description_bank_path
model.eval_query_path = d3_description_bank_path
model.vlm_query_path = d3_description_bank_path
model.classifier.zs_weight_path = d3_description_bank_path
model.classifier.eval_zs_weight_path = d3_description_bank_path
model.classifier.text_embed_path = d3_description_bank_path
model.classifier.eval_text_embed_path = d3_description_bank_path
model.classifier.use_tpa = False
model.classifier.static_multi_prototype_agg = "max"

train.output_dir = "./output/lami_convnext_large_12ep_lvis_zeroshot_d3_desc_bank_max"
dataloader.evaluator.output_dir = train.output_dir
