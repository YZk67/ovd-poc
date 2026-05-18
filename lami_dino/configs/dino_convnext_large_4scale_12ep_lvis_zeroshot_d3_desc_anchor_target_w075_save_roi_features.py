from .dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_desc_anchor_target_weighted_cls_only import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


d3_w075_bank_path = "dataset/metadata/d3_description_anchor_target_w075_bank_convnextl.npy"

model.classifier.zs_weight_path = d3_w075_bank_path
model.classifier.eval_zs_weight_path = d3_w075_bank_path
model.classifier.text_embed_path = d3_w075_bank_path
model.classifier.eval_text_embed_path = d3_w075_bank_path

# Save detector-side ROI features for every eval image. These .pth files are
# later matched back to verifier_pairs rows to train a verifier on the same
# feature distribution used by DINO.forward().
model.score_ensemble = True
model.backbone.score_ensemble = True
model.vlm_query_path = "dataset/metadata/d3_clip_convnextl_sentences.npy"
model.seen_classes = "dataset/metadata/d3_seen_empty.json"
model.all_classes = "dataset/metadata/d3_phrases.json"
model.unseen_classes = "dataset/metadata/d3_phrases.json"
model.alpha = 0.0
model.beta = 0.1
model.novel_scale = 1.0
model.save_dir = "output/d3_roi_features_w075/pth"
model.save_roi_features_only = True
model.save_roi_features_fp16 = True

train.output_dir = "./output/lami_convnext_large_12ep_lvis_zeroshot_d3_desc_anchor_target_w075_save_roi_features"
dataloader.evaluator.output_dir = train.output_dir
