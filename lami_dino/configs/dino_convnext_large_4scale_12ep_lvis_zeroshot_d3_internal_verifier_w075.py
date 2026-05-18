from .dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_desc_anchor_target_weighted_cls_only import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


# First-stage classifier uses the best target-framed weighted prototype found so far.
d3_w075_bank_path = "dataset/metadata/d3_description_anchor_target_w075_bank_convnextl.npy"
target_prompt_bank_path = "dataset/metadata/d3_description_anchor_target_w100_bank_convnextl.npy"

model.classifier.zs_weight_path = d3_w075_bank_path
model.classifier.eval_zs_weight_path = d3_w075_bank_path
model.classifier.text_embed_path = d3_w075_bank_path
model.classifier.eval_text_embed_path = d3_w075_bank_path

# Keep the existing score-ensemble branch enabled for the first-stage D3 baseline
# and for extracting internal 768D region features from ConvNeXt/CLIP head.
model.score_ensemble = True
model.backbone.score_ensemble = True
model.vlm_query_path = "dataset/metadata/d3_clip_convnextl_sentences.npy"
model.seen_classes = "dataset/metadata/d3_seen_empty.json"
model.all_classes = "dataset/metadata/d3_phrases.json"
model.unseen_classes = "dataset/metadata/d3_phrases.json"
model.alpha = 0.0
model.beta = 0.1
model.novel_scale = 1.0

# Detector-internal no-score verifier. The checkpoint was trained without
# detector_score, so fusion depends on region/text matching rather than score
# calibration. The target-prompt bank matches the verifier text prompt
# "the described target is {phrase}".
model.region_verifier_enabled = True
model.region_verifier_checkpoint = "output/d3_crop_verifier_w075_no_score/verifier_best.pt"
model.region_verifier_text_path = target_prompt_bank_path
model.region_verifier_text_index = None
model.region_verifier_fusion = "logit_add"
model.region_verifier_fusion_weight = 0.25
model.region_verifier_topk_per_image = 20

train.output_dir = "./output/lami_convnext_large_12ep_lvis_zeroshot_d3_internal_verifier_w075"
dataloader.evaluator.output_dir = train.output_dir
