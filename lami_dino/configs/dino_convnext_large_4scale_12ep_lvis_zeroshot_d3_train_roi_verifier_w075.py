from .dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_desc_anchor_target_weighted_cls_only import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


d3_w075_bank_path = "dataset/metadata/d3_description_anchor_target_w075_bank_convnextl.npy"
target_prompt_bank_path = "dataset/metadata/d3_description_anchor_target_w100_bank_convnextl.npy"

model.classifier.zs_weight_path = d3_w075_bank_path
model.classifier.eval_zs_weight_path = d3_w075_bank_path
model.classifier.text_embed_path = d3_w075_bank_path
model.classifier.eval_text_embed_path = d3_w075_bank_path

model.score_ensemble = True
model.backbone.score_ensemble = True
model.vlm_query_path = "dataset/metadata/d3_clip_convnextl_sentences.npy"
model.seen_classes = "dataset/metadata/d3_seen_empty.json"
model.all_classes = "dataset/metadata/d3_phrases.json"
model.unseen_classes = "dataset/metadata/d3_phrases.json"
model.alpha = 0.0
model.beta = 0.1
model.novel_scale = 1.0

# Train the verifier inside DINO.forward(). The verifier is initialized from
# scratch unless model.region_verifier_checkpoint is provided by an override.
# Eval uses the same trainable verifier through the existing internal fusion path.
model.region_verifier_train_enabled = True
model.region_verifier_enabled = True
model.region_verifier_checkpoint = None
model.region_verifier_text_path = target_prompt_bank_path
model.region_verifier_text_index = None
model.region_verifier_train_feature_mode = "no_detector_score"
model.region_verifier_train_hidden_dim = 512
model.region_verifier_train_dropout = 0.1
model.region_verifier_same_phrase_neg_per_pos = 1
model.region_verifier_wrong_phrase_neg_per_pos = 2
model.region_verifier_neg_iou_thresh = 0.3
model.region_verifier_max_pairs = 256
model.region_verifier_train_detach_region_features = True
model.region_verifier_fusion = "logit_add"
model.region_verifier_fusion_weight = 0.25
model.region_verifier_topk_per_image = 20

model.criterion.weight_dict["loss_region_verifier"] = 0.1

# Conservative single-GPU smoke defaults. Override these for a longer D3 verifier
# training run after the loss is confirmed stable.
dataloader.train.total_batch_size = 1
dataloader.train.num_workers = 4
train.max_iter = 2000
train.eval_period = 1000
train.log_period = 20
train.checkpointer.period = 1000
train.output_dir = "./output/lami_convnext_large_12ep_lvis_zeroshot_d3_train_roi_verifier_w075"
dataloader.evaluator.output_dir = train.output_dir
