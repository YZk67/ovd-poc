from .dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_internal_roi_verifier_w075 import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


# Two-stage D3 reranking:
# 1) keep the detector's top boxes by class-agnostic max score,
# 2) rerank each retained box against its strongest phrase candidates with the
#    ROI-trained region-description verifier.
model.region_verifier_candidate_mode = "box_phrase"
model.region_verifier_num_boxes_per_image = 300
model.region_verifier_num_phrases_per_box = 50
model.region_verifier_eval_chunk_size = 4096
model.region_verifier_candidate_only = True

train.output_dir = "./output/lami_convnext_large_12ep_lvis_zeroshot_d3_internal_roi_verifier_w075_top300_rerank"
dataloader.evaluator.output_dir = train.output_dir
