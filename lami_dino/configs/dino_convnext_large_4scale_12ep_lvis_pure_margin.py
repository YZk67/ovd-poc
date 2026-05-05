from .dino_convnext_large_4scale_12ep_lvis import (  # noqa: F401
    model, dataloader, optimizer, lr_multiplier, train,
)

# Pure-Margin (Exp-1): pairwise margin loss over confusable negatives.
# - Confusable index built offline by tools/mda/step{1,2,3}_*.py
# - filter_content_info force-injects each GT's K confusable classes into
#   the FedLoss content_inds, so the head produces scores for them.
# - Last-layer margin loss only (matches upstream zjkang/mmdetection:pure-margin).
model.margin_config = dict(
    confusable_index_path="data/mda/confusable_index.json",
    margin=0.2,
    num_negatives=3,
    warmup_iters=500,
)
model.criterion.weight_dict["loss_margin"] = 0.5
