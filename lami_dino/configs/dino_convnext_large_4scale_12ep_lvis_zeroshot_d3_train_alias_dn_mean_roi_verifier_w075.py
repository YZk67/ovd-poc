from .dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_desc_anchor_mean_roi_verifier_w075 import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


# Train-time novelty knob: the classifier still uses mean aggregation over the
# multi-description alias bank, but denoising queries see one randomly sampled
# alias embedding for each known label. This makes DN reconstruct boxes under
# varied descriptions instead of the fixed class-name/query prototype.
model.dn_label_embed_source = "classifier"
model.dn_multi_prototype_sampling = "random"

# Keep the ROI verifier frozen and use it only for evaluation fusion. The train
# signal here is the standard detector loss with alias-aware DN.
model.region_verifier_train_enabled = False
model.criterion.weight_dict["loss_region_verifier"] = 0.0

# Conservative single-GPU D3 finetune defaults; override max_iter/eval_period
# for quick smoke runs.
dataloader.train.total_batch_size = 1
dataloader.train.num_workers = 4
train.max_iter = 2000
train.eval_period = 1000
train.log_period = 20
train.checkpointer.period = 1000
train.output_dir = "./output/lami_convnext_large_12ep_lvis_zeroshot_d3_train_alias_dn_mean_roi_verifier_w075"
dataloader.evaluator.output_dir = train.output_dir
