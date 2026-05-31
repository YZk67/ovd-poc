from .dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_train_alias_dn_mean_roi_verifier_w075 import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


# Stable description-conditioned adaptation: keep the visual detector and
# classifier projection fixed, and train only the text-to-query adapter used by
# query initialization and DN label embeddings. This tests whether alias-DN gains
# can be made stable without full-detector fine-tuning drift.
model.dn_multi_prototype_sampling = "mean"
model.region_verifier_enabled = False
model.region_verifier_train_enabled = False
model.criterion.weight_dict["loss_region_verifier"] = 0.0


def _content_adapter_lr(module_name: str) -> float:
    return 1.0 if module_name.startswith("content_layer") else 0.0


optimizer.params.lr_factor_func = _content_adapter_lr

train.output_dir = "./output/lami_convnext_large_12ep_lvis_zeroshot_d3_train_alias_dn_mean_content_adapter_w075"
dataloader.evaluator.output_dir = train.output_dir
