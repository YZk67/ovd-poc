from .dino_convnext_large_4scale_12ep_lvis_zeroshot_d3_train_alias_dn_mean_roi_verifier_w075 import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


# Stable description-conditioned adaptation with a slightly wider trainable
# surface: update the text-to-query adapter and zero-shot classifier projection,
# while freezing the backbone, neck, transformer, and box heads.
model.dn_multi_prototype_sampling = "mean"
model.region_verifier_enabled = False
model.region_verifier_train_enabled = False
model.criterion.weight_dict["loss_region_verifier"] = 0.0


def _language_head_lr(module_name: str) -> float:
    trainable_prefixes = ("content_layer", "class_embed")
    return 1.0 if module_name.startswith(trainable_prefixes) else 0.0


optimizer.params.lr_factor_func = _language_head_lr

train.output_dir = "./output/lami_convnext_large_12ep_lvis_zeroshot_d3_train_alias_dn_mean_language_head_w075"
dataloader.evaluator.output_dir = train.output_dir
