"""K=5 fixed-radius removal: a single-variable, from-scratch 4ep ablation.

Inherit the native effective-batch-32 screening protocol, including the 12ep
LR horizon, CLIP-only initialization, slot prior, APR, conflict projection and
inference settings. Change ONLY the forward radius strength, not the learned
parameter initialization. This control is allowed to collapse: rank is an
outcome, not a property this configuration promises to preserve.
"""

from .dino_convnext_large_4scale_4ep_lvis_screen import (
    dataloader,
    iterations_per_epoch,
    lr_multiplier,
    model,
    optimizer,
    train,
)


# Zero already means identity in _centered_semantic_modes: retain the five
# attention-weighted prototypes BEFORE radius normalization. It does NOT mean
# model.tpa_eval_mode_scale=0 (which would replace slots by their mean).
# This persistent buffer also keeps no-radius behavior on checkpoint reload.
model.classifier.tpa_prototype_mode_strength = 0.0

train.output_dir = "./output/instructdet_k5_no_radius_bs32_4ep_seed42"
dataloader.evaluator.output_dir = train.output_dir
