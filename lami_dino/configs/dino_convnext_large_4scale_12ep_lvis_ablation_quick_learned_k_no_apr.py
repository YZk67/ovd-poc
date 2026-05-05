from .dino_convnext_large_4scale_12ep_lvis_ablation_quick_base import (
    _set_tpa,
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


_set_tpa("learned_k_no_apr", apr_weight=0.0, use_rpsa=False)
