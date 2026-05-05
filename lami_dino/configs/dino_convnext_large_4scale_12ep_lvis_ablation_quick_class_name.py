from .dino_convnext_large_4scale_12ep_lvis_ablation_quick_base import (
    _EMBED_ROOT,
    _set_static_embedding,
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


_set_static_embedding(
    f"{_EMBED_ROOT}/lvis_class_name_only_convnextl.npy",
    "class_name_only",
)
