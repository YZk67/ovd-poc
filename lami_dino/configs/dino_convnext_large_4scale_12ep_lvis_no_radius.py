"""Resume no-radius 4ep once, evaluate at 8ep, then continue to 12ep.

Use --resume in the ORIGINAL no-radius run directory. First validate/archive
the 4ep checkpoint with prepare_no_radius_8ep_resume.py --target-epochs 12.
This avoids a planned restart at 8ep; it does not retroactively restore the
RNG or dataloader position missing from the existing 4ep checkpoint.
"""

from .dino_convnext_large_4scale_4ep_lvis_no_radius import (
    dataloader,
    iterations_per_epoch,
    lr_multiplier,
    model,
    optimizer,
    train,
)


# Only extend the stop. Retain radius=0, the original 85,200-update LR horizon,
# accumulation=2, APR/projection/RPSA and locked inference settings. Inherited
# checkpoint period=14,200 saves model_0056799.pth; eval period=28,400 runs its
# 8ep evaluation in after_step and keeps training in this same process.
train.max_iter = 12 * iterations_per_epoch
