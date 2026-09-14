"""Continue the no-radius 4ep run to 8ep without changing its LR timeline.

Use --resume and the ORIGINAL no-radius output directory. Run the CPU
prepare_no_radius_8ep_resume.py preflight first; a missing last_checkpoint
would otherwise let train_net fall back to a fresh initialization.
"""

from .dino_convnext_large_4scale_4ep_lvis_no_radius import (
    dataloader,
    iterations_per_epoch,
    lr_multiplier,
    model,
    optimizer,
    train,
)


# Change only the stopping point. Inherit the 85,200-update LR horizon, batch
# accumulation, radius=0, APR, projection, RPSA and locked inference protocol.
# Keep the old output path to resume its full trainer/optimizer/scheduler state.
# Inherited checkpoint/eval periods give a 6ep checkpoint and final 8ep eval.
train.max_iter = 8 * iterations_per_epoch
