"""Matched four-epoch single-prototype control for InstructDet on OV-LVIS.

This is the causal control for the K=5 anti-collapse system.  It inherits the
locked four-epoch screening protocol verbatim and changes only the number of
TPA prototypes.  At K=1 both APR terms are exact, graph-connected zeros, so no
additional loss-weight or gradient-routing override is needed.
"""

from .dino_convnext_large_4scale_4ep_lvis_screen import (
    dataloader,
    iterations_per_epoch,
    lr_multiplier,
    model,
    optimizer,
    train,
)


model.classifier.tpa_num_prototypes = 1

train.output_dir = "./output/instructdet_lvis_k1_matched_4ep"
dataloader.evaluator.output_dir = train.output_dir
