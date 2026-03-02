from .dino_convnext_large_4scale_12ep_lvis import model, optimizer, lr_multiplier, train, dataloader

# Freeze backbone + slow decoder to preserve novel class awareness across epochs.
# Backbone: lr=0 (frozen CLIP features).
# Decoder: lr=0.01x — slows base-class specialization so cls_score[novel] stays
#   non-zero longer, keeping the score-ensemble gate effective past epoch 1.
# Encoder/neck/classifier: lr=1x (full speed).
optimizer.params.lr_factor_func = lambda n: (
    0.0  if "backbone" in n else
    0.01 if "decoder"  in n else
    1.0
)
model.freeze_backbone = True

train.output_dir = "./output/dino_convnext_large_ovcoco65_slow_decoder"
dataloader.evaluator.output_dir = train.output_dir
