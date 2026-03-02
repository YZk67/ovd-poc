from .dino_convnext_large_4scale_12ep_lvis import model, optimizer, lr_multiplier, train, dataloader

optimizer.params.lr_factor_func = lambda module_name: 0.0 if "backbone" in module_name else 1
model.freeze_backbone = True

train.output_dir = "./output/dino_convnext_large_ovcoco65_frozen_backbone"
dataloader.evaluator.output_dir = train.output_dir
