from .dino_convnext_large_4scale_12ep_lvis_zeroshot_d3 import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


# Stress-test split: every image is evaluated against all D3 descriptions.
# Use the base d3 config for the paper-main default/full AP.
dataloader.test.dataset.names = "d3_inter_full"
train.output_dir = "./output/lami_convnext_large_12ep_lvis_zeroshot_d3_inter_full"
dataloader.evaluator.output_dir = train.output_dir
