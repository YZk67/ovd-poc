#!/usr/bin/env python3

"""
Debug script to identify unused parameters in DINO model
This script helps identify which parameters are not receiving gradients
"""

import torch
import torch.nn as nn
from detectron2.config import LazyConfig, instantiate
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath('.'))

def debug_unused_parameters():
    """Debug unused parameters in the model"""
    
    # Load config
    config_path = "lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py"
    cfg = LazyConfig.load(config_path)
    
    # Create model
    model = instantiate(cfg.model)
    model.to("cuda")
    
    print("=== Model Parameter Analysis ===")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Create dummy input
    dummy_input = [{
        "image": torch.randn(3, 800, 1333).cuda(),
        "height": 800,
        "width": 1333,
        "instances": {
            "gt_classes": torch.tensor([0, 1, 2]).cuda(),
            "gt_boxes": torch.tensor([[0, 0, 100, 100], [50, 50, 150, 150], [100, 100, 200, 200]]).cuda().float(),
        }
    }]
    
    # Forward pass
    model.train()
    outputs = model(dummy_input)
    
    print("\n=== Forward Pass Output ===")
    print(f"Output keys: {list(outputs.keys())}")
    
    # Check which parameters received gradients
    print("\n=== Parameter Gradient Analysis ===")
    
    # Get all parameters
    all_params = list(model.named_parameters())
    print(f"Total named parameters: {len(all_params)}")
    
    # Check parameters that might be unused
    unused_params = []
    for i, (name, param) in enumerate(all_params):
        if param.grad is None:
            unused_params.append((i, name, param.shape))
    
    print(f"\nParameters without gradients: {len(unused_params)}")
    for idx, name, shape in unused_params:
        print(f"  Index {idx}: {name} {shape}")
    
    # Check specific indices mentioned in error (354-358)
    print(f"\n=== Specific Problematic Indices (354-358) ===")
    for i in range(354, 359):
        if i < len(all_params):
            name, param = all_params[i]
            print(f"Index {i}: {name} {param.shape} - grad: {param.grad is not None}")
    
    # Check TPA parameters specifically
    print(f"\n=== TPA Parameter Analysis ===")
    try:
        tpa = model.transformer.decoder.class_embed[0].tpa
        print(f"TPA parameters:")
        for name, param in tpa.named_parameters():
            print(f"  {name}: {param.shape} - grad: {param.grad is not None}")
    except Exception as e:
        print(f"Could not access TPA: {e}")
    
    # Check text classifier parameters
    print(f"\n=== Text Classifier Parameter Analysis ===")
    try:
        classifier = model.transformer.decoder.class_embed[0]
        print(f"Classifier parameters:")
        for name, param in classifier.named_parameters():
            print(f"  {name}: {param.shape} - grad: {param.grad is not None}")
    except Exception as e:
        print(f"Could not access classifier: {e}")

if __name__ == "__main__":
    debug_unused_parameters()
