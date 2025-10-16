#!/usr/bin/env python3
"""
Manual evaluation script for LaMI-DETR training
This script can be run independently to evaluate the current training checkpoint
"""

import os
import sys
import logging
from pathlib import Path

# Add detectron2 to path
sys.path.insert(0, str(Path(__file__).parent / "detectron2"))

from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import inference_on_dataset, print_csv_format

def do_test(cfg, model):
    """Run evaluation on the model"""
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator), cfg.DDEBUG
        )
        print_csv_format(ret)
        return ret

def main():
    # Setup logger
    logger = setup_logger()
    logger.info("Starting manual evaluation...")
    
    # Load config
    config_file = "lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py"
    cfg = LazyConfig.load(config_file)
    
    # Create model
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)
    
    # Find the latest checkpoint
    output_dir = Path(cfg.train.output_dir)
    checkpoints = list(output_dir.glob("model_*.pth"))
    
    if not checkpoints:
        logger.error("No checkpoints found!")
        return
    
    # Get the latest checkpoint
    latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
    logger.info(f"Loading checkpoint: {latest_checkpoint}")
    
    # Load checkpoint
    DetectionCheckpointer(model).load(str(latest_checkpoint))
    
    # Run evaluation
    logger.info("Starting evaluation...")
    results = do_test(cfg, model)
    
    logger.info("Evaluation completed!")
    return results

if __name__ == "__main__":
    main()

