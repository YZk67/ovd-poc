#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path

# Add detectron2 to path
sys.path.insert(0, str(Path(__file__).parent / "detectron2"))

from detectron2.evaluation.lvis_evaluation import _evaluate_predictions_on_lvis
from lvis import LVIS
from detectron2.utils.logger import setup_logger

def evaluate_lvis():
    """Evaluate LVIS results and get mAP metrics"""
    
    # Setup logger
    logger = setup_logger()
    
    # Configuration
    model_output_dir = "./output/lami_convnext_large_12ep_lvis"
    results_file = os.path.join(model_output_dir, "lvis_instances_results.json")
    lvis_gt_file = "dataset/lvis/lvis_v1_val.json"
    
    if not os.path.exists(results_file):
        logger.error(f"Results file not found: {results_file}")
        return
    
    if not os.path.exists(lvis_gt_file):
        logger.error(f"LVIS ground truth file not found: {lvis_gt_file}")
        return
    
    logger.info(f"Loading results from: {results_file}")
    
    # Load predictions
    with open(results_file, 'r') as f:
        predictions = json.load(f)
    
    logger.info(f"Loaded {len(predictions)} predictions")
    
    # Load LVIS ground truth
    logger.info(f"Loading LVIS ground truth from: {lvis_gt_file}")
    lvis_gt = LVIS(lvis_gt_file)
    
    # Evaluate
    logger.info("Starting evaluation...")
    results = _evaluate_predictions_on_lvis(lvis_gt, predictions, "bbox")
    
    # Print results
    logger.info("Evaluation completed!")
    logger.info("Results:")
    for metric, value in results.items():
        logger.info(f"  {metric}: {value:.2f}")
    
    # Save results to file
    results_summary = {}
    for metric, value in results.items():
        results_summary[metric] = float(value)
    
    with open(os.path.join(model_output_dir, "evaluation_results.json"), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"Results saved to: {os.path.join(model_output_dir, 'evaluation_results.json')}")
    
    return results

if __name__ == "__main__":
    evaluate_lvis()
