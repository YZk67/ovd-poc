#!/bin/bash

# Auto-resume training script
# This script automatically detects if there's a checkpoint and resumes training

CONFIG_FILE="lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py"
INIT_CHECKPOINT="./pretrained_models/lami_convnext_large_12ep_lvis/model_final.pth"
NUM_GPUS=4

echo "=== Auto-Resume Training Script ==="
echo "Config: $CONFIG_FILE"
echo "Initial checkpoint: $INIT_CHECKPOINT"
echo "GPUs: $NUM_GPUS"
echo ""

# Check if output directory exists and has checkpoints
OUTPUT_DIR=$(python -c "
import sys
sys.path.append('.')
from detectron2.config import LazyConfig
cfg = LazyConfig.load('$CONFIG_FILE')
print(cfg.train.output_dir)
" 2>/dev/null)

if [ -d "$OUTPUT_DIR" ]; then
    echo "Output directory found: $OUTPUT_DIR"
    
    # Check for last checkpoint
    if [ -f "$OUTPUT_DIR/last_checkpoint" ]; then
        LAST_CHECKPOINT=$(cat "$OUTPUT_DIR/last_checkpoint")
        echo "Last checkpoint found: $LAST_CHECKPOINT"
        
        if [ -f "$OUTPUT_DIR/$LAST_CHECKPOINT" ]; then
            echo "✅ Checkpoint file exists. Resuming training..."
            python tools/train_net.py \
                --config-file "$CONFIG_FILE" \
                --num-gpus $NUM_GPUS \
                --resume \
                train.init_checkpoint="$INIT_CHECKPOINT"
        else
            echo "❌ Checkpoint file not found. Starting fresh training..."
            python tools/train_net.py \
                --config-file "$CONFIG_FILE" \
                --num-gpus $NUM_GPUS \
                train.init_checkpoint="$INIT_CHECKPOINT"
        fi
    else
        echo "No last_checkpoint file found. Starting fresh training..."
        python tools/train_net.py \
            --config-file "$CONFIG_FILE" \
            --num-gpus $NUM_GPUS \
            train.init_checkpoint="$INIT_CHECKPOINT"
    fi
else
    echo "No output directory found. Starting fresh training..."
    python tools/train_net.py \
        --config-file "$CONFIG_FILE" \
        --num-gpus $NUM_GPUS \
        train.init_checkpoint="$INIT_CHECKPOINT"
fi
