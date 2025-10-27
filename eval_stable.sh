#!/bin/bash

# Evaluation-specific optimization script
# This script runs evaluation with minimal multiprocessing to avoid BrokenPipeError

CONFIG_FILE="lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py"
INIT_CHECKPOINT="./pretrained_models/lami_convnext_large_12ep_lvis/model_final.pth"

echo "=== Evaluation Optimization ==="
echo "Running evaluation with minimal multiprocessing..."

# Set environment variables for maximum stability
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_DISTRIBUTED_DEBUG=OFF

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run evaluation only with single GPU to avoid multiprocessing issues
echo "Running evaluation on single GPU..."
python tools/train_net.py \
    --config-file "$CONFIG_FILE" \
    --num-gpus 1 \
    --eval-only \
    train.init_checkpoint="$INIT_CHECKPOINT"

echo "Evaluation completed!"
