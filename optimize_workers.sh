#!/bin/bash

# Dynamic num_workers optimization script
# This script tests different num_workers settings and finds the optimal one

CONFIG_FILE="lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py"
INIT_CHECKPOINT="./pretrained_models/lami_convnext_large_12ep_lvis/model_final.pth"
NUM_GPUS=4

echo "=== Dynamic num_workers Optimization ==="
echo "Testing different num_workers settings..."
echo ""

# Function to test a specific num_workers setting
test_num_workers() {
    local workers=$1
    echo "Testing num_workers = $workers"
    
    # Create a temporary config with the test setting
    python -c "
import sys
sys.path.append('.')
from detectron2.config import LazyConfig
cfg = LazyConfig.load('$CONFIG_FILE')
cfg.dataloader.train.num_workers = $workers
cfg.train.max_iter = 100  # Short test run
cfg.train.eval_period = 200  # Skip evaluation
cfg.train.log_period = 10
cfg.train.output_dir = 'output/test_workers_${workers}'
cfg.dataloader.evaluator.output_dir = 'output/test_workers_${workers}'
LazyConfig.save(cfg, 'temp_config_${workers}.py')
"
    
    # Run short training test
    echo "Running test with num_workers=$workers..."
    timeout 300 python tools/train_net.py \
        --config-file temp_config_${workers}.py \
        --num-gpus $NUM_GPUS \
        train.init_checkpoint="$INIT_CHECKPOINT" 2>&1 | tee test_workers_${workers}.log
    
    # Extract timing information
    if [ -f "test_workers_${workers}.log" ]; then
        # Look for iteration timing
        local timing=$(grep "data_time\|iter_time" test_workers_${workers}.log | tail -5 | awk '{sum+=$2} END {print sum/NR}')
        echo "Average timing for workers=$workers: ${timing}s"
        echo "$workers,$timing" >> workers_performance.csv
    fi
    
    # Cleanup
    rm -f temp_config_${workers}.py
    rm -rf output/test_workers_${workers}
}

# Initialize performance log
echo "workers,timing" > workers_performance.csv

# Test different num_workers settings
echo "Testing different num_workers settings..."
test_num_workers 2
test_num_workers 4
test_num_workers 6
test_num_workers 8

# Find optimal setting
echo ""
echo "=== Performance Results ==="
if [ -f "workers_performance.csv" ]; then
    cat workers_performance.csv
    
    # Find the setting with best performance
    optimal=$(tail -n +2 workers_performance.csv | sort -t',' -k2 -n | head -1 | cut -d',' -f1)
    echo ""
    echo "🎯 Optimal num_workers: $optimal"
    
    # Update the main config
    echo "Updating main config with optimal setting..."
    python -c "
import sys
sys.path.append('.')
from detectron2.config import LazyConfig
cfg = LazyConfig.load('$CONFIG_FILE')
cfg.dataloader.train.num_workers = $optimal
print(f'Updated num_workers to {optimal}')
"
else
    echo "No performance data collected. Using default setting."
fi

echo ""
echo "=== Cleanup ==="
rm -f test_workers_*.log
rm -f workers_performance.csv

echo "Optimization complete!"
