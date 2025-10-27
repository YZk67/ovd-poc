#!/bin/bash

# Training performance monitor
# Monitor GPU utilization and data loading performance

echo "=== Training Performance Monitor ==="
echo "Monitoring GPU utilization and data loading..."
echo ""

# Function to monitor GPU usage
monitor_gpu() {
    while true; do
        echo "$(date): GPU Utilization:"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
        echo ""
        sleep 10
    done
}

# Function to monitor training logs
monitor_training() {
    echo "Monitoring training logs for performance indicators..."
    tail -f training.log | grep -E "(data_time|iter_time|ETA|loss)" &
}

# Start monitoring
echo "Starting GPU monitoring..."
monitor_gpu &
GPU_PID=$!

echo "Starting training log monitoring..."
monitor_training &
LOG_PID=$!

# Cleanup function
cleanup() {
    echo "Stopping monitors..."
    kill $GPU_PID $LOG_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "Monitors started. Press Ctrl+C to stop."
echo ""

# Wait for user to stop
while true; do
    sleep 1
done
