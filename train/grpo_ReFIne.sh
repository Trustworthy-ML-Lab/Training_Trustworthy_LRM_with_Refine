#!/bin/bash

# List of model short names to train with GRPO
MODEL_NAMES=(
    "ReFIne-qwen3-1.7b-sft-only"
    "ReFIne-qwen3-4b-sft-only"
    "ReFIne-qwen3-8b-sft-only"
)

# Path to RL dataset on disk
DATASET="openr1-math-2k-rl-hard"

# Ensure logs directory exists
mkdir -p logs

# Loop over each model
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    timestamp=$(date +"%Y%m%d_%H%M%S")
    log_file="logs/grpo_${MODEL_NAME}_${timestamp}.log"

    echo "Starting GRPO training for $MODEL_NAME ..."
    accelerate launch --num_processes 8 grpo_ReFIne.py \
        --model "$MODEL_NAME" \
        --dataset "$DATASET" \
        > "$log_file" 2>&1

    echo "Finished GRPO training for $MODEL_NAME. Logs saved to $log_file"
done
