#!/bin/bash

# List of model short names to finetune
MODEL_NAMES=(
    "qwen3-1.7b"
    "qwen3-4b"
    "qwen3-8b"
)

# Ensure logs directory exists
mkdir -p logs

# Loop over each short name
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    timestamp=$(date +"%Y%m%d_%H%M%S")
    log_file="logs/finetune_Plain_${MODEL_NAME}_${timestamp}.log"

    echo "Starting normal reasoning finetune for $MODEL_NAME ..."
    accelerate launch --num_processes 8 sft_Plain.py \
        --base_model "$MODEL_NAME" > "$log_file" 2>&1
    echo "Finished finetuning for $MODEL_NAME. Logs saved to $log_file"

done
