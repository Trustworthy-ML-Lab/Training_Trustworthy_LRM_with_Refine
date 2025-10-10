#!/usr/bin/env bash
set -euo pipefail

N_SAMPLE=10

# Datasets first
DATASETS=(
  "AIME2024"
  "gpqa"
  "MATH-500"
  "gsm8k"
)

# List of normal-model names; structural counterparts will be derived
NORMAL_MODELS=(
  "Plain-qwen3-1.7b"
  "Plain-qwen3-4b"
  "Plain-qwen3-8b"
)

LOG_DIR="logs_readability"
mkdir -p "$LOG_DIR"

# Set tensor parallel size (tweak if you need different TP sizes per model)
TENSOR_PARALLEL_SIZE=8

for DATASET in "${DATASETS[@]}"; do
  for NORMAL in "${NORMAL_MODELS[@]}"; do
    # Derive the structural-model name by replacing the first occurrence of "Plain" → "ReFIne"
    STRUCTURAL=${NORMAL/Plain/ReFIne}

    LOG_PREFIX="${STRUCTURAL}__vs__${NORMAL}__${DATASET}"
    echo "Running comparison: $STRUCTURAL vs $NORMAL on $DATASET"

    python compare_readability.py \
      --structural_model "$STRUCTURAL" \
      --normal_model     "$NORMAL" \
      --dataset          "$DATASET" \
      --n_sample         "$N_SAMPLE" \
      --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
      > "$LOG_DIR/${LOG_PREFIX}.log" 2>&1
  done
done
