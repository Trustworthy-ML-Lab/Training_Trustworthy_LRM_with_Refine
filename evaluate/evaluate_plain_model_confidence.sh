#!/usr/bin/env bash
set -euo pipefail

N_SAMPLE=10

MODELS=(
  "Plain-qwen3-1.7b"
  "Plain-qwen3-4b"
  "Plain-qwen3-8b"
)

DATASETS=(
  "MATH-500"
  "AIME2024"
  "gpqa"
  "gsm8k"
)

for MODEL in "${MODELS[@]}"; do
  TENSOR_PARALLEL_SIZE=8

  for DATASET in "${DATASETS[@]}"; do
    LOG_DIR="logs_plain_model_confidence"
    mkdir -p "$LOG_DIR"
    LOG_PREFIX="${MODEL}__${DATASET}"

    echo "Running evaluate_plain_model_confidence.py --model $MODEL --dataset $DATASET --n_sample $N_SAMPLE"
    python evaluate_plain_model_confidence.py \
      --model "$MODEL" \
      --dataset "$DATASET" \
      --n_sample "$N_SAMPLE" \
      --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
      > "$LOG_DIR/${LOG_PREFIX}.log" 2>&1
  done
done
