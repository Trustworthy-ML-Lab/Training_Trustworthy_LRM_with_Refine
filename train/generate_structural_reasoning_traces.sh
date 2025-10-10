#!/usr/bin/env bash
set -euo pipefail

N_SAMPLE=1

MODELS=(
  "qwen3-8b"
)

DATASETS=(
  "openr1-math-10k"
)

for MODEL in "${MODELS[@]}"; do
  TENSOR_PARALLEL_SIZE=8

  for DATASET in "${DATASETS[@]}"; do

    LOG_DIR="logs"
    mkdir -p "$LOG_DIR"
    LOG_PREFIX="${MODEL}__${DATASET}"

    echo "Running generate_structural_reasoning_traces.py --model $MODEL --dataset $DATASET --n_sample $N_SAMPLE"
    python generate_structural_reasoning_traces.py \
      --model "$MODEL" \
      --dataset "$DATASET" \
      --n_sample "$N_SAMPLE" \
      --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
      > "$LOG_DIR/${LOG_PREFIX}__structural.log" 2>&1
  done
done
