#!/usr/bin/env bash
# Train the 3-way policy with alpha=1.0 efficiency term. Only run AFTER the
# alpha=0 run completes or has a viable intermediate checkpoint.
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

: "${RUN_NAME:=3way_alpha1_bl32_mixture}"
: "${CONFIG:=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_3way_alpha1.yaml}"
: "${OUTPUT_DIR:=outputs/${RUN_NAME}}"
: "${N_GPUS:=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)}"
: "${WANDB_PROJECT:=rl-dllm-3way}"
: "${WANDB_RUN_ID:=${RUN_NAME}}"

export WANDB_PROJECT
export WANDB_RESUME=allow
export WANDB_RUN_ID

mkdir -p "$OUTPUT_DIR"

# shellcheck disable=SC1091
source "$HERE/_resume_loop.sh"

restart_loop accelerate launch \
  --num_processes "$N_GPUS" \
  -m train.train \
  --config "$CONFIG" \
  --output_dir "$OUTPUT_DIR" \
  --run_name "$RUN_NAME"
