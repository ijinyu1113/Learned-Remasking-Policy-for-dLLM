#!/usr/bin/env bash
# Full 1319-problem GSM8K evaluation of the existing 2-way checkpoint 700.
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

: "${TWO_WAY_CKPT_DIR:?set TWO_WAY_CKPT_DIR to the local directory containing checkpoint-700/}"
: "${CONFIG:=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml}"
: "${SAVE_PATH:=eval_results/2way_ckpt700_gsm8k_full}"
: "${SEEDS:=42}"
: "${TEMPERATURES:=1.0}"

python -m eval.pipeline "$TWO_WAY_CKPT_DIR" \
  "$CONFIG" \
  --checkpoints 700 \
  --datasets gsm8k \
  --seeds "$SEEDS" \
  --temperatures "$TEMPERATURES" \
  --save_path "$SAVE_PATH"
