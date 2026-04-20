#!/usr/bin/env bash
# Full GSM8K evaluation of a 3-way checkpoint. Pass a specific checkpoint or
# use "last" / "first" / a comma-separated list (same semantics as eval.pipeline).
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

: "${THREE_WAY_CKPT_DIR:?set THREE_WAY_CKPT_DIR to outputs/3way_alpha0_bl32_mixture (or equivalent)}"
: "${CONFIG:=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_3way_alpha0.yaml}"
: "${CHECKPOINTS:=last}"
: "${SAVE_PATH:=eval_results/3way_alpha0_gsm8k_full}"
: "${SEEDS:=42}"
: "${TEMPERATURES:=1.0}"

python -m eval.pipeline "$THREE_WAY_CKPT_DIR" \
  "$CONFIG" \
  --checkpoints "$CHECKPOINTS" \
  --datasets gsm8k \
  --seeds "$SEEDS" \
  --temperatures "$TEMPERATURES" \
  --save_path "$SAVE_PATH"
