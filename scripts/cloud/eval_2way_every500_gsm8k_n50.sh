#!/usr/bin/env bash
# GSM8K eval on the 2-way policy every 500 steps (500..3000), 50 problems each.
# Uses the local copy of the Drive checkpoints at $TWO_WAY_CKPT_DIR.
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

: "${TWO_WAY_CKPT_DIR:=outputs/policy2way_gdrive/policy2way}"
: "${CONFIG:=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml}"
: "${SAVE_PATH:=eval_results/2way_every500_gsm8k_n50}"
: "${CHECKPOINTS:=500,1000,1500,2000,2500,3000}"
: "${SEEDS:=42}"
: "${TEMPERATURES:=1.0}"
: "${N_TEST:=50}"

python -m eval.pipeline "$TWO_WAY_CKPT_DIR" \
  "$CONFIG" \
  --checkpoints "$CHECKPOINTS" \
  --datasets gsm8k \
  --seeds "$SEEDS" \
  --temperatures "$TEMPERATURES" \
  --save_path "$SAVE_PATH" \
  --n_test "$N_TEST"
