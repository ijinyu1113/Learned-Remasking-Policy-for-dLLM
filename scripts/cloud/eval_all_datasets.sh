#!/usr/bin/env bash
# Nice-to-have full eval sweep for 2-way and 3-way across all 4 datasets.
# Skip blocks you don't want by setting the corresponding SKIP_* env var.
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

: "${TWO_WAY_CKPT_DIR:?set TWO_WAY_CKPT_DIR}"
: "${THREE_WAY_CKPT_DIR:?set THREE_WAY_CKPT_DIR}"
: "${TWO_WAY_CHECKPOINT:=700}"
: "${THREE_WAY_CHECKPOINT:=last}"
: "${SEEDS:=42}"
: "${TEMPERATURES:=1.0}"
: "${CONFIG_2WAY:=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml}"
: "${CONFIG_3WAY:=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_3way_alpha0.yaml}"

DATASETS=(gsm8k math humaneval mbpp)

if [ -z "${SKIP_2WAY:-}" ]; then
  for d in "${DATASETS[@]}"; do
    echo "=== 2-way eval on $d ==="
    python -m eval.pipeline "$TWO_WAY_CKPT_DIR" "$CONFIG_2WAY" \
      --checkpoints "$TWO_WAY_CHECKPOINT" \
      --datasets "$d" \
      --seeds "$SEEDS" \
      --temperatures "$TEMPERATURES" \
      --save_path "eval_results/2way_ckpt${TWO_WAY_CHECKPOINT}_${d}_full"
  done
fi

if [ -z "${SKIP_3WAY:-}" ]; then
  for d in "${DATASETS[@]}"; do
    echo "=== 3-way eval on $d ==="
    python -m eval.pipeline "$THREE_WAY_CKPT_DIR" "$CONFIG_3WAY" \
      --checkpoints "$THREE_WAY_CHECKPOINT" \
      --datasets "$d" \
      --seeds "$SEEDS" \
      --temperatures "$TEMPERATURES" \
      --save_path "eval_results/3way_${d}_full"
  done
fi

# Merge everything so the analysis scripts have one CSV to consume.
echo "=== Aggregating everything under eval_results/ ==="
python -m eval.aggregate_results --results_dir eval_results
