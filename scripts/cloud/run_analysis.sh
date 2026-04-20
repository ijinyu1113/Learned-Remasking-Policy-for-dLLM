#!/usr/bin/env bash
# Post-training analysis: runs win/loss, Pareto, learning curves, counterfactual,
# and trajectory heatmaps end-to-end. Run this AFTER all evals have completed.
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

: "${TWO_WAY_GEN:?set TWO_WAY_GEN to the 2-way *_generations.json for GSM8K}"
: "${THREE_WAY_GEN:?set THREE_WAY_GEN to the 3-way *_generations.json for GSM8K}"
: "${THREE_WAY_CONFIG:=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_3way_alpha0.yaml}"
: "${THREE_WAY_SAFETENSORS:?set THREE_WAY_SAFETENSORS to outputs/.../checkpoint-XXX/model.safetensors}"
: "${RESULTS_CSV:=eval_results/detailed_results.csv}"
: "${OUT:=eval_results/analysis}"

mkdir -p "$OUT"

echo "=== 1. Aggregate all eval_results into one CSV ==="
python -m eval.aggregate_results --results_dir eval_results

echo "=== 2. Win/loss on GSM8K ==="
python -m scripts.analysis.win_loss \
  --two_way "$TWO_WAY_GEN" \
  --three_way "$THREE_WAY_GEN" \
  --dataset gsm8k \
  --out "$OUT/win_loss_gsm8k"

echo "=== 3. Pareto plot on GSM8K ==="
python -m scripts.analysis.pareto_plot \
  --results "$RESULTS_CSV" \
  --dataset gsm8k \
  --out "$OUT/pareto_gsm8k.png"

echo "=== 4. Learning curves on GSM8K ==="
python -m scripts.analysis.learning_curves \
  --results "$RESULTS_CSV" \
  --dataset gsm8k \
  --out "$OUT/learning_curves_gsm8k.png"

echo "=== 5. Counterfactual remask ablation (100 problems) ==="
python -m scripts.analysis.counterfactual_remask \
  --policy_config "$THREE_WAY_CONFIG" \
  --policy_ckpt "$THREE_WAY_SAFETENSORS" \
  --dataset gsm8k \
  --n_test 100 \
  --out "$OUT/counterfactual_gsm8k.json"

echo "=== 6. Trajectory-level remask heatmap + conf delta ==="
python -m scripts.analysis.trajectory_stats \
  --policy_config "$THREE_WAY_CONFIG" \
  --policy_ckpt "$THREE_WAY_SAFETENSORS" \
  --dataset gsm8k \
  --n_prompts 50 \
  --out "$OUT/trajectory_stats_gsm8k"

echo "Analysis outputs written to $OUT"
