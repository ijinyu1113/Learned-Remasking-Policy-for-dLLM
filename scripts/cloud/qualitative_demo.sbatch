#!/bin/bash
# Qualitative demo: run 2-way + 3-way + counterfactual remask probe on a few
# GSM8K problems and emit a markdown report ready to drop into slides / paper.
#
# Output: eval_results/analysis/qualitative_demo.md (and .json with full data).
#SBATCH --job-name=qualitative_demo
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=ghx4
#SBATCH --account=benv-dtai-gh
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

module reset
module load python/miniforge3_pytorch/2.7.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rldllm-eval
cd /u/iyu1/Learned-Remasking-Policy-for-dLLM

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=${HF_TOKEN:-dummy}
export WANDB_MODE=disabled

# Use the v8 3-way checkpoint and 2-way ckpt 2000 (best at matched compute).
timeout 50m python -u -m scripts.analysis.qualitative_demo \
  --two_way_ckpt   pretrained_checkpoints/2way_ckpt2000.safetensors \
  --two_way_config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
  --three_way_ckpt   pretrained_checkpoints/3way_v8_ckpt1000.safetensors \
  --three_way_config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_3way_alpha0.yaml \
  --n_problems 5 \
  --max_drift_per_problem 5 \
  --out eval_results/analysis/qualitative_demo
