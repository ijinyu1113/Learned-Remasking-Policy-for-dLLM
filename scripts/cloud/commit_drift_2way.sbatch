#!/bin/bash
# Diagnostic: how often does the base model disagree with its own earlier commits
# at final-context time? Upper bound on the maximum remasking benefit.
#
# Output: eval_results/analysis/commit_drift_2way_ckpt1000.json
#SBATCH --job-name=commit_drift_2way
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=ghx4
#SBATCH --account=benv-dtai-gh
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

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

# Stage 2-way ckpt 1000 if not already
if [ ! -f outputs/2way_eval/checkpoint-1000/model.safetensors ]; then
  mkdir -p outputs/2way_eval/checkpoint-1000
  cp pretrained_checkpoints/2way_ckpt1000.safetensors outputs/2way_eval/checkpoint-1000/model.safetensors
  cp pretrained_checkpoints/2way_ckpt1000.config.json  outputs/2way_eval/checkpoint-1000/config.json
fi

# n_test=200 gives reasonable distribution stats (~30 min wall clock)
timeout 90m python -u -m scripts.analysis.commit_drift \
  --policy_config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
  --policy_ckpt outputs/2way_eval/checkpoint-1000/model.safetensors \
  --dataset gsm8k --n_test 200 \
  --out eval_results/analysis/commit_drift_2way_ckpt1000.json \
  --commit_high_conf 0.5 --final_low_conf 0.2
