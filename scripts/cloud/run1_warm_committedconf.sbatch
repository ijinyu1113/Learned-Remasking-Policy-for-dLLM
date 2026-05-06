#!/bin/bash
# Run 1: warm-start (from 2-way ckpt 1000) + committed-token conf signal.
# Tests whether fixing the wrong policy input feature unlocks REMASK learning.
# This is the highest-priority experiment.
#
# Expected behavior if the fix works: REMASK rate climbs above 0.001 over training,
# remask_conf_mean (= mean p(committed) at remasks) is LOW (< 0.5),
# accuracy beats v8 baseline (68.8% on full GSM8K).
#SBATCH --job-name=run1_warm_committedconf
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=ghx4
#SBATCH --account=benv-dtai-gh
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export TOKENIZERS_PARALLELISM=false


module reset
module load python/miniforge3_pytorch/2.7.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rldllm-eval
cd /u/iyu1/Learned-Remasking-Policy-for-dLLM

# CUDA + HF
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=${HF_TOKEN:-dummy}

# wandb -> irl-cs288-final-project entity
export WANDB_ENTITY=irl-cs288-final-project
export WANDB_PROJECT=cs288-remasking
export WANDB_RESUME=allow
export RUN_NAME=run1_warm_committedconf
export WANDB_RUN_ID=${RUN_NAME}

# Stage warm-start init from 2-way ckpt 1000 (skip if already built)
if [ ! -f outputs/warmstart_init/model.safetensors ]; then
  python -m scripts.warm_start_3way_from_2way \
    --two_way_ckpt pretrained_checkpoints/2way_ckpt1000.safetensors \
    --out_ckpt outputs/warmstart_init/model.safetensors \
    --remask_bias -4.0
fi

# Train. ~1000 steps in 18-24 hours on this hardware.
timeout 24h python -u -m train.train \
  --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_3way_alpha0_committedconf.yaml \
  --output_dir outputs/${RUN_NAME} \
  --run_name ${RUN_NAME} \
  --warm_start_policy_path outputs/warmstart_init/model.safetensors \
  --max_steps 1500
