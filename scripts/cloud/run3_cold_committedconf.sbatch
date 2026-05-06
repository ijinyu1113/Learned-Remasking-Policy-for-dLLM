#!/bin/bash
# Run 3: cold-start (no warm-start) + committed-token conf signal.
# Run only if Runs 1 and 2 fail. Tests whether the warmstart was masking the issue:
# if cold-start with the corrected signal also fails, the problem is deeper than
# the policy's input feature.
#SBATCH --job-name=run3_cold_committedconf
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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=${HF_TOKEN:-dummy}

export WANDB_ENTITY=irl-cs288-final-project
export WANDB_PROJECT=cs288-remasking
export WANDB_RESUME=allow
export RUN_NAME=run3_cold_committedconf
export WANDB_RUN_ID=${RUN_NAME}

# No warmstart — cold start from random init (smart_init=-2.0 etc.)
timeout 24h python -u -m train.train \
  --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_3way_alpha0_committedconf.yaml \
  --output_dir outputs/${RUN_NAME} \
  --run_name ${RUN_NAME} \
  --max_steps 1500
