# 3-Way Policy Ablations — Run Plan

We have one bug-free run (v8) with all four design choices on simultaneously. To
support the paper's claims about *why* each choice matters, we need at least two
ablations trained to a comparable checkpoint count (~1000 steps). This document
specifies them.

---

## Background

After fixing the PPO ratio bug (commit `e48bb20`), the **v8** run combines:

1. **Warm-start** from a trained 2-way checkpoint (`pretrained_checkpoints/2way_ckpt1000.safetensors`)
2. **Confidence-aware REMASK prior** (`-5 × conf_top1` added to remask logit at unmasked positions)
3. **Entropy bonus** (`entropy_coef: 0.001`)
4. **3-way termination rule** (loop terminates only when block fully unmasked AND policy idle)

Reported v8 numbers (50-sample GSM8K screening):

| Checkpoint | Accuracy | Avg NFE |
|---|---|---|
| 1000 | 66.00% | 211 |
| 2000 | 64.00% | 68 |

We can't say "warm-start helped" or "entropy bonus helped" without ablating these.

---

## A2 — No warm-start (cold-start with bugfix)

**Question:** does the PPO bugfix alone enable convergence, or is warm-starting from 2-way
necessary to get useful reward signal?

**Config:** identical to v8 (uses `BL32_3way_alpha0.yaml`); just **omit the
`--warm_start_policy_path` flag**. The policy starts from random init with
`smart_init=-2.0` (UNMASK), `0` (KEEP), `-4.0` (REMASK).

**Run:** ~24 hours wall on preemptible A100 to reach ckpt 1000.

```bash
cd ~/Learned-Remasking-Policy-for-dLLM
git checkout ijin && git pull origin ijin

# Pause v8 if running
tmux kill-session -t train3way 2>/dev/null
pkill -9 -f accelerate 2>/dev/null
sleep 3

export RUN_NAME=3way_a2_cold_start
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=cs288-remasking
export WANDB_RESUME=allow
export WANDB_RUN_ID=$RUN_NAME

tmux new -s a2 -d "cd ~/Learned-Remasking-Policy-for-dLLM && bash -c '
source \$HOME/miniconda3/etc/profile.d/conda.sh && conda activate rldllm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=cs288-remasking
export WANDB_RESUME=allow
export WANDB_RUN_ID=3way_a2_cold_start
accelerate launch --num_processes 1 -m train.train \
  --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_3way_alpha0.yaml \
  --output_dir outputs/3way_a2_cold_start \
  --run_name 3way_a2_cold_start \
  --max_steps 1500
' 2>&1 | tee train3way_a2.log"

sleep 15
tmux attach -t a2
```

**Eval at ckpt 1000:**

```bash
python -m eval.pipeline outputs/3way_a2_cold_start \
  configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_3way_alpha0.yaml \
  --checkpoints 1000 --datasets gsm8k --seeds 42 --temperatures 1.0 \
  --save_path eval_results/3way_a2_screening --n_test 50 2>&1 \
  | grep -E "Accuracy:|Avg NFEs:"
```

**What to expect:**
- If accuracy < 30% at ckpt 1000: warm-start was critical. Paper claim supported.
- If accuracy 30-55%: warm-start helped but bugfix did some work alone. Mention nuance.
- If accuracy ≥ 60%: bugfix alone is sufficient; warm-start is a nice-to-have not necessity.

---

## A3 — No entropy bonus

**Question:** is the entropy bonus contributing anything when the policy is warm-started
from a competent baseline? The hypothesis is "no" — the warm-started policy already has
good entropy structure inherited from 2-way training.

**Config:** new yaml `llada_8b_instruct_dit_confidence_BL32_3way_alpha0_no_entropy.yaml`,
which is identical to v8 except `entropy_coef: 0.0`.

**Run:** ~24 hours wall to reach ckpt 1000.

```bash
cd ~/Learned-Remasking-Policy-for-dLLM
git pull origin ijin

tmux kill-session -t train3way 2>/dev/null
tmux kill-session -t a2 2>/dev/null
pkill -9 -f accelerate 2>/dev/null
sleep 3

# Make sure warmstart_init exists (should from v8)
ls outputs/warmstart_init/model.safetensors || \
  python -m scripts.warm_start_3way_from_2way \
    --two_way_ckpt pretrained_checkpoints/2way_ckpt1000.safetensors \
    --out_ckpt outputs/warmstart_init/model.safetensors --remask_bias -4.0

export RUN_NAME=3way_a3_no_entropy
tmux new -s a3 -d "cd ~/Learned-Remasking-Policy-for-dLLM && bash -c '
source \$HOME/miniconda3/etc/profile.d/conda.sh && conda activate rldllm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=cs288-remasking
export WANDB_RESUME=allow
export WANDB_RUN_ID=3way_a3_no_entropy
accelerate launch --num_processes 1 -m train.train \
  --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_3way_alpha0_no_entropy.yaml \
  --output_dir outputs/3way_a3_no_entropy \
  --run_name 3way_a3_no_entropy \
  --warm_start_policy_path outputs/warmstart_init/model.safetensors \
  --max_steps 1500
' 2>&1 | tee train3way_a3.log"

sleep 15
tmux attach -t a3
```

**Eval at ckpt 1000:**

```bash
python -m eval.pipeline outputs/3way_a3_no_entropy \
  configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_3way_alpha0_no_entropy.yaml \
  --checkpoints 1000 --datasets gsm8k --seeds 42 --temperatures 1.0 \
  --save_path eval_results/3way_a3_screening --n_test 50 2>&1 \
  | grep -E "Accuracy:|Avg NFEs:"
```

**What to expect:**
- If accuracy ≈ v8's 66%: entropy bonus didn't matter much given warm-start. Honest framing: "we found the entropy bonus had marginal effect when starting from a competent policy; including it for safety."
- If accuracy < v8's 66% by 5+pp: entropy bonus was load-bearing. Strong paper claim.
- If accuracy > v8's 66%: entropy bonus was *hurting* (over-exploration). Even more interesting — paper says "we initially included entropy regularization as a precaution; ablation revealed it slightly degraded the warm-started policy."

---

## Suggested order

Day 1 morning → kick off **A2** (the higher-stakes ablation; warm-start is the central claim).
Day 1 evening → A2 should be at ~ckpt 700+. Screen at whatever it hits in 12 hr if needed.
Day 2 morning → kick off **A3** after A2 finishes (or in parallel on a teammate's GPU if available).
Day 2 evening → A3 at ckpt 1000.
Day 3 → analysis, write Results section with the 3-way ablation table.

---

## Final results table for the paper

After both ablations finish, populate this table (all 50-sample GSM8K screening at
ckpt 1000):

| Method | Accuracy | NFE | Notes |
|---|---|---|---|
| 2-way ckpt 700 (best) | 74% | ~145 | Baseline (Apple paper config) |
| 3-way A2 (cold-start) | _____ | _____ | + bugfix only |
| 3-way A3 (no entropy) | _____ | _____ | + bugfix + warm-start + conf prior |
| 3-way v8 (full) | 66% | 211 | + bugfix + warm-start + conf prior + entropy |

The deltas between rows isolate each design choice's contribution. Even if individual
deltas are small/noisy, the *direction* of each effect is paper-worthy.

---

## What if a run still crashes / OOMs?

The same config that's running v8 successfully is reused for A2/A3, so OOM is unlikely.
If something does break:
- OOM in compute_loss → drop `timestep_batch_size: 16 → 8` in the yaml
- Process crash → just relaunch, auto-resume kicks in from the last checkpoint
- Preempt → restart VM, relaunch tmux, training resumes from latest ckpt

---

## Don't bother with these (low ROI vs. time cost)

- Ablating the conf prior (A4): would be informative but takes another full day; mention as "left for future work."
- Lower entropy coefficients (0.0001, 0.005): too noisy to detect in 1000 steps with 50-sample eval.
- Different warm-start checkpoints (ckpt 500, 2000): tangential to the main claim.
