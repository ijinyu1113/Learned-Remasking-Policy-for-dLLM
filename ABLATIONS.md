# 3-Way Policy Ablations — Run Plan

We have one bug-free run (v8) with all design choices on simultaneously. To
support the paper's claims, we need two ablations forming a 2x2 design with
the third (warmstart, no entropy) corner left out:

|              | warm-start  | cold start  |
|--------------|-------------|-------------|
| entropy      | **v8**      | **A3**      |
| no entropy   | (skip)      | **A2**      |

Comparisons isolated by this design:
- **A2 vs A3** — does entropy bonus help on cold start? (warmstart held off)
- **A3 vs v8** — does warmstart help when entropy is on? (entropy held on)
- **A2 vs v8** — combined effect of warmstart + entropy

---

## Background

After fixing the PPO ratio bug (commit `e48bb20`), the **v8** run combines:

1. **Warm-start** from a trained 2-way checkpoint (`pretrained_checkpoints/2way_ckpt1000.safetensors`)
2. **Confidence-aware REMASK prior** (`-5 × conf_top1` added to remask logit at unmasked positions) — kept in ALL ablations (separate fix, not under test)
3. **Entropy bonus** (`entropy_coef: 0.001`) — toggled ON/OFF in this ablation
4. **3-way termination rule** (loop terminates only when block fully unmasked AND policy idle) — kept in ALL ablations

Reported v8 numbers (50-sample GSM8K screening; full eval pending):

| Checkpoint | Accuracy (50-sample) | Avg NFE |
|---|---|---|
| 1000 | 66.00% | 211 |
| 2000 | 64.00% | 68 |

Final paper numbers will come from full 1319-problem GSM8K evals on the same checkpoints.

---

## A2 — Cold start, NO entropy (most naive baseline)

**Question:** what does the bug-free 3-way training do with no help — no warmstart, no
entropy regularization?

**Config:** `BL32_3way_alpha0_no_entropy.yaml` (entropy_coef=0.0). No `--warm_start_policy_path` flag.

**Run:** ~24 hours wall on preemptible A100 to reach ckpt 1000.

```bash
cd ~/Learned-Remasking-Policy-for-dLLM
git checkout main && git pull origin main

# Pause v8 if running
tmux kill-session -t train3way 2>/dev/null
pkill -9 -f accelerate 2>/dev/null
sleep 3

tmux new -s a2 -d "cd ~/Learned-Remasking-Policy-for-dLLM && bash -c '
source \$HOME/miniconda3/etc/profile.d/conda.sh && conda activate rldllm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=cs288-remasking
export WANDB_RESUME=allow
export WANDB_RUN_ID=3way_a2_cold_no_entropy
accelerate launch --num_processes 1 -m train.train \
  --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_3way_alpha0_no_entropy.yaml \
  --output_dir outputs/3way_a2_cold_no_entropy \
  --run_name 3way_a2_cold_no_entropy \
  --max_steps 1500
' 2>&1 | tee train3way_a2.log"

sleep 15
tmux attach -t a2
```

**Full GSM8K eval at ckpt 1000 (1319 problems, ~2-3 hrs on a free A100):**

```bash
# Optional fast screening first (50 problems, ~5 min) to spot-check
python -m eval.pipeline outputs/3way_a2_cold_no_entropy \
  configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_3way_alpha0_no_entropy.yaml \
  --checkpoints 1000 --datasets gsm8k --seeds 42 --temperatures 1.0 \
  --save_path eval_results/3way_a2_screening --n_test 50 2>&1 \
  | grep -E "Accuracy:|Avg NFEs:"

# Full eval — PUT IN TMUX, takes 2-3 hrs
tmux new -s eval_a2 -d "cd ~/Learned-Remasking-Policy-for-dLLM && bash -c '
source \$HOME/miniconda3/etc/profile.d/conda.sh && conda activate rldllm
python -m eval.pipeline outputs/3way_a2_cold_no_entropy \
  configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_3way_alpha0_no_entropy.yaml \
  --checkpoints 1000 --datasets gsm8k --seeds 42 --temperatures 1.0 \
  --save_path eval_results/3way_a2_full
' 2>&1 | tee eval_a2_full.log"
```

The full eval's results land in `eval_results/3way_a2_full/...` and the
aggregator prints accuracy + NFE. To re-extract:
```bash
python -m eval.aggregate_results --results_dir eval_results/3way_a2_full
```

**What to expect:**
- If accuracy < 20%: cold start without any regularization collapses (likely because REMASK exploration dies). Validates the need for SOMETHING beyond bugfix.
- If accuracy 20-50%: bugfix alone produces a partially-working policy — useful baseline number.
- If accuracy > 50%: cold start works without help. Surprising but possible.

---

## A3 — Cold start, WITH entropy bonus

**Question:** does the entropy bonus alone (no warmstart) make 3-way training viable?

**Config:** `BL32_3way_alpha0.yaml` (entropy_coef=0.001, same as v8). No `--warm_start_policy_path` flag.

**Run:** ~24 hours wall to reach ckpt 1000.

```bash
cd ~/Learned-Remasking-Policy-for-dLLM
git pull origin main

tmux kill-session -t train3way 2>/dev/null
tmux kill-session -t a2 2>/dev/null
pkill -9 -f accelerate 2>/dev/null
sleep 3

tmux new -s a3 -d "cd ~/Learned-Remasking-Policy-for-dLLM && bash -c '
source \$HOME/miniconda3/etc/profile.d/conda.sh && conda activate rldllm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=cs288-remasking
export WANDB_RESUME=allow
export WANDB_RUN_ID=3way_a3_cold_with_entropy
accelerate launch --num_processes 1 -m train.train \
  --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_3way_alpha0.yaml \
  --output_dir outputs/3way_a3_cold_with_entropy \
  --run_name 3way_a3_cold_with_entropy \
  --max_steps 1500
' 2>&1 | tee train3way_a3.log"

sleep 15
tmux attach -t a3
```

**Full GSM8K eval at ckpt 1000 (1319 problems, ~2-3 hrs):**

```bash
# Optional 50-problem screening
python -m eval.pipeline outputs/3way_a3_cold_with_entropy \
  configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_3way_alpha0.yaml \
  --checkpoints 1000 --datasets gsm8k --seeds 42 --temperatures 1.0 \
  --save_path eval_results/3way_a3_screening --n_test 50 2>&1 \
  | grep -E "Accuracy:|Avg NFEs:"

# Full eval — TMUX, 2-3 hrs
tmux new -s eval_a3 -d "cd ~/Learned-Remasking-Policy-for-dLLM && bash -c '
source \$HOME/miniconda3/etc/profile.d/conda.sh && conda activate rldllm
python -m eval.pipeline outputs/3way_a3_cold_with_entropy \
  configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_3way_alpha0.yaml \
  --checkpoints 1000 --datasets gsm8k --seeds 42 --temperatures 1.0 \
  --save_path eval_results/3way_a3_full
' 2>&1 | tee eval_a3_full.log"

# Re-extract numbers later:
python -m eval.aggregate_results --results_dir eval_results/3way_a3_full
```

**What to expect (relative to A2):**
- If A3 > A2 by 10+pp: entropy bonus helped substantially on cold start.
- If A3 ≈ A2: entropy bonus didn't matter without warmstart (probably the policy needs something to give it real reward signal first).
- If A3 < A2: entropy bonus hurt — possibly by encouraging too much remasking that destroyed coherent outputs (the v3 "thrashing" mode).

---

## Suggested order (3-day plan, tight)

Day 1 morning → kick off **A2** training (most naive baseline). ~24 hr.
Day 1 evening → optional 50-problem screening on whatever ckpt A2 has reached.
Day 2 morning → A2 at ckpt 1000. Kill training. Start full eval on A2 ckpt 1000 (~3 hr).
                In parallel (after eval finishes): kick off **A3** training. ~24 hr.
Day 3 morning → A3 at ckpt 1000. Kill training. Start full eval on A3 ckpt 1000 (~3 hr).
                Also start full eval on **v8** ckpt 1000 if not done already.
Day 3 afternoon → fill in the 4-row table. Write Results section.

If timing is brutal: skip the 50-problem screenings and go straight to full eval — it's
cleaner anyway since the paper number will come from full eval.

---

## Final results table for the paper

All numbers below are from **full 1319-problem GSM8K eval** at ckpt 1000:

| Method                            | Warm-start | Entropy | Accuracy | NFE | Notes                |
|-----------------------------------|------------|---------|----------|-----|----------------------|
| 2-way ckpt 700 (best)             | -          | -       | _____    | ___ | Baseline (Apple)     |
| 3-way A2 (cold, no entropy)       |            |         | _____    | ___ | "naive 3-way"        |
| 3-way A3 (cold, with entropy)     |            | ✓       | _____    | ___ | + entropy            |
| 3-way v8 (full)                   | ✓          | ✓       | _____    | ___ | + warmstart + entropy|

The deltas isolate each design choice's contribution:
- **A3 − A2** = entropy bonus contribution (cold-start setting)
- **v8 − A3** = warm-start contribution (entropy-on setting)

---

## What if a run still crashes / OOMs?

The same config that's running v8 successfully is reused for A2/A3, so OOM is unlikely.
If something does break:
- OOM in compute_loss → drop `timestep_batch_size: 16 → 8` in the yaml
- Process crash → just relaunch, auto-resume kicks in from the last checkpoint
- Preempt → restart VM, relaunch tmux, training resumes from latest ckpt

---

## Don't bother with these (low ROI vs. time cost)

- "Warm-start, no entropy" cell — completes the 2x2 but each ablation costs 24 hrs.
  The two diagonal comparisons (A2 vs A3, A3 vs v8) already isolate each effect.
- Ablating the conf prior: would be informative but takes another full day; mention as
  "left for future work."
- Lower entropy coefficients (0.0001, 0.005): too noisy to detect in 1000 steps with
  the 1319-problem GSM8K eval (~3pp standard error).
- Different warm-start checkpoints (ckpt 500, 2000): tangential to the main claim.
