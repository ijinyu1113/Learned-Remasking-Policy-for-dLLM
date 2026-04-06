# CS 288 Final Project Plan

## Learning Adaptive Remasking Policies for Masked Diffusion Language Models

---

## 1. One-Line Pitch

Train a lightweight external policy network to make per-token unmask/keep/remask decisions for a frozen diffusion LLM, improving both accuracy and efficiency over heuristic baselines.

---

## 2. Problem Statement

Masked diffusion LLMs (e.g., LLaDA) generate text by iteratively unmasking tokens from a fully-masked sequence. Two critical decisions happen at each step:

1. **Which positions to unmask** (scheduling)
2. **Whether to remask previously-revealed tokens** (error correction)

Current approaches either use fixed heuristics (top-k confidence, thresholding) which can't adapt to context, or train a full dual-stream architecture like RemeDi (8.9B params) which is expensive. The recent "Learning Unmasking Policies" paper (Jazbec et al., 2025) showed a tiny RL-trained policy can match heuristics for unmasking — but it doesn't support remasking.

**Our contribution:** Extend the learned unmasking policy to a 3-way action space (unmask / keep-masked / remask) so the policy can also correct earlier mistakes, without modifying the base model.

---

## 3. Method

### 3.1 MDP Formulation

- **State** $s_t$: For each position $i$:
  - Token confidence $c_t^i = \max_v p_\theta^i(v | x_t)$ from frozen dLLM
  - Status indicator: masked / unmasked
  - (Optional) Entropy of the token distribution at position $i$
  - Current timestep $t / T$

- **Action** $a_t^i \in \{0, 1, 2\}$ per position:
  - 0 = keep current state (stay masked or stay unmasked)
  - 1 = unmask (sample token from dLLM's distribution)
  - 2 = remask (convert revealed token back to [M])
  - Constraint: action 1 only valid if currently masked; action 2 only valid if currently unmasked

- **Transition**: Apply actions → run frozen dLLM on updated sequence → get new confidences

- **Reward**: $R = R_\text{correct} + \alpha \cdot R_\text{efficiency}$
  - $R_\text{correct}$: task-specific (exact match for math, pass@1 for code)
  - $R_\text{efficiency}$: $-\text{NFE} / \text{max\_NFE}$ (penalize number of forward passes)
  - $\alpha$ controls speed-accuracy tradeoff (sweep over $\alpha \in \{0, 0.3, 1, 3, 10\}$)

### 3.2 Policy Network

Single-layer transformer with adaptive layer normalization for timestep conditioning:

- **Input per position**: $[c_t^i, \text{status}^i, t/T]$ (optionally add entropy, top-2 confidence gap)
- **Output per position**: logits over 3 actions → Categorical distribution
- **Size**: ~0.5–1M parameters (<0.01% of base model)
- **Training**: GRPO (Group Relative Policy Optimization)
  - Sample $G=8$ trajectories per prompt
  - Score final outputs → compute group-relative advantages
  - Update only policy network weights

### 3.3 Key Design Choices

1. **Frozen base model**: LLaDA-8B-Instruct is never modified. This keeps the project tractable and makes results a pure test of the policy's value.

2. **Monotonicity constraint**: Total number of masked positions should decrease across steps (same constraint as RemeDi Eq. 3). Enforce by limiting remasking budget: at each step, $|\text{remask positions}| < |\text{unmask positions}|$.

3. **Confidence input for unmasked tokens**: When a position is already unmasked, feed the dLLM's current confidence for that specific revealed token (not just the max confidence). Low confidence on a revealed token = natural remasking signal.

---

## 4. Baselines

| Method | Description | Modifies base model? |
|--------|-------------|---------------------|
| Random unmasking | Unmask K random positions per step | No |
| Top-k confidence | Unmask K highest-confidence positions | No |
| Fast-dLLM | Unmask all positions above threshold $\lambda$ | No |
| Learned unmasking (Jazbec et al.) | RL-trained 2-way policy (unmask/keep) | No |
| RemeDi (oracle comparison) | Full dual-stream with SFT+RL | Yes (8.9B) |

Our method should beat the first four (same model, better decisions). We don't expect to beat RemeDi on peak accuracy (they also improve token predictions), but we should be competitive at a fraction of the training cost.

---

## 5. Experiments

### 5.1 Core Evaluation

**Datasets**: GSM8K (math), MATH500 (harder math), HumanEval (code), MBPP (code)

**Base model**: LLaDA-8B-Instruct (frozen)

**Metrics**:
- Accuracy at fixed NFE budgets (Pareto frontier: accuracy vs. NFE)
- Accuracy at fixed number of denoising steps
- Remask frequency: how often does the policy actually remask? Does it learn meaningful remasking patterns?

### 5.2 Ablation Studies

| Ablation | Question |
|----------|----------|
| 2-way vs 3-way actions | Does adding remasking help over unmask-only? |
| Policy input features | Confidence only vs. confidence + entropy vs. confidence + top-2 gap |
| Policy depth | 1-layer vs. 2-layer transformer |
| Monotonicity constraint | Strict (always decrease masks) vs. relaxed (allow temporary increases) |
| $\alpha$ sweep | Different speed-accuracy tradeoff points |

### 5.3 Analysis

- **When does remasking fire?** Plot remasking frequency across denoising timesteps. Hypothesis: more remasking in early steps (less context, more mistakes), tapering off later.
- **What gets remasked?** Analyze which token types/positions get remasked most. Are they function words? Tokens at reasoning boundaries?
- **Qualitative examples**: Show side-by-side generation trajectories with and without remasking. Find cases where remasking corrected an error that propagated in the no-remask baseline.

---

## 6. Compute Budget

### Hardware
- 1× NVIDIA GH200 (98GB VRAM) — fits LLaDA-8B-Instruct comfortably

### Cost Estimates

| Component | Estimate |
|-----------|----------|
| LLaDA inference per trajectory | ~2 sec (64 denoising steps × 1 forward pass) |
| Trajectories per GRPO step | 8 per prompt |
| Training prompts | ~15K (GSM8K + MATH train sets) |
| Forward passes per epoch | 15K × 8 × 64 = ~7.7M LLaDA forwards |
| Wall time per epoch | ~4–5 days on 1 GPU (conservative) |
| Policy network update | Negligible (< 1M params) |

### Feasibility Notes
- The bottleneck is LLaDA inference, not policy training
- Can speed up with: fewer denoising steps (32 instead of 64), smaller training set, semi-AR decoding
- Start with GSM8K-only (7.5K train examples) for initial experiments, add MATH later
- Budget 2 weeks for training experiments, which allows ~2-3 full training runs

---

## 7. Timeline (10 weeks)

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1 | Infrastructure | LLaDA-8B running on GH200, baseline inference pipeline working (random, top-k, Fast-dLLM). Reproduce baseline accuracy numbers. |
| 2 | MDP + policy network | Policy network implemented. State extraction (confidences, status) from LLaDA verified. Single trajectory rollout working end-to-end. |
| 3 | GRPO training loop | Full training loop: sample trajectories → score → compute advantages → update policy. Verify gradients flow correctly. Train 2-way policy (unmask/keep only) as sanity check. |
| **CP1** | **Checkpoint 1** | **Working 2-way policy matching Jazbec et al. results on GSM8K. Baseline numbers. Problem statement + related work written.** |
| 4 | 3-way extension | Extend to 3-way action space. Add monotonicity constraint. Start training 3-way policy on GSM8K. |
| 5 | Training + debugging | Complete GSM8K training for multiple $\alpha$ values. Debug remasking behavior (is it actually remasking? how often?). |
| 6 | Full evaluation | Run all baselines and trained policies on GSM8K, MATH500, HumanEval, MBPP. Generate Pareto frontiers. |
| 7 | Ablations | Run ablation studies (2-way vs 3-way, input features, policy depth, monotonicity). |
| 8 | Analysis | Remasking frequency analysis, qualitative trajectory examples, error case studies. |
| 9 | Writing + figures | Draft full report. All figures and tables finalized. |
| **Final** | **Final report** | **6-8 page paper, ICLR/NeurIPS workshop format** |

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Remasking rarely fires (policy learns to never remask) | Medium-High | High | Analyze early. If so, try: (a) reward shaping to encourage exploration of remasking, (b) curriculum: start with high noise tasks where remasking is more needed, (c) initialize policy with bias toward remasking. Even a negative result ("remasking doesn't help with a frozen model") is publishable. |
| Training too slow (can't finish in 2 weeks) | Medium | Medium | Start with GSM8K only (7.5K examples). Use 32 denoising steps. Reduce $G$ from 8 to 4 if needed. |
| 2-way policy doesn't reproduce Jazbec et al. | Low | High | This is our sanity check (week 3). If it fails, debug before attempting 3-way. Their code may be available — check. |
| Remasking helps but only marginally | Medium | Low | Still publishable. Show the Pareto frontier shift, analyze when/why the margin is small. |
| LLaDA-8B is too weak for reasoning tasks | Low | Medium | LLaDA-8B-Instruct gets ~63% on GSM8K with basic decoding, which is enough headroom for improvement. |

---

## 9. Novelty Argument

**What exists:**
- Learned unmasking policies (Jazbec et al.) — 2-way action, no remasking
- RemeDi — remasking via confidence scores, but requires training the whole 8.9B model
- Various heuristic remasking schemes — not learned

**What's new:**
- First work to treat remasking as an explicit learned RL action in a lightweight external policy
- Combines the efficiency of the external-policy approach (train <1M params, base model frozen) with the error-correction capability of remasking
- Directly ablatable: 2-way vs 3-way comparison on the same architecture isolates the value of learned remasking

**Why it matters:**
- If it works: you get RemeDi-style error correction for free (no architecture changes, no base model retraining)
- If remasking doesn't help with a frozen model: that's evidence that the token distributions themselves need to improve (supporting RemeDi's design choice of joint training), which is also a useful finding

---

## 10. Team Division (if 3 people)

| Person | Responsibility |
|--------|---------------|
| A | Infrastructure: LLaDA inference pipeline, baseline implementations, evaluation scripts |
| B | Policy network: architecture, GRPO training loop, 3-way action space implementation |
| C | Experiments: training runs, ablations, analysis, visualization of remasking patterns |

Overlap on writing. Everyone should understand the full pipeline.

---

## 11. Related Work

### Diffusion Language Models
- **LLaDA** (Nie et al., 2025) — Masked diffusion LLM, 8B scale, competitive with Llama-3
- **Dream** (Ye et al., 2025) — Another masked diffusion LLM at scale
- **MDLM** (Sahoo et al., 2024) — Theoretical foundations for masked discrete diffusion

### RL for Diffusion LLMs
- **d1 / diffu-GRPO** (Zhao et al., 2025) — First RL post-training for dLLMs
- **RemeDi** (Huang et al., 2025) — Dual-stream model with remasking via SFT+RL
- **Learning Unmasking Policies** (Jazbec et al., 2025) — External RL-trained policy for unmasking (our direct baseline)
- **DiFFPO** (2025) — Joint training of samplers/controllers with RL
- **TraceRL** (2025) — Trajectory-aware RL with process rewards
- **dUltra** (2026) — On-policy RL for efficient parallel decoding
- **DCoLT / LLaDOU** (2025) — Plackett-Luce unmasking policy trained with RL

### Sampling Strategies
- **Fast-dLLM** (Ben-Hamu et al., 2025) — Confidence thresholding heuristic
- **KLASS** (2025) — KL-divergence guided sampling
- **LookUM** (2025) — Lookahead unmasking via path selection

---

## 12. Stretch Goals

- **LoRA on the base model**: Jointly train the policy + LoRA adapter on LLaDA. This would let the base model's token predictions also improve, bridging the gap to RemeDi without the full dual-stream cost.
- **Transfer experiments**: Train policy on GSM8K, test on MATH/code without retraining. Does remasking transfer better or worse than unmasking-only?
- **Process reward**: Instead of outcome-only reward, add per-step reward based on how many "correct" tokens are unmasked at each step (requires ground truth, so only works on training set). Compare convergence speed.
