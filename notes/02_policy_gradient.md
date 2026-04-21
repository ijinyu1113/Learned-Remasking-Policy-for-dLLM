# 02 — Policy Gradient (REINFORCE → GRPO)

> **Concept 2** in the CLAUDE.md concept map. Foundation for every RL-for-LLM method
> you'll touch in this repo.

## Sources

- **Primary — lecture slides**: CS 185/285 Lec 5, *Policy Gradients* — Sergey Levine, UC
  Berkeley (Spring 2026 slide deck). `rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf`
- **LLM RL application**: CS 185/285 Section 7 discussion notes, Kevin Black (Spring 2026).
  `rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-7.pdf`
- **Lecture recording** (Fall 2023): youtu.be/GKoKNYaBvM0 (Lec 5 Part 1) — *not fetched*;
  WebFetch can't pull YouTube captions. The 2023 lecture may differ slightly from the 2026
  slides.
- **Original GRPO paper**: Shao et al. 2024 (DeepSeekMath), `arxiv.org/abs/2402.03300`.

---

## One-line summary

**Upweight the log-probability of actions in high-reward trajectories; downweight the
low-reward ones.** Everything else — baselines, clipping, GRPO — is variance reduction
bolted on top.

---

## 1. Objects (nouns before verbs)

Before any math, know what every symbol *is*.

### 1.1 Neural network `f_θ`

A neural network is **a function you don't know, built from tunable parameters**.

Two pieces:

1. **Architecture** — the *form* of the function (e.g. `Linear → ReLU → Linear → Softmax`).
   You choose this; it's fixed.
2. **Parameters θ** — the numbers (weights, biases) inside the architecture. Initially
   random; found during training.

The notation `f_θ(x)` reads: *"this architecture, evaluated at input `x`, with weights
`θ`."* Change θ → change what the function computes.

Analogy: a quadratic `a x² + b x + c`. The quadratic *form* is the architecture. The
coefficients `(a, b, c)` are θ. For an NN, same story — just millions of θ instead of 3.

### 1.2 The RL basics — state, action, reward, trajectory

| Symbol | What it is | In this repo |
|---|---|---|
| `s` (state) | Whatever the agent observes at one step. | Per-position features: `(confidence, mask status, t/T)` for each position in the sequence. |
| `a` (action) | What the agent does at that step. | Per-position 3-way choice: `{keep, unmask, remask}`. |
| `r(s, a)` (reward) | Scalar feedback the environment gives. | `R_correct + α · R_efficiency`. **Sparse** — delivered only at the end of the denoising trajectory. |
| `τ` (trajectory) | The **whole sequence** of states and actions in one rollout. | One full denoising run, from all-masked to final decoded answer. |

A trajectory looks like this:

$$
\tau = (s_1, a_1, s_2, a_2, s_3, a_3, \dots, s_H, a_H)
$$

`H` is the horizon (max number of steps). In this project, `H` is the number of denoising
steps (e.g. 64).

### 1.3 Policy `π(a|s)` and policy network `π_θ(a|s)`

A **policy** is a decision rule: given a state, what do I do?

- Deterministic: `a = π(s)` — always the same action for a given state.
- **Stochastic (what we use)**: `π(a|s)` is the **probability** of choosing action `a` in
  state `s`. So for each state, the policy is a probability distribution over the action
  set.

A **policy network** `π_θ(a|s)` is a policy implemented as a neural network with
parameters θ. Feed in `s`, get out a probability for each action.

Concretely in this repo:

- **Input** per step: the per-position feature vector (§1.2).
- **Output** per step: per-position logits over `{keep, unmask, remask}`, converted to
  probabilities by softmax.

"Policy" is just jargon for "strategy" or "decision rule." Don't let the symbol spook you.

### 1.3.1 The actual policy network in this repo

File: [common/models/policy.py](common/models/policy.py). Two classes are defined; **every
experiment config uses [`DiTConfidencePolicy`](common/models/policy.py#L203)** (the
`policy_type: dit_confidence` line in every YAML under [configs/experiment_configs/](configs/experiment_configs/)).
The second class, `DiTHiddenStatePolicy`, is an unused-in-current-experiments richer
variant that takes LLaDA's hidden states instead of just confidences.

**Yes — it's a transformer.** Specifically a **DiT block** (Diffusion Transformer, from
Peebles & Xie 2022): multi-head attention + feed-forward + **adaptive layer norm (adaLN)**
conditioned on `(timestep, mask_status)`. See
[common/models/policy_layers.py:36](common/models/policy_layers.py#L36) for the block
definition.

**Forward pass of `DiTConfidencePolicy`** ([line 282](common/models/policy.py#L282)):

```
   timestep (*B,1) ──► sinusoidal_time_embedding ──► MLP ─┐
                                                          │
   mask m (*B,L) ────► nn.Embedding(2, H) ────────────────┤── (+) ──► cond (*B, L, H)
                                                          │
   confidences c (*B,L,top_p) ──► Linear(top_p → H) ──► x (*B, L, H)
                                                          │
   x, cond ──► [ RoPEDiTBlock × num_blocks ] ─────────────┤
                          ▲                               │
                          │  each block: adaLN uses cond for scale/shift (6 modulations)
                          ▼
   x ──► LayerNorm ──► Linear(H → num_actions) ──► logits (*B, L, num_actions)
```

**Defaults** (from the constructor, [common/models/policy.py:203](common/models/policy.py#L203)):

| hyperparameter | default | meaning |
|---|---|---|
| `hidden_dim` (`H`) | 128 | transformer width |
| `feedforward_dim` | 512 | FFN inner width |
| `num_heads` | 1 | attention heads |
| `num_blocks` | 1 | **single-layer** DiT, matching the Jazbec design |
| `num_actions` | 3 (2 for baseline) | `{unmask, keep, remask}` or `{unmask, keep}` |
| `confidences_top_p` | 1 | how many top-confidence values per position to feed in |

**Total parameter count**: ~0.5–1M. Less than **0.01%** of the frozen LLaDA-8B base
model — that's the whole point of the external-policy approach.

Key architectural choices:

- **RoPE** (rotary position embeddings) inside the attention — so the policy knows
  *where in the sequence* each position is. Standard modern choice.
- **adaLN** conditioning ([policy_layers.py:78](common/models/policy_layers.py#L78))
  — the conditioning vector (time + mask) is projected to **6 scale/shift modulations**
  per block (pre-attention γ/β/α, pre-FFN γ/β/α), exactly like Figure 3 in the DiT
  paper. This is how the network "knows what timestep it's at" and "which positions are
  currently masked."
- **Smart init** ([policy.py:253](common/models/policy.py#L253)): on 3-way policies,
  the output bias is initialized to `[target_logit, 0.0, -10.0]` for
  `[unmask, keep, remask]`. The `-10.0` bias makes initial `remask` probability ≈ 0, so
  the cold-start policy looks like a 2-way policy and gradually learns to remask as
  training proceeds. Safety measure against early destabilization.

So in one sentence: **the policy is a tiny single-block DiT transformer
(~1M params) with RoPE attention and adaLN conditioning on (timestep, mask status), reading
per-position confidences and outputting 3-way action logits.**

### 1.4 Trajectory distribution `p_θ(τ)`

**Q**: "What's the probability of seeing a specific trajectory τ if I run policy π_θ?"

**A**: Every trajectory is random for two independent reasons:

- The **agent** samples an action each step: `a_t ~ π_θ(·|s_t)` — depends on θ.
- The **environment** samples the next state: `s_{t+1} ~ p(·|s_t, a_t)` — independent of θ.

Multiply those step-by-step probabilities (valid because each step depends only on the
previous one — Markov property):

$$
p_\theta(\tau) = \underbrace{p(s_1)}_{\text{env: where we start}} \cdot \underbrace{\pi_\theta(a_1|s_1)}_{\text{agent}} \cdot \underbrace{p(s_2|s_1,a_1)}_{\text{env: dynamics}} \cdot \underbrace{\pi_\theta(a_2|s_2)}_{\text{agent}} \cdot \dots
$$

Compactly:

$$
p_\theta(\tau) = p(s_1) \prod_{t=1}^{H} \pi_\theta(a_t|s_t) \cdot p(s_{t+1}|s_t,a_t)
$$

The subscript θ is a reminder: change θ → change `π_θ(a_t|s_t)` at every step → different
trajectories become more/less probable.

### 1.5 Expected reward `J(θ)`

The **scalar** we want to maximize. The average total reward over all trajectories the
policy could produce:

$$
J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\!\left[\sum_{t=1}^{H} r(s_t, a_t)\right]
$$

Read this as: *"sample lots of trajectories by running `π_θ`, sum up their rewards,
average."*

`J(θ)` is a function of θ — a different θ gives a different expected reward. Our goal is
to find the θ that maximizes it.

### 1.6 What we want, in one sentence

**Find θ such that, on average, running `π_θ` earns high reward.**

Everything in the rest of this note is about *how you do gradient ascent on `J(θ)` when
you don't know the environment's transitions `p(s_{t+1}|s_t,a_t)`*.

---

## 2. The core problem

Supervised learning has labels. For each `(x_i, y_i)` pair, the loss is
`-log p_θ(y_i|x_i)`, and you know what `y_i` should be. Easy: compute gradient, step.

RL has **no labels**. You run the policy, get a trajectory, see a scalar reward at the
end. Nobody told you which actions were "correct."

Two natural-but-naive ideas:

- **Naive idea #1**: Filter to high-reward trajectories, do supervised learning on them
  (behavior cloning on good runs). Problem: throws away low-reward data; can't propagate
  any signal from "this was bad, don't do it again."
- **Naive idea #2**: Compute `∇_θ J(θ)` by directly differentiating the expectation.
  Problem: the expectation is an integral over trajectories, and the trajectory
  distribution depends on the environment's transition `p(s_{t+1}|s_t,a_t)` — which we
  don't know.

**The policy gradient trick** (§3) solves #2: it rewrites `∇_θ J(θ)` as an expectation we
can **estimate from samples**, without needing the transition dynamics.

Levine's RL loop:

```
         ┌──────────────────────┐
         │ 1. Generate samples  │
         │   (run the policy)   │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ 2. Estimate J(θ)     │
         │   (or its gradient)  │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ 3. Improve the       │
         │    policy: θ ← θ+αg  │
         └──────────┬───────────┘
                    │
                    └──────► (back to 1)
```

All the hardness is in step 3: what *is* `g = ∇_θ J(θ)`?

---

## 3. The log-derivative trick and REINFORCE (Lec 5 Part 1)

### 3.1 The identity

The one equation that unlocks everything:

$$
p_\theta(\tau) \, \nabla_\theta \log p_\theta(\tau) = p_\theta(\tau) \cdot \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} = \nabla_\theta p_\theta(\tau)
$$

Read right-to-left: **gradient of a probability = probability × gradient of its log.**
(This is just `d/dx log f(x) = f'(x)/f(x)` solved for `f'(x)`.) It looks innocent but it's
what lets us pull `∇` inside an expectation.

### 3.2 Gradient of `J(θ)`

Start from `J(θ) = ∫ p_θ(τ) r(τ) dτ` where `r(τ) = Σ_t r(s_t, a_t)`. Take `∇_θ`:

$$
\nabla_\theta J(\theta)
= \int \nabla_\theta p_\theta(\tau) \, r(\tau) \, d\tau
\stackrel{\text{trick}}{=} \int p_\theta(\tau) \, \nabla_\theta \log p_\theta(\tau) \, r(\tau) \, d\tau
= \mathbb{E}_{\tau \sim p_\theta}\!\bigl[\nabla_\theta \log p_\theta(\tau) \cdot r(\tau)\bigr]
$$

We turned `∇J` into an expectation. That's the key step — expectations can be estimated
by sample means.

### 3.3 Factorizing `log p_θ(τ)`

Recall `p_θ(τ) = p(s_1) · Π_t π_θ(a_t|s_t) · p(s_{t+1}|s_t,a_t)` (§1.4). Take `log`:

$$
\log p_\theta(\tau) = \log p(s_1) + \sum_{t=1}^{H} \log \pi_\theta(a_t|s_t) + \sum_{t=1}^{H-1} \log p(s_{t+1}|s_t,a_t)
$$

Now take `∇_θ`. The first term doesn't depend on θ → gradient is 0. The last term doesn't
depend on θ either (the environment doesn't know about our neural net) → gradient is 0.
**Only the middle term survives:**

$$
\nabla_\theta \log p_\theta(\tau) = \sum_{t=1}^{H} \nabla_\theta \log \pi_\theta(a_t|s_t)
$$

This is the moment the transition dynamics drop out. **We never needed to know them.**

### 3.4 REINFORCE

Plug back into §3.2:

$$
\boxed{\;\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta}\!\left[\left(\sum_{t=1}^{H} \nabla_\theta \log \pi_\theta(a_t|s_t)\right) \left(\sum_{t=1}^{H} r(s_t, a_t)\right)\right]\;}
$$

Estimate it by a sample mean — the **REINFORCE gradient**:

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \left(\sum_{t=1}^{H} \nabla_\theta \log \pi_\theta(a_t^{(i)}|s_t^{(i)})\right) \left(\sum_{t=1}^{H} r(s_t^{(i)}, a_t^{(i)})\right)
$$

The **REINFORCE algorithm** (3 lines):

1. Sample `N` trajectories `{τ^(i)}` from `π_θ` (run the policy).
2. Compute `∇_θ J(θ)` as above.
3. `θ ← θ + α ∇_θ J(θ)`.

Repeat.

---

## 4. The supervised-learning tie-back (Lec 5 Part 2)

Here's the connection that makes everything click.

**Supervised learning loss**, for one training pair `(x_i, y_i)`:

$$
\mathcal{L}_{\text{SL}}(\theta) = - \log p_\theta(y_i | x_i)
$$

Minimizing it pushes `p_θ(y_i|x_i)` up. Every training example contributes **equally** —
weight 1.

**Policy gradient "loss"** (pseudo-loss; see §8), for one sampled action:

$$
\mathcal{L}_{\text{PG}}(\theta) = - \log \pi_\theta(a_t^{(i)} | s_t^{(i)}) \cdot r(\tau^{(i)})
$$

Minimizing it pushes `π_θ(a_t|s_t)` up **if and only if** the reward `r(τ)` was positive.
High-reward trajectories get a big push; low-reward ones get a push in the opposite
direction.

**Same `-log p` — different weights.** SL uses weight `1` uniformly. PG uses weight
`r(τ)` (or an advantage, §7). That is the entire structural difference.

Levine's tagline: *"Policy gradient is maximum likelihood, weighted by reward."*

```
   SL gradient:    (1/N) Σ_i ∇log p_θ(y_i|x_i)          ← weight = 1
   PG gradient:    (1/N) Σ_i ∇log π_θ(τ^(i)) · r(τ^(i))  ← weight = r(τ)
```

Internalize this. Every line of `compute_loss` in this repo is "weighted supervised
learning" with progressively more refined weights.

---

## 5. What did we just do? (Lec 5 Part 2)

### 5.1 Trial and error, formalized

- Good trajectories (high `r(τ)`) are made **more likely** in the future.
- Bad trajectories (low/negative `r(τ)`) are made **less likely**.
- The policy drifts toward things that worked.

Schematic intuition (reward as a function of trajectory):

```
    high │          ~~~~~
    r(τ) │        ~~     ~~          ✓ ← sampled here: upweight
         │      ~~         ~~
         │    ~~             ~~
    low  │  ~~                 ~~    ✗ ← sampled here: downweight
    r(τ) │~~                     ~~
         └──────────────────────────→ trajectory space
```

### 5.2 Markov property is never used (!)

The derivation only used that we can factor `log π(τ) = Σ log π(a_t | s_t)`. Nowhere did
we use the Markov assumption on state transitions — that happened to be how we wrote
`p_θ(τ)`, but it didn't matter for the gradient. Replace `s_t` with an observation `o_t`
and the algorithm works unchanged in partially-observed MDPs. (Lec 5 Part 2.)

---

## 6. What's wrong with it? (Lec 5 Part 2 → 3)

Policy gradient has **high variance**. Levine's chess example (preset position, reward
= +1 win / −1 lose):

- **Lucky starts** get positive multipliers; **unlucky starts** get negative — even if the
  player's moves were identical.
- A good move gets a *negative* multiplier if you happened to blunder later in the game.
- A bad move gets a *positive* multiplier if your opponent happened to blunder later.

These all "average out" with enough samples — but the required `N` can be enormous.

---

## 7. Variance reduction (Lec 5 Part 3)

Three fixes, in increasing order of power.

### 7.1 Baselines

Subtract a constant `b` from the reward:

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_i \nabla_\theta \log p_\theta(\tau^{(i)}) \bigl[r(\tau^{(i)}) - b\bigr]
$$

This is **unbiased** (the true gradient doesn't change), because:

$$
\mathbb{E}\!\bigl[\nabla_\theta \log p_\theta(\tau) \cdot b\bigr] = b \nabla_\theta \int p_\theta(\tau)\, d\tau = b \nabla_\theta 1 = 0
$$

Subtracting a constant from everything drops the variance without biasing the gradient.
Simple choice that works well: `b = (1/N) Σ r(τ^(i))` (average reward across this batch).

**Why this matters**: if you shift all rewards by +100, the *true* gradient is unchanged
but the REINFORCE *estimator's* variance explodes (every trajectory gets a big positive
multiplier, so you're adding a lot of big vectors and the noise dominates). A baseline
(essentially subtracting the batch mean) saves you from that.

### 7.2 Causality / reward-to-go

Actions at time `t` can't affect rewards at time `t' < t`. So instead of weighting
`∇log π(a_t|s_t)` by the *whole* trajectory's reward, weight it by the **future** reward
only:

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_i \sum_t \nabla_\theta \log \pi_\theta(a_t^{(i)}|s_t^{(i)}) \; \underbrace{\left(\sum_{t'=t}^{H} r(s_{t'}^{(i)}, a_{t'}^{(i)}) - b_t\right)}_{\hat Q_t^{(i)} - b_t}
$$

`Q̂_t^{(i)}` is the **reward-to-go**. Strictly lower variance than the full-trajectory
form, still unbiased.

### 7.3 Off-policy importance weighting

Sample from a *behavior* policy `π̄` instead of the current `π_θ` (e.g. `π̄` = the policy
at the start of this update step). Reweight:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \bar\pi}\!\left[\frac{\pi_\theta(a|s)}{\bar\pi(a|s)} \, \nabla_\theta \log \pi_\theta(a|s)\, r(s,a)\right]
$$

The ratio `π_θ/π̄` is the **importance weight** — it corrects for the mismatch between
"the policy you actually sampled from" and "the policy you're optimizing." This is what
lets PPO and GRPO reuse samples across multiple gradient steps instead of re-collecting
after every update.

---

## 8. Implementing it with autograd (Lec 5 Part 4)

You don't compute `∇log π · r` explicitly. You define a **pseudo-loss** whose gradient
*is* the policy gradient, and let autograd handle everything:

$$
\tilde J(\theta) = \frac{1}{N} \sum_i \sum_t \log \pi_\theta(a_t^{(i)}|s_t^{(i)}) \cdot \hat Q_t^{(i)}
$$

For discrete actions, `-log π(a|s)` is just **cross-entropy** with the sampled action as
the label. So the implementation is one line away from supervised learning:

```python
# Supervised learning:
logits = policy(states)                                      # (N*T, D_a)
neg_logp = cross_entropy(logits, actions)                    # (N*T,)
loss = neg_logp.mean()

# Policy gradient:
logits = policy(states)
neg_logp = cross_entropy(logits, actions)
weighted = neg_logp * q_values                               # ← only new line
loss = weighted.mean()
```

Exactly one line different. That's the whole implementation.

Levine's three practical warnings (Lec 5 Part 4):

1. The gradient is **noisy** — much noisier than supervised gradients.
2. Use **much larger batches** than you'd use for supervised learning.
3. **Learning rates are hard.** Adam is OK-ish; later lectures (TRPO/PPO) give you
   PG-specific step-size rules.

---

## 9. Policy gradient for LLMs (Section 7 §2)

LLM RL is usually framed as a **one-step contextual bandit**: prompt `s`, completion `a`,
reward `r(s,a)`. You don't need multi-step MDP formalism because the LLM is
autoregressive — the per-token factorization happens inside
`log π_θ(a|s) = Σ_t log π_θ(token_t | s, tokens_{<t})`.

(Our dLLM setup is a slight generalization — see §11.5 below.)

**Vanilla (REINFORCE) estimator for LLMs:**

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s,a \sim \pi_\theta}\!\left[\nabla_\theta \log \pi_\theta(a|s) \cdot r(s,a)\right]
$$

**PPO-style importance-weighted estimator:**

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s,a \sim \bar\pi}\!\left[\frac{\pi_\theta(a|s)}{\bar\pi(a|s)} \nabla_\theta \log \pi_\theta(a|s) \cdot r(s,a)\right]
$$

Shao et al. 2024 (**GRPO**) found that for LLM RL, a **per-prompt averaging baseline**
works as well as a learned value function:

$$
b(s_i) = \frac{1}{K} \sum_{k=1}^{K} r(s_i, a_k)
$$

where `{a_1, …, a_K}` are `K` different completions sampled for the *same* prompt `s_i`.
Intuitively: *"how good is this completion compared to other completions of the same
prompt?"*

**Why this matters for our project:** GRPO replaces the expensive value-function baseline
with a cheap group-mean baseline. That's why a <1M-param policy can be trained on top of
a frozen LLaDA-8B without also training a separate value network.

**Reference-model KL regularization** (prevents reward hacking when the reward is a
learned model):

$$
\bar r(s,a) = r_\psi(s,a) - \beta\, D_{\mathrm{KL}}\bigl(\pi_\theta(a|s) \,\|\, \pi_{\text{ref}}(a|s)\bigr)
$$

---

## 10. Where it lives in this repo

File: [train/trainer.py](train/trainer.py), function `compute_loss` ([line 160](train/trainer.py#L160)).

Key inner-loop computation ([lines 290–309](train/trainer.py#L290)):

```python
coeff_1 = torch.exp(logps_timestep - old_logps_slice)           # importance ratio π_θ / π_old
coeff_2 = torch.clamp(coeff_1, 1 - self.args.epsilon, 1 + self.args.epsilon)

batch_advantages = inputs["advantages"][…].detach().view((-1,) + (1,)*(coeff_1.ndim-1))

per_timestep_loss1 = coeff_1 * batch_advantages                  # PG with importance weight
per_timestep_loss2 = coeff_2 * batch_advantages                  # clipped version
per_timestep_loss  = torch.min(per_timestep_loss1, per_timestep_loss2)   # PPO surrogate
```

Mapping to the theory above:

| Variable               | Theory term                         | Section            |
|------------------------|-------------------------------------|--------------------|
| `logps_timestep`       | `log π_θ(a_t | s_t)`                | §3, §9             |
| `old_logps_slice`      | `log π̄(a_t | s_t)` (behavior policy) | §7.3, §9          |
| `coeff_1`              | importance ratio `π_θ/π̄`            | §7.3, §9          |
| `batch_advantages`     | `Q̂_t − b` (reward-to-go minus baseline) | §7.1 + §7.2    |
| `torch.min(…clamp…)`   | **PPO clipped surrogate**           | CS 285 Lec 9       |
| `loss_acummulator -= ` | `−` sign = maximize expected reward | §3                 |

The **group-relative baseline** `b(s_i) = (1/K) Σ_k r(s_i, a_k)` lives at
[train/trainer.py:917-920](train/trainer.py#L917):

```python
advantages = rewards - mean_grouped_rewards
```

That single line is the GRPO baseline. No value network — just subtract the group mean.

---

## 11. Common confusions (for you specifically)

**1. "Policy gradient" ≠ "the gradient of the policy."** It's the gradient of **expected
reward** `J(θ)` with respect to policy parameters θ. Easy to mis-parse on first read.

**2. Why `log π` and not `π`?** Two reasons:
   - (a) The log-derivative trick (§3.1) is what turns `∇J` into an expectation — so you
     can estimate it from samples.
   - (b) `log π` in the pseudo-loss is exactly the cross-entropy loss you already know, so
     the implementation is "supervised learning weighted by advantage" — one line
     different (§4, §8).

**3. Credit assignment in *this* project is brutal.** The reward is sparse (final-answer
correctness + efficiency penalty — see [cs288_remasking_policy_plan.md §3.1](cs288_remasking_policy_plan.md)).
Meanwhile, the policy takes **per-position × per-denoising-step** actions. One scalar
reward has to be distributed across thousands of `(position, timestep)` action choices.
This is why GRPO's group baseline and the per-position `log π` factorization matter so
much — they reduce variance enough that learning is possible at all.

**4. `coeff_1 = 1` on the first gradient step.** When `π_θ = π_old` (right after
sampling), the importance ratio is exactly 1, so the PPO loss reduces to
`advantage · 1 = advantage`, and the gradient is `advantage · ∇log π_θ`. That's vanilla
REINFORCE with a baseline — §7.1 exactly. Clipping only matters after the first few
updates drift `π_θ` away from `π_old`.

**5. dLLM MDP vs. §9's contextual bandit.** Section 7 frames LLM RL as a *one-step* MDP.
That works for standard autoregressive LLMs because you can factor the whole completion's
probability as `Π_t π(token_t | context)` in a single forward pass. Our dLLM setup is
different: the policy acts **multiple times** (one action-vector per denoising step), and
the final answer is the joint product of those action sequences. So we're closer to a
proper multi-step MDP with a sparse terminal reward, even though each individual step's
action is per-position. Keep this in mind — §9's framing is a useful baseline but not
exactly our setup.

---

## 12. Check-yourself questions

Answer these before moving on. Don't Google. Show me your answers.

**Q1.** (Warmup) A Bernoulli policy has `π(0|s) = 0.3`. You sampled `a = 0` at state `s`
and the trajectory returned `R = +1`. What direction does vanilla policy gradient (no
baseline) push `π(0|s)`? Up or down? Roughly by how much (proportional to what)?

**Q2.** (Mechanism) Why do we use `log π` in the estimator rather than `π` itself?
Answer in terms of (a) the log-derivative trick, (b) what kind of estimator it gives
you, and (c) why this also makes the implementation drop-in with cross-entropy loss.

**Q3.** (Hard — motivates variance reduction) Suppose you shift every reward by +100 so
every trajectory now has `R ≥ 100`. The *true* gradient of expected reward is unchanged
(constants fall out in expectation). But the REINFORCE **estimator** gets much worse —
higher variance, slower learning, possibly wrong-signed updates in a finite-sample
regime. **Why?** And how does Levine's baseline (§7.1) fix specifically address this?

**Q4.** (Repo-grounded) In `compute_loss` ([train/trainer.py:301](train/trainer.py#L301)),
`batch_advantages` is **detached** before being multiplied by `coeff_1`. Why is that
detach critical? What would break if you forgot it? (Hint: think about what autograd
would do if `advantages` had a gradient path back to θ.)

---

## 13. What's next

Once Q1–Q4 are solid, CLAUDE.md concept 3 is *"Variance reduction: baselines,
advantages"* — but you've already seen the core of that here (§7). The next genuinely
new content is:

- **GAE** (generalized advantage estimation) — interpolates between high-variance MC and
  high-bias bootstrap. Not used in this repo (GRPO sidesteps it), but useful context.
- **Group-relative advantage** in GRPO — already covered in §9; next step is to stare at
  the actual GRPO loss in [train/trainer.py:160](train/trainer.py#L160) and match every
  term to an equation.
- **PPO clipping** — the `torch.min(coeff_1·adv, coeff_2·adv)` trick with `coeff_2`
  clamped to `[1−ε, 1+ε]`. Briefly motivated in §7.3 above; full treatment in CS 285
  Lec 9.

Then we resume the shape-tracing exercise from the starter sequence (step 2 of
CLAUDE.md's starter sequence).
