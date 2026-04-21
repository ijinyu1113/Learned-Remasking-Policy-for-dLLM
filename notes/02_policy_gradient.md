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
p_\theta(\tau) = p(s_1) \left(\prod_{t=1}^{H} \pi_\theta(a_t|s_t)\right) \left(\prod_{t=1}^{H-1} p(s_{t+1}|s_t,a_t)\right)
$$

Count check: trajectory `τ = (s_1, a_1, …, s_H, a_H)` has `H` actions and `H-1`
transitions between the `H` visited states. The two products reflect that. (Levine's
slides use a slightly different convention where the trajectory implicitly includes a
terminal state `s_{H+1}` after the last action, making the transitions product run to `H`
instead of `H-1`. Either convention works — the transition terms drop out under `∇_θ`
anyway, §3.3.)

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

**Policy gradient "loss"** (pseudo-loss; see §9), for one sampled action:

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

Two fixes, both based on the same underlying identity.

### 7.1 Baselines

#### What a baseline IS (the noun)

A **baseline** `b` is any number (or function) we *subtract from the reward* before
multiplying by the score `∇log p`. The baseline-adjusted estimator:

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta \log p_\theta(\tau^{(i)}) \cdot \bigl[r(\tau^{(i)}) - b\bigr]
$$

Compare to the raw REINFORCE estimator from §3.4, which uses `r(τ)` alone:

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta \log p_\theta(\tau^{(i)}) \cdot r(\tau^{(i)})
$$

**Three valid forms for `b`** (increasing generality; all unbiased):

| Form | Example | Used in |
|---|---|---|
| `b` = constant | batch-mean reward | Vanilla PG |
| `b = b(s_t)` | learned value function `V(s_t)` | Actor-critic, PPO |
| `b = b(s_1, …, s_t)` | history-dependent | Rare in practice |

**Only constraint**: `b` cannot depend on the action `a_t` (for reasons that become clear
in the proof, Step B).

We claim two things:

- **Claim 1 (unbiased):** `E[∇log p · (r − b)] = E[∇log p · r]`. Subtracting any valid `b`
  leaves the true gradient unchanged.
- **Claim 2 (variance may decrease):** with the *right* `b`, the estimator has lower
  variance than raw REINFORCE. (Wrong `b` can make it worse.)

We'll prove both, then do a concrete numerical example.

---

#### Proof of Claim 1 (unbiased)

**Step A — distribute the expectation across the subtraction**:

$$
\mathbb{E}\bigl[\nabla_\theta \log p_\theta(\tau) \cdot (r(\tau) - b)\bigr]
= \mathbb{E}\bigl[\nabla_\theta \log p_\theta(\tau) \cdot r(\tau)\bigr]
- \mathbb{E}\bigl[\nabla_\theta \log p_\theta(\tau) \cdot b\bigr]
$$

The first term is the true gradient `∇J(θ)` (from §3). For unbiasedness we need the
**second term to be zero**.

**Step B — if `b` is a constant, pull it out of the expectation**:

$$
\mathbb{E}\bigl[\nabla_\theta \log p_\theta(\tau) \cdot b\bigr]
= b \cdot \mathbb{E}\bigl[\nabla_\theta \log p_\theta(\tau)\bigr]
$$

(This is why `b` can't depend on `a_t`: if it did, you couldn't pull it out, and the
rest of the argument fails. For `b(s_t)` see "State-dependent baselines" below — same
argument with a conditional expectation.)

**Step C — write the remaining expectation as an integral**:

$$
\mathbb{E}\bigl[\nabla_\theta \log p_\theta(\tau)\bigr]
= \int p_\theta(\tau)\, \nabla_\theta \log p_\theta(\tau)\, d\tau
$$

**Step D — log-derivative trick in reverse** (`p · ∇log p = ∇p`):

$$
= \int \nabla_\theta p_\theta(\tau)\, d\tau
$$

**Step E — Leibniz's rule** (swap gradient and integral):

$$
= \nabla_\theta \int p_\theta(\tau)\, d\tau
$$

**Step F — probabilities normalize to 1 by definition**:

$$
= \nabla_\theta (1) = 0
$$

So `E[∇log p] = 0` under its own distribution — always, for any valid probability
density. This is called the **score-function identity**.

**Combining**: `E[∇log p · b] = b · 0 = 0`. Step A becomes:

$$
\mathbb{E}[\nabla_\theta \log p_\theta(\tau) \cdot (r - b)] = \mathbb{E}[\nabla_\theta \log p_\theta(\tau) \cdot r] - 0 = \nabla_\theta J(\theta)
$$

True gradient preserved. ∎

**Why "b cannot depend on `a_t`":** if `b` depended on the action, you couldn't pull it
out at Step B, and the conditional expectation over `a_t` wouldn't vanish. Making `b`
action-dependent would **bias** the gradient. That's why value functions `V(s)` are OK
(they depend only on state) but `Q(s, a)` is not (depends on action).

**State-dependent baselines `b(s_t)`** work because at step `t`, `b(s_t)` is a constant
given the history up to `s_t`. Same Steps B–F apply to the *conditional* expectation, and
the unconditional expectation is 0 by the tower rule. (Same move as §7.2's causality
argument — we'll use it there.)

---

#### Proof of Claim 2 (which `b` minimizes variance?)

Unbiased ≠ low variance. You can pick a bad `b` and make variance **worse**. Here's how
to find the best `b`.

**Setup**: variance of our estimator (dropping the subscripts for readability):

$$
\mathrm{Var}\bigl[\nabla\log p \cdot (r - b)\bigr]
= \mathbb{E}\bigl[(\nabla\log p)^2 (r-b)^2\bigr] - \Bigl(\underbrace{\mathbb{E}[\nabla\log p \cdot (r-b)]}_{\,=\,\nabla J,\text{ by Claim 1}}\Bigr)^2
$$

The second term `(∇J)²` **doesn't depend on `b`** (because Claim 1 says the expectation
is unchanged by `b`). So minimizing variance = minimizing the first term:

$$
\min_b \mathbb{E}\bigl[(\nabla\log p)^2 \cdot (r-b)^2\bigr]
$$

This is a quadratic in `b`. Set derivative to 0:

$$
\frac{d}{db} \mathbb{E}\bigl[(\nabla\log p)^2 (r-b)^2\bigr]
= -2\, \mathbb{E}\bigl[(\nabla\log p)^2 (r - b)\bigr] = 0
$$

Solving:

$$
\boxed{\;b^\star = \frac{\mathbb{E}\bigl[(\nabla\log p)^2 \cdot r\bigr]}{\mathbb{E}\bigl[(\nabla\log p)^2\bigr]}\;}
$$

This is a `(∇log p)²`-**weighted average** of `r`. Hard to compute in practice (needs
per-sample score magnitudes).

**Simple, near-optimal choice**: ignore the weights and use `b = E[r]` — just the mean
reward. Good when `(∇log p)²` is roughly uniform across samples (often approximately
true). Levine's phrasing: *"average reward is not the best baseline, but it's pretty
good."* ∎

---

#### Concrete example (why the variance actually goes down)

Three rollouts with total rewards `r ∈ {+100, +110, +120}`. Policy log-probs give us
three gradient-vector directions `g_1, g_2, g_3` (`g_i = ∇log p_θ(τ^(i))`; we leave them
abstract because the baseline doesn't change them).

**Without baseline** (raw REINFORCE):

| trajectory | reward | multiplier on `g_i` | contribution to gradient |
|---|---|---|---|
| τ_1 | +100 | **+100** | `+100 · g_1` |
| τ_2 | +110 | **+110** | `+110 · g_2` |
| τ_3 | +120 | **+120** | `+120 · g_3` |

All three contributions are big positive scalars times their respective `g_i`. The
estimator is `(100 g_1 + 110 g_2 + 120 g_3) / 3`. Dominated by the common-mode `+110`
offset; the *relative* ordering (`τ_3 > τ_2 > τ_1`) is buried in that offset.

Across many batches, different batches give different specific values of `g_i` (random
samples), and multiplying each by `~110` amplifies that randomness → high variance.

**With baseline `b = 110`** (batch mean):

| trajectory | `r − b` | multiplier | contribution |
|---|---|---|---|
| τ_1 | +100 − 110 | **−10** | `−10 · g_1` |
| τ_2 | +110 − 110 | **0** | `0 · g_2` |
| τ_3 | +120 − 110 | **+10** | `+10 · g_3` |

Estimator: `(−10 g_1 + 0 g_2 + +10 g_3) / 3`. Same *direction* as before (τ_3 pushed up,
τ_1 pushed down), but **magnitudes are ~10× smaller**. The randomness in `g_i` across
batches still matters, but it's multiplied by small numbers → small variance.

**Interpretation**:

- The baseline **centers** the signal. A reward isn't "good" absolutely — it's "good
  relative to peers from the same policy."
- An action that ties the average gets zero update. Beat the average → push up.
  Underperform → push down.
- You kept the **signal** (relative differences) and killed the **common-mode offset**
  (the "+110 because all rewards happen to be around 100" bias).

---

#### How this lives in this repo (GRPO's baseline)

File: [train/trainer.py:917-920](train/trainer.py#L917):

```python
# Normalize the rewards to compute the advantages
advantages = rewards - mean_grouped_rewards
```

This is the **GRPO baseline** (Shao et al. 2024). `mean_grouped_rewards` is the
**per-prompt** mean of `K` completions sampled for the same prompt. So for each
completion `a_i` to prompt `s`:

$$
\hat A_i = r(s, a_i) - \underbrace{\frac{1}{K}\sum_{k=1}^{K} r(s, a_k)}_{\text{per-prompt mean}}
$$

**Key differences from Levine's "average reward" baseline**:

- **Per-prompt, not per-batch**: each prompt has its own baseline, computed from the
  `K=8` completions of that prompt only. Two different prompts don't share a baseline.
- **Why?** Prompts can have wildly different reward scales. An easy GSM8K problem might
  have all `K=8` completions getting +1; a hard problem might have all getting +0.5.
  Using a global batch mean would conflate "this prompt is hard" with "this completion
  is bad for this prompt." Per-prompt baseline cleanly isolates "is this completion
  good *for this prompt*?"
- **Formally**: this is a **state-dependent** baseline `b(s)`, not a constant. Still
  unbiased by the conditional-expectation argument above.

**One consequence to remember** (ties back to §7.1 Claim 1): within a group of K
completions for a single prompt, if all K have identical reward, the group mean equals
each reward, so **every advantage is zero**. No learning signal from that prompt.
[train/trainer.py:148-152](train/trainer.py#L148) explicitly skips the optimizer step in
that case (saves compute).

### 7.2 Causality / reward-to-go

Starting claim: actions at time `t` can't affect rewards at time `t' < t`. So we should
weight `∇log π(a_t|s_t)` by the **future** reward only, not the *whole* trajectory's
reward. Here's how this is derived carefully — it changes the *structure* of the
expression from a product-of-sums to a sum-of-products, so we need to show every step.

**Step 1 — start from the product-of-sums form** (§3.4):

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta}\!\left[\,\underbrace{\left(\sum_{t=1}^{H} \nabla_\theta\log\pi_\theta(a_t|s_t)\right)}_{\text{sum of gradients}} \cdot \underbrace{\left(\sum_{t=1}^{H} r(s_t, a_t)\right)}_{\text{sum of rewards}}\,\right]
$$

**Step 2 — expand the product with FOIL.** Every term in the first sum multiplied by
every term in the second:

$$
\left(\sum_{t=1}^{H} \nabla_\theta\log\pi_\theta(a_t|s_t)\right) \left(\sum_{t=1}^{H} r(s_t, a_t)\right)
= \sum_{t=1}^{H} \sum_{t'=1}^{H} \nabla_\theta\log\pi_\theta(a_t|s_t) \cdot r(s_{t'}, a_{t'})
$$

This is a **double sum over all pairs `(t, t')`**. Structurally identical to Step 1, just
written out. Split the pairs into three groups based on the relationship between `t` and
`t'`:

- `t' > t`: reward arrived *after* the action → keep.
- `t' = t`: reward at the same step as the action → keep.
- `t' < t`: reward arrived *before* the action → we'll show this vanishes.

**Step 3 — causality argument: the `t' < t` cross-terms are zero in expectation.** Claim:

$$
\mathbb{E}_{\tau \sim p_\theta}\!\bigl[\nabla_\theta\log\pi_\theta(a_t|s_t) \cdot r(s_{t'}, a_{t'})\bigr] = 0, \qquad \text{when } t' < t
$$

**Why:** `r(s_{t'}, a_{t'})` is a function of `(s_{t'}, a_{t'})` only — both of which were
realized *before* `a_t` was sampled. So conditional on the trajectory history up to `s_t`,
the reward `r_{t'}` is a *constant* (it's already been observed). We can pull it out of
the inner expectation over `a_t`:

$$
\mathbb{E}_{a_t \sim \pi_\theta(\cdot|s_t)}\!\bigl[\nabla_\theta\log\pi_\theta(a_t|s_t) \cdot r_{t'} \,\big|\, \text{history}\bigr]
= r_{t'} \cdot \mathbb{E}_{a_t \sim \pi_\theta(\cdot|s_t)}\!\bigl[\nabla_\theta\log\pi_\theta(a_t|s_t)\bigr]
$$

Now the inner expectation is the expected **score function** under its own distribution:

$$
\mathbb{E}_{a_t \sim \pi_\theta}\!\bigl[\nabla_\theta\log\pi_\theta(a_t|s_t)\bigr]
= \int \pi_\theta(a|s_t)\, \nabla_\theta\log\pi_\theta(a|s_t)\, da
\stackrel{\text{log-deriv}}{=} \int \nabla_\theta \pi_\theta(a|s_t)\, da
\stackrel{\text{Leibniz}}{=} \nabla_\theta \!\!\underbrace{\int \pi_\theta(a|s_t)\, da}_{=1}
= \nabla_\theta 1 = 0
$$

Same identity as §7.1: probabilities normalize to 1 by definition, so their gradient is
0. This gives us `r_{t'} · 0 = 0` conditional on the history. By the **tower rule**
(iterated expectation), the unconditional expectation is also 0:

$$
\mathbb{E}\bigl[\nabla_\theta\log\pi_\theta(a_t|s_t) \cdot r_{t'}\bigr]
= \mathbb{E}\!\bigl[\underbrace{\mathbb{E}[\nabla_\theta\log\pi_\theta(a_t|s_t) \cdot r_{t'} \mid \text{history}]}_{=\,0}\bigr] = 0
$$

**Step 4 — drop the zero-in-expectation pairs.** What remains is only `t' ≥ t`:

$$
\sum_{t=1}^{H} \sum_{t' \geq t} \nabla_\theta\log\pi_\theta(a_t|s_t) \cdot r(s_{t'}, a_{t'})
$$

**Step 5 — re-group.** The inner sum over `t' ≥ t` doesn't involve `∇log π_θ(a_t|s_t)`,
so factor it out of the inner sum:

$$
= \sum_{t=1}^{H} \nabla_\theta\log\pi_\theta(a_t|s_t) \cdot \underbrace{\left(\sum_{t'=t}^{H} r(s_{t'}, a_{t'})\right)}_{\hat Q_t}
$$

This is now a **sum of products** — outer sum over `t`, and each term is
`∇log π_θ(a_t|s_t) · Q̂_t`. That's the structural change you spotted: product-of-sums →
sum-of-products, bridged by Steps 2–4.

**Step 6 — combine with the baseline from §7.1.** Subtracting a per-step baseline `b_t`
from `Q̂_t` doesn't bias the gradient (same argument as §7.1, applied per step):

$$
\boxed{\;\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_i \sum_t \nabla_\theta \log \pi_\theta(a_t^{(i)}|s_t^{(i)}) \cdot \underbrace{\bigl(\hat Q_t^{(i)} - b_t\bigr)}_{\text{advantage }\hat A_t^{(i)}}\;}
$$

**Terminology:**

- `Q̂_t^{(i)} = Σ_{t' ≥ t} r_{t'}^{(i)}` is the **reward-to-go** from step `t` onward in
  trajectory `i`.
- `Â_t^{(i)} = Q̂_t^{(i)} - b_t` is the **advantage**: how much better did this action
  do than baseline, looking only at future reward.

**Why variance decreases:**

The steps dropped terms whose *expectation* was zero. But in any finite sample, those
"should-be-zero" terms aren't exactly zero — they have variance. Removing them removes
noise without changing the true gradient.

Analogy: averaging `[5, 5, 5, 5]` vs `[5, 5, 5, 5] + (zero-mean noise per entry)`. Same
mean; the first has zero variance, the second has positive variance. Reward-to-go drops
the noisy no-signal terms.

**In one sentence**: the original form attributes a trajectory's total reward to *every*
action in it; reward-to-go attributes it only to actions that causally preceded the
reward. Same expectation, strictly lower variance.

---

## 8. Off-policy importance weighting

**Purpose: sample efficiency, not variance reduction.** Sampling trajectories from an LLM
is expensive — seconds per rollout, multi-GPU forward passes. Importance weighting lets
us reuse one expensive batch of rollouts across many gradient steps, paying the sampling
cost once instead of K times.

**Vocabulary:**

- **Target policy** `π_θ` — the one being *updated*. Parameters θ change with every
  gradient step.
- **Behavior policy** `π̄` (also `π_old`) — the policy that *produced the trajectories*
  we're training on. Its parameters are *frozen*.
- **On-policy** algorithm: `π̄ = π_θ` at all times. Every gradient step is followed by
  fresh sampling.
- **Off-policy** algorithm: `π̄` and `π_θ` can differ. Requires a correction factor (§8.1).

### 8.1 The importance sampling identity

We want an expectation under `π_θ` but we have samples from `π̄`.

**Step 1 — start from the definition** of expectation:

$$
\mathbb{E}_{a \sim \pi_\theta}[f(a)] = \int \pi_\theta(a)\, f(a)\, da
$$

**Step 2 — multiply and divide by `π̄(a)`** (valid wherever `π̄(a) > 0`):

$$
= \int \pi_\theta(a)\, f(a) \cdot \frac{\bar\pi(a)}{\bar\pi(a)}\, da
= \int \bar\pi(a) \cdot \underbrace{\frac{\pi_\theta(a)}{\bar\pi(a)}}_{\text{importance ratio}} \cdot f(a)\, da
$$

**Step 3 — recognize the expectation under `π̄`**:

$$
= \mathbb{E}_{a \sim \bar\pi}\!\left[\frac{\pi_\theta(a)}{\bar\pi(a)} \cdot f(a)\right]
$$

Sampling `a ~ π̄` and multiplying the integrand by the ratio `π_θ/π̄` gives an
**unbiased** estimate of the expectation under `π_θ`. The samples live "under π̄" but
the correction re-weights them to "as if under π_θ."

**Applied to the policy gradient**:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\bigl[\nabla_\theta \log \pi_\theta(\tau) \cdot r(\tau)\bigr]
= \mathbb{E}_{\tau \sim \bar\pi}\!\left[\frac{\pi_\theta(\tau)}{\bar\pi(\tau)} \cdot \nabla_\theta \log \pi_\theta(\tau) \cdot r(\tau)\right]
$$

**Caveat**: unbiased ≠ low variance. Importance weighting can *increase* variance if the
ratio is extreme. §8.3 handles that with clipping.

### 8.2 Why K gradient steps ≠ one big step

A natural question: if we're going to do `K` gradient steps on one batch, why not just
take *one* step with `K×` the learning rate? Answer: **the gradient depends on θ**, and
re-evaluating it at each intermediate `θ_k` gives a fundamentally different (better)
update path.

**K small steps** (re-evaluating the gradient each time):

$$
\theta_1 = \theta_0 + \alpha \cdot \nabla J(\theta_0)
$$
$$
\theta_2 = \theta_1 + \alpha \cdot \nabla J(\theta_1)
$$
$$
\vdots
$$
$$
\theta_K = \theta_{K-1} + \alpha \cdot \nabla J(\theta_{K-1})
$$

**One big step** (gradient evaluated only at `θ_0`):

$$
\theta_K^{\text{big}} = \theta_0 + K\alpha \cdot \nabla J(\theta_0)
$$

These are equal **only if** `∇J(θ_0) = ∇J(θ_1) = … = ∇J(θ_{K-1})` — i.e., the gradient
is *constant* over the path. That's true only on a flat landscape (Hessian = 0). In
general, different θ values give different gradients, so re-evaluating at `θ_0, θ_1,
…, θ_{K-1}` uses fresh information at each step.

**Intuition — walking down a curved hill:**

- One big step = jump blindly in the direction of the slope at your current position.
  If the valley curves, you might overshoot and end up uphill on the other side.
- K small steps = walk a bit, check the slope again, walk a bit more. You follow the
  curve of the landscape.

**What specifically changes with θ here?** The score function `∇log π_θ(a|s)`. At `θ_0`
it tells you "which direction makes this sampled action more likely under `π_{θ_0}`."
After a gradient step, `π_θ` has changed, so the *same* sampled action gets a different
`∇log π_{θ_1}(a|s)`. Re-evaluating the gradient at the new θ uses fresh information
about the current policy.

Second effect: **adaptive optimizers** (Adam, RMSProp) maintain state (momentum, running
variance estimates) that updates with each step. One big step can't replicate that
evolution — you'd lose adaptive step sizing.

**The PPO/GRPO optimization pattern:**

1. Sample one expensive batch of rollouts from `π_θ` → freeze a snapshot as `π̄`.
2. Take `K` gradient steps on `π_θ` using that batch, re-evaluating `∇log π_{θ_k}` at
   each step. Apply the importance ratio `π_{θ_k}/π̄` to correct for the drift.
3. After `K` steps, re-sample fresh rollouts with the updated `π_θ`. Repeat.

You get `K` rounds of fresh gradient evaluation for the price of **one** batch of
rollouts.

### 8.3 Clipping (the PPO safety net)

**The problem with raw importance weighting**: if `π_θ` drifts far from `π̄`, the ratio
`π_θ/π̄` can blow up (or go near zero) on some samples. A few large-ratio terms dominate
the estimator → high variance, unstable training.

**Fix — PPO's clipped surrogate**: clip the ratio into `[1-ε, 1+ε]` and take the **more
pessimistic** of the clipped and unclipped objectives:

$$
L^{\text{PPO}}(\theta) = \mathbb{E}\!\left[\min\!\Bigl(\underbrace{\frac{\pi_\theta}{\bar\pi} \cdot \hat A}_{\text{unclipped}},\; \underbrace{\operatorname{clip}\!\Bigl(\tfrac{\pi_\theta}{\bar\pi},\, 1-\epsilon,\, 1+\epsilon\Bigr) \cdot \hat A}_{\text{clipped}}\Bigr)\right]
$$

- When `π_θ/π̄ ∈ [1-ε, 1+ε]`: clipped and unclipped agree → gradient is normal.
- When outside: clipping activates → the loss is capped, and the gradient is small.
  Prevents a single high-ratio sample from blowing up the update.

**In this repo** ([train/trainer.py:290](train/trainer.py#L290)):

```python
coeff_1 = torch.exp(logps_timestep - old_logps_slice)        # raw ratio π_θ/π̄
coeff_2 = torch.clamp(coeff_1, 1 - self.args.epsilon, 1 + self.args.epsilon)
per_timestep_loss = torch.min(coeff_1 * batch_advantages,
                              coeff_2 * batch_advantages)
```

Full treatment of PPO/TRPO: CS 285 Lec 9 (Advanced Policy Gradients).

---

## 9. Implementing it with autograd (Lec 5 Part 4)

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

## 10. Policy gradient for LLMs (Section 7 §2)

Standard LLM RL is framed as a **one-step contextual bandit**:

- **State** `s` = the prompt.
- **Action** `a` = the **whole completion** (all tokens, as a single action).
- **Reward** `r(s, a)` = scalar, evaluated at the end (RLHF reward model, math correctness,
  code-ran-or-not, etc.).

We'll unpack two aspects of this framing that are not obvious on first read: (§10.1) why
multi-step MDP formalism isn't needed, and (§10.2) what GRPO's `b(s_i)` baseline is
replacing from pre-GRPO LLM RL.

### 10.1 Why no multi-step MDP?

**What a multi-step framing WOULD look like.** In a standard multi-step MDP you'd have:

- `s_t` = context at token step `t` (prompt + tokens generated so far).
- `a_t` = the single next token.
- Transition `p(s_{t+1} | s_t, a_t)` = how the state evolves after generating the token.
- Reward `r(s_t, a_t)` at each step (possibly only at the final one).
- Trajectory `τ = (s_1, a_1, s_2, a_2, …, s_T, a_T)` with `T` tokens.

You'd apply full REINFORCE / reward-to-go / per-step GAE with per-token actions.

**Why you can collapse it into one step.** Two enabling facts make the one-step framing
equivalent to the multi-step one:

**Fact 1 — autoregressive factorization.** The completion probability factorizes into
per-token probabilities, computable in a single forward pass:

$$
\pi_\theta(a \mid s) = \prod_{t=1}^{T} \pi_\theta\!\bigl(\text{token}_t \,\big|\, s,\, \text{tokens}_{<t}\bigr)
$$

So `log π_θ(a | s) = Σ_t log π_θ(token_t | s, tokens_{<t})` — just run the model once on
the full sequence and sum token-level log-probs. The "one action" is already secretly
per-token structured; we're just bundling the sum into a single log-prob number.

**Fact 2 — deterministic transitions.** In vanilla text LLM RL, there's no stochastic
environment dynamics. Given the prompt and the tokens generated so far, the next "state"
is `(prompt, tokens_{≤t+1})` — **deterministic**. No dice rolls, no opponent, no noise.
So there's nothing to model between tokens.

Combine the two: the completion probability fully factorizes (Fact 1), and no
environment-side stochasticity needs tracking (Fact 2). You can treat the whole
completion as one "action," compute one scalar reward, and use a one-step PG estimator.

**When you DO need multi-step for LLMs.** Two cases:

1. **Intermediate (process) rewards.** If the reward model gives feedback per reasoning
   step (not just the final answer), you have per-step `r_t` and reward-to-go / GAE
   starts mattering. CS 288 Section 7 flags this: *"The only reason to use a multi-step
   MDP is to support intermediate rewards, but we won't cover those in this section."*
2. **Stochastic environment between steps.** Uncommon for pure text generation, but
   applies to agentic settings (tool calls returning non-deterministic observations) —
   and applies to **us** (see below).

**Why our dLLM setup is genuinely multi-step.** The policy acts once per denoising
**step**, not once per token. At step `t`:

- State `s_t` includes per-position confidences, mask status, timestep `t/T` — these
  depend on what the **frozen LLaDA** just produced, which has stochasticity baked in
  (the dLLM's token predictions are a distribution, and observed confidences depend on
  how the sequence was partially filled in).
- Action `a_t` is the per-position 3-way choice.
- `s_{t+1}` depends on both `a_t` and the frozen dLLM's stochastic next output.

That's a genuine multi-step MDP with stochastic transitions (§12.5 common-confusion
#5). The clean contextual-bandit framing of Section 7 doesn't apply to our setting —
but it does apply to standard RLHF, GRPO-on-math, DeepSeek-style training.

### 10.2 The PG estimator for LLMs

**Vanilla (REINFORCE) estimator:**

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s,a \sim \pi_\theta}\!\left[\nabla_\theta \log \pi_\theta(a|s) \cdot r(s,a)\right]
$$

**PPO-style importance-weighted estimator** (from §8):

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s,a \sim \bar\pi}\!\left[\frac{\pi_\theta(a|s)}{\bar\pi(a|s)} \nabla_\theta \log \pi_\theta(a|s) \cdot r(s,a)\right]
$$

Same structure as everything in §3 and §8 — the contextual-bandit wrapper just means
we're treating the prompt-completion as a single `(s, a)` pair.

### 10.3 The GRPO baseline `b(s_i)`

#### First: what a baseline IS (recalling §7.1)

From §7.1: a **baseline** `b` is a number (or function of state) that we **subtract from
the reward** before multiplying by the score `∇log π`. It is unbiased as long as `b` does
not depend on the action.

**A baseline does NOT replace the reward.** The reward `r(s, a)` is still there. The
baseline is subtracted from it to form the **advantage**:

$$
\hat A(s, a) = r(s, a) - b(s)
$$

The PG estimator then uses `Â` — a *centered* version of `r` — as the scalar multiplier
on `∇log π`. The roles:

- `r(s, a)` = reward. A property of the environment. Unchanged by any baseline choice.
- `b(s)` = baseline. Any state-dependent scalar. Different choices of `b` trade off
  variance vs. complexity (§7.1 Claim 2).
- `Â(s, a) = r(s, a) - b(s)` = advantage. What actually multiplies `∇log π` in the
  gradient estimate.

Keep those three separate in your head. The whole §10.3 discussion is about which
specific `b(s)` GRPO picks.

#### GRPO's specific choice of `b(s_i)`

For each prompt `s_i`, sample **`K` completions** `a_1, …, a_K ~ π_θ(·|s_i)` in parallel.
Compute each completion's reward `r(s_i, a_k)`. The baseline is the mean over the group:

$$
b(s_i) = \frac{1}{K} \sum_{k=1}^{K} r(s_i, a_k)
$$

Crucial property: `b(s_i)` depends only on the **prompt** `s_i`. Every completion
`a_1, …, a_K` in the same group **shares the same baseline** — a single scalar per
prompt, broadcast to all `K` completions.

**Per-completion advantage**:

$$
\hat A_k = r(s_i, a_k) - b(s_i)
$$

Positive if `a_k` scored above the group mean; negative if below; zero if all K
completions got the same reward.

#### Plugging `b(s_i)` into the PG estimator

Take §10.2's importance-weighted estimator and substitute `r → Â`:

$$
\nabla_\theta J(\theta) \approx \frac{1}{NK} \sum_{i=1}^{N}\sum_{k=1}^{K} \frac{\pi_\theta(a_k|s_i)}{\bar\pi(a_k|s_i)} \cdot \nabla_\theta \log \pi_\theta(a_k|s_i) \cdot \underbrace{\bigl[r(s_i, a_k) - b(s_i)\bigr]}_{\hat A_k}
$$

Each `(s_i, a_k)` pair contributes one term. The reward `r(s_i, a_k)` is this
completion's absolute reward; `b(s_i)` is the prompt-shared mean; their difference is the
advantage.

**In code** ([train/trainer.py:917-920](train/trainer.py#L917)):

```python
# rewards: shape (N*K,) — each completion's scalar r(s_i, a_k)
# mean_grouped_rewards: shape (N*K,) — per-prompt mean b(s_i), broadcast across group
advantages = rewards - mean_grouped_rewards
# advantages[i*K + k] = r(s_i, a_k) - b(s_i) = Â_k
```

**One line of code. That's the entire GRPO baseline.**

#### What `b(s_i)` REPLACES from the pre-2024 LLM RL pipeline

So far we've said: `b(s)` is *some* state-dependent baseline. GRPO picks the group mean.
But LLM RL had a `b(s)` before GRPO — it just used a *different* one. The question is:
**what specific form of `b(s)` did the field use before GRPO, and why did GRPO's choice
win?**

Pre-GRPO answer: **a learned value function `V_φ(s_t)`**. It served the *same role*
(state-dependent baseline subtracted from reward) — just a different way of computing it.

##### What `V_φ` was

- A **separate neural network** `V_φ`, conventionally the same architecture and size as
  the policy (often initialized from the SFT model).
- Input: state `s_t` (prompt + tokens generated so far).
- Output: scalar — an estimate of expected future reward from `s_t`.
- Trained alongside the policy by MSE against observed returns:

$$
\mathcal{L}_V(\phi) = \mathbb{E}\!\left[\bigl(V_\phi(s_t) - \hat R_t\bigr)^2\right]
$$

where `R̂_t` is some target return (observed or bootstrapped via GAE).

- Advantage computed via **Generalized Advantage Estimation (GAE)** — a specific
  formula for combining observed rewards with `V_φ` predictions:

$$
\hat A_t = \sum_{k=0}^{T-t} (\gamma\lambda)^k\, \delta_{t+k}, \qquad \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

Per-token advantages, bootstrapped through the value network.

##### Cost of `V_φ`

- **Memory**: training and backpropping through a neural net the size of the policy
  → roughly 2× memory.
- **Compute**: each training step does forward + backward on `V_φ` *and* the policy.
- **Hyperparameters**: discount `γ`, GAE `λ`, value-loss coefficient, value-clip range.
- **Stability**: value-loss instability is a well-known PPO pain point; a poorly-trained
  `V_φ` poisons advantage estimates, which poisons the policy update.

##### Same role, different implementation

**Both `V_φ(s_t)` and `b(s_i)` play the identical role in the PG estimator**: a
state-dependent baseline subtracted from reward to form the advantage. They're
interchangeable in the math — the gradient estimator has the same shape either way.

| Role | Pre-GRPO (PPO-RLHF) | GRPO |
|---|---|---|
| Formula for `b(s)` | Learned `V_φ(s)`, MSE loss | Sample mean `(1/K) Σ r(s, a_k)` |
| Requires a separate neural network? | **Yes** (~size of policy) | **No** |
| Extra hyperparameters | `γ`, `λ`, value-loss coef, value-clip | Just `K` |
| Runtime cost per update step | +1 forward + backward on `V_φ` | `K×` more sampling per prompt |

**GRPO replaces `V_φ(s)` with the group mean, as a specific choice of `b(s)`. Nothing
else about the PG estimator changes.** The reward `r(s, a)` is computed the same way.
The importance weighting is the same. The clipping is the same. The KL regularization
is the same. Just the baseline changes.

##### Trade-off in one sentence

**GRPO trades an expensive value network for more rollouts per prompt.** You pay in
sampling cost (K× more base-LLM forward passes); you save on training cost (no value
network, no GAE).

##### Why this trade works for LLM RL specifically

Value-function machinery earns its cost when rewards are **dense per-step** and need to
be propagated through the trajectory. For sparse, outcome-only rewards (math
correctness, code-ran-or-not), per-token value estimates don't add much — the key
question is *"how good is this completion compared to peers for the same prompt?"*, and
the group mean captures that directly.

With dense per-step rewards, `V_φ` would still win. For the GRPO regime, the group-mean
baseline is simpler and empirically at least as effective.

#### One small nuance (skippable)

The strict unbiasedness proof from §7.1 required `b` to not depend on the action being
evaluated. GRPO's `b(s_i) = (1/K) Σ_{k'} r(s_i, a_{k'})` **includes** `r(s_i, a_k)`
itself in the mean — so `b` technically depends on `a_k`. This introduces an `O(1/K)`
bias.

At `K = 8`, the bias is small and everyone ignores it. A strictly-unbiased
**leave-one-out** version would be `b_k(s_i) = (1/(K-1)) Σ_{k' ≠ k} r(s_i, a_{k'})` —
some implementations use this; the original GRPO paper does not. In practice both work
similarly.

### 10.4 Reference-model KL regularization

Orthogonal to the baseline choice: when the reward is a **learned model** `r_ψ` (as in
RLHF), the policy can learn to exploit quirks in `r_ψ` — generating nonsense that the
reward model happens to rate highly ("reward hacking"). Standard fix: penalize
divergence from a trusted reference policy `π_ref` (usually the SFT model we started
from):

$$
\bar r(s, a) = r_\psi(s, a) - \beta\, D_{\mathrm{KL}}\bigl(\pi_\theta(a|s) \,\|\, \pi_{\text{ref}}(a|s)\bigr)
$$

Not relevant in verifiable-reward settings (math, code) because there's no reward model
to exploit — the reward is the ground-truth check. But it's a standard piece of RLHF
machinery.

### 10.5 Why this matters for our project

- **GRPO fits the "sparse, verifiable reward" regime exactly** — our reward is
  `R_correct + α · R_efficiency`, no learned reward model.
- **Saving the value network is what makes our setup tractable**: we're training a
  <1M-param policy on top of a frozen 8B LLaDA. Adding another 8B value network would
  blow up the compute budget. GRPO lets us skip it.
- **The trade-off lands in our favor**: we already pay heavily for sampling (each
  rollout = 64 LLaDA forward passes). Paying `K = 8` of those per prompt is manageable;
  training a value network on top would not be.

---

## 11. Where it lives in this repo

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
| `logps_timestep`       | `log π_θ(a_t | s_t)`                | §3, §10            |
| `old_logps_slice`      | `log π̄(a_t | s_t)` (behavior policy) | §8, §10          |
| `coeff_1`              | importance ratio `π_θ/π̄`            | §8, §10          |
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

## 12. Common confusions (for you specifically)

**1. "Policy gradient" ≠ "the gradient of the policy."** It's the gradient of **expected
reward** `J(θ)` with respect to policy parameters θ. Easy to mis-parse on first read.

**2. Why `log π` and not `π`?** Two reasons:
   - (a) The log-derivative trick (§3.1) is what turns `∇J` into an expectation — so you
     can estimate it from samples.
   - (b) `log π` in the pseudo-loss is exactly the cross-entropy loss you already know, so
     the implementation is "supervised learning weighted by advantage" — one line
     different (§4, §9).

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

**5. dLLM MDP vs. §10's contextual bandit.** Section 7 frames LLM RL as a *one-step* MDP.
That works for standard autoregressive LLMs because you can factor the whole completion's
probability as `Π_t π(token_t | context)` in a single forward pass. Our dLLM setup is
different: the policy acts **multiple times** (one action-vector per denoising step), and
the final answer is the joint product of those action sequences. So we're closer to a
proper multi-step MDP with a sparse terminal reward, even though each individual step's
action is per-position. Keep this in mind — §10's framing is a useful baseline but not
exactly our setup.

---

## 13. Check-yourself questions

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

## 14. What's next

Once Q1–Q4 are solid, CLAUDE.md concept 3 is *"Variance reduction: baselines,
advantages"* — but you've already seen the core of that here (§7). The next genuinely
new content is:

- **GAE** (generalized advantage estimation) — interpolates between high-variance MC and
  high-bias bootstrap. Not used in this repo (GRPO sidesteps it), but useful context.
- **Group-relative advantage** in GRPO — already covered in §10; next step is to stare at
  the actual GRPO loss in [train/trainer.py:160](train/trainer.py#L160) and match every
  term to an equation.
- **PPO clipping** — the `torch.min(coeff_1·adv, coeff_2·adv)` trick with `coeff_2`
  clamped to `[1−ε, 1+ε]`. Covered in §8.3 above; full treatment in CS 285
  Lec 9.

Then we resume the shape-tracing exercise from the starter sequence (step 2 of
CLAUDE.md's starter sequence).
