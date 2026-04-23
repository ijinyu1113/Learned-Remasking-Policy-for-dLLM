# Study Roadmap — Learned Remasking Policy for dLLM

Full plan from "just finished concept 2 (policy gradient)" through "conceptually fluent,
ready to ship the project." Covers all 9 concepts in [CLAUDE.md](../CLAUDE.md)'s concept
map plus exercises.

Parallel to the CLAUDE.md concept map, but organized by time/phase rather than topic.

---

## Current state (as of 2026-04-22)

- ✅ **Concept 1** — MDPs (touched informally via starter-sequence step 1)
- ✅ **Concept 2** — Policy gradients (deep dive done; see [02_policy_gradient.md](02_policy_gradient.md))
- 🟡 **Concept 3** — Variance reduction (largely covered in §7 of notes 02; GAE remains)
- 🟡 **Concept 4** — RLHF vocabulary (largely covered in §10 of notes 02; Bradley-Terry remains)
- 🟡 **Concept 5** — GRPO specifically (covered §10.3; exercise Ex 01 in progress)
- ⬜ **Concept 6** — Masked diffusion LMs (new territory)
- ⬜ **Concept 7** — Unmasking policies (Jazbec 2025)
- ⬜ **Concept 8** — 3-way extension (own project)
- ⬜ **Concept 9** — Why this MDP is weird (integration)

---

## Phase 1 — Finish concept 2 via exercises (current phase)

Take the §3-§10 theory from notes 02 and ground it in the repo's actual code.

| # | Exercise | Source | Tests |
|---|---|---|---|
| Ex 01 | GRPO advantage computation | `train/trainer.py:912-920` | §10.3, reshape/broadcast, per-prompt baseline |
| Ex 02 | 3-way constrained action sampling | `common/generation/generation.py:493-605` | §1.3-1.4, §10, action space + constraints |
| Ex 03 | `_get_per_timestep_logps_block` | `train/trainer.py` | Shape manipulation + log-prob indexing + masking |

**Time budget**: ~2-3 hours total.

**Deliverable**: can close a random line of `compute_loss` with a mental `print()` of
the tensor's shape+meaning, and reproduce the GRPO formula on paper without looking.

---

## Phase 2 — Consolidate concepts 3-5 (RL side, mostly review)

### Concept 3: Variance reduction (baselines, advantages, GAE)

- **Already covered**: §7 of notes 02.
- **Gap**: formal GAE derivation — `Â_t = Σ_k (γλ)^k δ_{t+k}`. Not used in this repo
  (GRPO skips value functions), but it's in every PPO implementation.
- **Study**: CS 285 Lec 6 slides, ~30 min.
- **Ex 04**: derive GAE from reward-to-go by introducing a value-function-corrected
  advantage. Step-by-step derivation in a new sub-section of notes 02 or standalone.
- **Time**: ~1-1.5h.

### Concept 4: RLHF vocabulary

- **Already covered**: §10.1-§10.4 of notes 02.
- **Gap**: Bradley-Terry reward model. Preference data → reward model.
- **Study**: CS 288 Section 7 §2.1 (already have PDF) + skim InstructGPT paper intro.
- **Ex 05**: implement Bradley-Terry RM loss `-log σ(r_ψ(τ_i) - r_ψ(τ_j))`. Standalone
  PyTorch function + unit tests. Not used in our project but anchors the RLHF stack.
- **Time**: ~1h.

### Concept 5: GRPO specifically

- **Already covered**: §10.3 of notes 02 + Ex 01.
- **Gap**: read the actual DeepSeek-Math paper (arxiv.org/abs/2402.03300). See how they
  frame the contribution and what ablations they run.
- **Ex 06**: write out the full GRPO loss in one line, match every term to code in
  `train/trainer.py`. No AI, no peeking.
- **Time**: ~1h.

**Phase 2 total**: ~3-4 hours.

---

## Phase 3 — Diffusion LMs (concepts 6-7, new territory)

This is where the real study time goes. Admit weakness and budget time.

### Concept 6: Masked diffusion language models

- **Goal**: understand how LLaDA generates text — masking schedule, denoising loop,
  training objective. Contrast with autoregressive LMs.
- **Study order**:
  1. **LLaDA paper** (arxiv.org/abs/2502.09992) — intro + method. ~1.5h.
  2. **LLaDA inference loop in repo** — `common/generation/generation.py`, trace the
     denoising schedule. ~1h.
  3. **MDLM paper** (optional) — theoretical foundation. ~1h if you want depth.
- **Note to write**: `notes/03_masked_diffusion_lms.md` (same format as 02).
- **Ex 07**: shape-trace the LLaDA inference loop. What's the mask ratio at step t?
  What does the model output look like?
- **Ex 08** (Format 3 — bug-finding): I'll give you a broken version of
  `generate_with_dllm`. Find the bug. Candidates: wrong mask semantics, wrong schedule,
  forgetting to re-mask at end of step.
- **Time**: ~4-6h.

### Concept 7: Unmasking policies (Jazbec 2025 framework)

- **Goal**: understand the 2-way external policy — inputs, outputs, training.
- **Study**:
  1. **Jazbec paper** (arxiv.org/abs/2512.09106) — method section. ~1h.
  2. **Repo's 2-way mode** — trace exactly what differs from the 3-way path. ~1h.
- **Note to write**: `notes/04_unmasking_policies.md`.
- **Ex 09**: derive — on paper — why the 2-way policy gradient is a special case of §3's
  REINFORCE. Tests integration of §3 + Jazbec framing.
- **Time**: ~2-3h.

**Phase 3 total**: ~6-9 hours.

---

## Phase 4 — This project specifically (concepts 8-9)

You're already the expert here. This phase is about **articulating** what you know.

### Concept 8: The 3-way extension

- **Goal**: explain, cold: (a) why 3-way is strictly more expressive than 2-way,
  (b) what can go wrong (policy thrashing, early remasking destabilization),
  (c) why smart-init `[target, 0, -10]` matters.
- **Study**: re-read `cs288_remasking_policy_plan.md` §3-§4 with everything you now know.
- **Ex 10**: derive — on paper — the 3-way policy gradient. Identify where
  `num_actions=3` propagates through the code.
- **Ex 11** (Format 2, derive-then-implement): argue that GRPO's group-mean baseline is
  independent of `num_actions`, so the only change for 3-way is in the policy head.
  Verify by tracing the code.
- **Time**: ~2-3h.

### Concept 9: Why this MDP is weird

- **Goal**: hold the full mental model simultaneously — state, action, transition, reward,
  credit assignment, sample efficiency, parallelism.
- **Study**: synthesize §1 of notes 02 + concepts 6-8.
- **Ex 12** (writing exercise): produce a 1-page prose explanation of the project's MDP
  that a classmate (took CS 285, hasn't read your code) would understand. Rubric:
  state/action/reward/transition each in 2-3 sentences, credit assignment discussion,
  why GRPO fits. Becomes your final paper's MDP section.
- **Time**: ~2h.

**Phase 4 total**: ~4-5 hours.

---

## Phase 5 — Run, debug, interpret (ongoing)

- Run GRPO training on GSM8K (single GPU, short run).
- Monitor: loss curves, advantage distribution, remask frequency, entropy.
- **Ex 13** (Format 5, minimal repro): 30-line standalone CartPole REINFORCE. No
  framework. Confirm you can do PG from scratch. Then contrast with our GRPO loop.
- **Ex 14**: Pareto frontier analysis — accuracy vs NFE trade-off across α values.
  Real research task, feeds into final paper.
- **Time**: weeks of training + analysis (per CS 288 timeline in `cs288_remasking_policy_plan.md`).

---

## Cumulative time estimates

| Phase | Hours |
|---|---|
| 1 | 2-3 |
| 2 | 3-4 |
| 3 | 6-9 |
| 4 | 4-5 |
| **Phase 1-4 total** | **15-21** |
| 5 | weeks |

At ~4h/session: **4-6 focused sessions** to "conceptually fluent." After that, research,
not studying.

---

## Parallel tracks (don't block sequentially)

- **Coding-without-AI drills**: Ex 13 onward. All Phase 1-4 exercises are no-AI.
- **Paper reading**: LLaDA + Jazbec + GRPO (+ RemeDi) — read anytime, they'll click
  progressively.
- **Experiments**: Phase 5 can overlap with Phase 3-4 if infra is up. Long training
  runs happen in background.

---

## Adjustments & notes to self

If a phase feels tedious or low-value, prune exercises. This roadmap is a scaffold,
not a prison. Some exercises (Ex 04 GAE derivation, Ex 05 Bradley-Terry) exist mostly
for completeness — skip if you'd rather spend the time on Phase 3.

Re-open this file at the start of each session to check off progress and re-calibrate.
