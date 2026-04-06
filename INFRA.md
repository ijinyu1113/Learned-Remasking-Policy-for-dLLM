# Infrastructure Notes

This repo is a fork of [apple/ml-rl-dllm](https://github.com/apple/ml-rl-dllm) (Jazbec et al., "Learning Unmasking Policies for Diffusion Language Models", arXiv:2512.09106). It's the foundation for our 3-way (unmask / keep / remask) extension.

- **Person A** — infra (this doc), baselines, eval scripts
- **Person B** — 3-way action head, GRPO loop changes
- **Person C** — experiments, ablations, analysis
- **Cloud GPU lead (TBD)** — stand up an A100 on GCP using class credits (see §"Cloud GPU" below). Without this, nobody can run real experiments.

## Hardware reality

LLaDA-8B-Instruct is **8B params**. Practical needs:

| Mode | VRAM | Where it runs |
|---|---|---|
| Training (GRPO, 2-way or 3-way) | ≥40 GB | A100 40GB / A100 80GB / H100 |
| Eval, bf16 | ~20 GB | A100, A10G, RTX 4090/5090 (24GB) |
| Eval, 4-bit (`load_in_4bit`) | ~6 GB | Any modern CUDA GPU; **Linux only** (`bitsandbytes` is broken on macOS and unreliable on Windows) |

**Apple Silicon Macs**: there's no realistic local path. MPS is supported by PyTorch but `bitsandbytes` doesn't run on it, and a 16GB unified-memory Mac can't hold the model in fp16. M-series users should develop policy/training code locally (no LLaDA load) and run actual experiments on the GCP A100.

**Windows**: works with caveats. Use `attn_implementation: sdpa` (not `flash_attention_2`), use `bitsandbytes` for 4-bit, and apply the cross-platform patch already in [eval/eval.py](eval/eval.py) (the `dist.is_initialized()` fallback for `CustomDistributedSampler`). `accelerate launch --num_processes 1` does not init a process group on Windows; the patch handles that.

## Cloud GPU (someone's job)

We have free Google Cloud credits via the class. We need one teammate to:

1. Spin up a GCP VM with **1× A100 40GB** (region with capacity — `us-central1-a`, `us-central1-c`, or `europe-west4-a` typically have A100s; check the quota page first). Recommended: `a2-highgpu-1g` machine type, Deep Learning VM with CUDA 12 + PyTorch image.
2. Open SSH access for the team (or set up VS Code Remote-SSH instructions).
3. Clone this repo, run §"Setup" below, run §"Smoke test" below to confirm it works.
4. Document the SSH command + any cost-saving notes (always stop the VM when not in use — A100s cost ~$3/hr; an idle one will burn the credits).
5. Decide cadence: shared box, or each person spins their own from a snapshot.

This is a blocker for Persons B and C — please claim it early.

## Setup (on a Linux GPU box)

```bash
pip install -e .
pip install s3fs bitsandbytes
huggingface-cli login   # if downloads return 401/403; otherwise skip
```

## Smoke test (on the GPU box)

Runs the **top-K confidence baseline** on 4 GSM8K problems. No training, no checkpoint download.

```bash
mkdir -p ./outputs/baseline-low_confidence-K32/checkpoint-baseline-low_confidence-K32
touch     ./outputs/baseline-low_confidence-K32/checkpoint-baseline-low_confidence-K32/.baseline_marker

python -m eval.pipeline ./outputs/baseline-low_confidence-K32 \
  configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
  --checkpoints self \
  --datasets gsm8k --seeds 42 --temperatures 0.0 \
  --save_path ./eval_results/lowconf_smoke \
  --n_test 4
```

If that produces a `*_generations.json` under `./eval_results/lowconf_smoke/`, infra is working.

## Repo layout (inherited from upstream)

- [common/generation/generation.py](common/generation/generation.py) — LLaDA denoising loop. **Person B injects the 3-way action sampler here.** All four samplers branch on the `remasking` arg:
  - `"random"` — random unmask K positions per step
  - `"low_confidence"` — top-K confidence (standard heuristic)
  - `"fastdllm"` — confidence threshold (Fast-dLLM)
  - `"policy"` — learned 2-way policy from a checkpoint
- [common/generation/sampling.py](common/generation/sampling.py) — Bernoulli / Plackett-Luce / DPLS distribution primitives. Probably untouched.
- [train/train.py](train/train.py) — GRPO loop. Person B extends policy output dim 1 → 3 here.
- [eval/eval.py](eval/eval.py) — single-run eval. `parse_baseline_checkpoint` at line 70 implements the magic `baseline-<method>-K<steps>-t<thres>` checkpoint name for heuristic baselines (no trained weights needed).
- [eval/pipeline.py](eval/pipeline.py) — multi-seed/multi-checkpoint orchestrator.
- [configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml](configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml) — starting config.

## Run all baselines for the paper (on the A100)

Per [plan §4](cs288_remasking_policy_plan.md), we need four baselines × four datasets (GSM8K, MATH500, HumanEval, MBPP) × multiple seeds. Three are no-train heuristics; the fourth requires training the 2-way policy first. Single script:

```bash
CFG=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml
DATASETS=gsm8k,math,humaneval,mbpp
SEEDS=42,43,44

# --- Heuristic baselines (no training) ---
for NAME in baseline-random-K32 baseline-low_confidence-K32 baseline-fastdllm-t0.7; do
  mkdir -p ./outputs/$NAME/checkpoint-$NAME
  touch     ./outputs/$NAME/checkpoint-$NAME/.baseline_marker
  python -m eval.pipeline ./outputs/$NAME $CFG \
    --checkpoints self \
    --datasets $DATASETS \
    --seeds $SEEDS \
    --temperatures 0.0 \
    --save_path ./eval_results/$NAME
done

# --- Learned 2-way policy (Jazbec et al.) — train, then eval ---
python -m train.train --config $CFG
# checkpoints land in the output_dir set inside the YAML; assume ./outputs/policy2way

python -m eval.pipeline ./outputs/policy2way $CFG \
  --checkpoints last \
  --datasets $DATASETS \
  --seeds $SEEDS \
  --temperatures 1.0 \
  --sampling_mode bernoulli-argmax \
  --save_path ./eval_results/policy2way
```

Notes:

- **`K<N>` / `t<x>` in baseline names** — `K32` = 32 denoising steps; `t0.7` = Fast-dLLM threshold 0.7. The parser at [eval/eval.py:70](eval/eval.py#L70) accepts both. To plot the Pareto frontier (plan §5.1), sweep K for `random`/`low_confidence` (e.g. K=8,16,32,64) and t for `fastdllm` (e.g. t=0.5,0.7,0.9) — just add more entries to the loop.
- **Temperature** — heuristics ignore policy temperature; `--temperatures 0.0` is correct and faster (pipeline creates one output dir per temperature). The trained 2-way policy uses `1.0` per upstream's recommendation for BL=32.
- **Cost** — full GSM8K (1319) + MATH500 (500) + HumanEval (164) + MBPP (~500) × 32 steps × 8B forward × 3 seeds × 4 baselines dominates infra cost. Budget ~half a day per heuristic baseline on a single A100, ~1 day for the trained policy (train + eval). Don't leave the VM idle.
- **Iteration** — add `--n_test 50` while debugging, drop it for paper numbers.
- **Results** — `eval.pipeline` auto-runs `eval.aggregate_results` and dumps a CSV under `--save_path`. Person C reads from those CSVs; commit them to `results/baselines/`.

## Run a single baseline (on the A100)

All four use the same eval pipeline with magic checkpoint names parsed at [eval/eval.py:70](eval/eval.py#L70). The naming convention: `baseline-<method>[-K<steps>][-t<thres>]`. Pre-create the marker file for each.

```bash
CFG=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml

for NAME in baseline-random-K32 baseline-low_confidence-K32 baseline-fastdllm-t0.7; do
  mkdir -p ./outputs/$NAME/checkpoint-$NAME
  touch     ./outputs/$NAME/checkpoint-$NAME/.baseline_marker
  python -m eval.pipeline ./outputs/$NAME $CFG \
    --checkpoints self \
    --datasets gsm8k --seeds 42 --temperatures 0.0 \
    --save_path ./eval_results/$NAME
done
```

For the 2-way **learned** baseline, train first then eval the checkpoint:

```bash
python -m train.train --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml
# checkpoints land in the output_dir defined in the YAML — override if needed

python -m eval.pipeline ./outputs/my_run $CFG \
  --checkpoints last \
  --datasets gsm8k --seeds 42,43,44 --temperatures 1.0 \
  --sampling_mode bernoulli-argmax \
  --save_path ./eval_results/policy2way
```

Multi-GPU later:
```bash
accelerate launch --config_file configs/accelerate_configs/8gpu_ddp.yaml \
  -m train.train --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml
```

## Where Person B plugs in the 3-way action

1. **Action sampling** — add a `remasking == "policy3way"` branch in [common/generation/generation.py](common/generation/generation.py) alongside the existing `policy` branch. Current `policy` branch produces a per-position Bernoulli (unmask vs. keep); 3-way needs a per-position Categorical over {unmask, keep, remask}, with the constraint mask from plan §3.1 (action 1 only on masked positions, action 2 only on unmasked).
2. **Policy head** — in the DiT-confidence policy module (`policy_type: dit_confidence` in the YAML), change the output projection from 1 logit to 3.
3. **GRPO log-likelihood** — in [train/train.py](train/train.py), the existing path uses `bernoulli_batch_loglik`. The 3-way path needs `Categorical.log_prob` over the constrained action space.
4. **Monotonicity constraint** — enforced in the action sampler (limit `|remask| < |unmask|` per step, plan §3.3).

## Where Person C reads results

`eval.pipeline` writes per-seed JSON generations and an aggregated CSV under `--save_path`. See [eval/aggregate_results.py](eval/aggregate_results.py) for the format. Commit baseline numbers under `results/baselines/` so 3-way runs can be diffed against them.

## Cross-platform patches already applied

- [eval/eval.py:584](eval/eval.py#L584) — `CustomDistributedSampler` now falls back to `sampler=None` when `torch.distributed` isn't initialized. This unblocks single-process eval on Windows/Mac and on any A100 box you launch with plain `python -m eval.pipeline ...` (no `accelerate launch`).

## Verification checklist

- [ ] GCP A100 stood up; SSH instructions in team chat
- [ ] Smoke test (top-K, n_test=4) produces `*_generations.json`
- [ ] Full top-K baseline on GSM8K within ±1pt of Jazbec et al. Table 1
- [ ] Full Fast-dLLM baseline on GSM8K within ±1pt of Jazbec et al. Table 1
- [ ] 2-way learned policy reaches the published GSM8K number (CP1 sanity check)
- [ ] Aggregated CSVs under `results/baselines/` for the three heuristic baselines
