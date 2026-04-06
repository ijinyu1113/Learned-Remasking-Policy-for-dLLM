# Learned Remasking Policy for Diffusion LLMs

CS 288 final project. We extend the learned 2-way unmasking policy of Jazbec et al. ([arXiv:2512.09106](https://arxiv.org/abs/2512.09106)) to a **3-way action space** — *unmask / keep / remask* — so a tiny external policy can also correct earlier mistakes made by a frozen diffusion LLM, without modifying the base model.

See [cs288_remasking_policy_plan.md](cs288_remasking_policy_plan.md) for the full project plan, MDP formulation, baselines, and timeline.

## What's in this repo

This repo is a fork of [apple/ml-rl-dllm](https://github.com/apple/ml-rl-dllm) (the official Jazbec et al. codebase, which is also our direct 2-way baseline). We use it as the foundation and add the 3-way action head on top.

| Path | Source | What it is |
|---|---|---|
| [common/](common/), [train/](train/), [eval/](eval/), [data/](data/), [configs/](configs/) | upstream | LLaDA-8B inference, GRPO training loop, eval pipeline, baseline samplers |
| [INFRA.md](INFRA.md) | ours | Setup, hardware notes, baseline commands, where Person B/C plug in |
| [cs288_remasking_policy_plan.md](cs288_remasking_policy_plan.md) | ours | Project plan: MDP, method, experiments, timeline, team division |
| `README.md` (this file) | ours | Project overview |

## Team

| Person | Role |
|---|---|
| A | Infrastructure: LLaDA inference, baselines, eval scripts ([INFRA.md](INFRA.md)) |
| B | 3-way policy network, GRPO loop changes |
| C | Experiments, ablations, analysis |
| Cloud GPU lead (TBD) | Stand up GCP A100 with class credits — see [INFRA.md §"Cloud GPU"](INFRA.md#cloud-gpu-someones-job) |

## Quick start

LLaDA-8B-Instruct doesn't fit on consumer GPUs and `bitsandbytes` doesn't run on Apple Silicon — for real work everyone uses the team's GCP A100. See [INFRA.md](INFRA.md) for full setup, smoke test, and per-baseline run commands.

```bash
pip install -e .
pip install s3fs bitsandbytes
huggingface-cli login   # if needed
```

Smoke test (top-K confidence baseline, 4 GSM8K problems):

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

## Method (one paragraph)

A frozen LLaDA-8B-Instruct generates by iteratively unmasking tokens. At each denoising step, a tiny single-block DiT-style policy (~1M params) reads the current per-position confidences and outputs, for every position, a 3-way action: *unmask*, *keep*, or *remask*. We train it with GRPO on outcome rewards (GSM8K / MATH / HumanEval / MBPP correctness) plus an efficiency term. The base model is never updated; we expect to recover RemeDi-style error correction at <0.01% of RemeDi's training cost.

## Status

Phase 1 (infra). Upstream foundation merged, cross-platform eval patch landed, smoke test runnable on the team A100 once it's provisioned. CP1 (2-way reproduction on GSM8K) is the next milestone.

## Acknowledgements

Built on [apple/ml-rl-dllm](https://github.com/apple/ml-rl-dllm) (Apple Inc., 2026), released under the terms in [LICENSE](LICENSE) and [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS). If you use this code, please also cite the original paper:

```bibtex
@misc{jazbec2025learningunmaskingpoliciesdiffusion,
  title={Learning Unmasking Policies for Diffusion Language Models},
  author={Metod Jazbec and Theo X. Olausson and Louis Béthune and Pierre Ablin and Michael Kirchhof and Jo\~ao Monteiro and Victor Turrisi and Jason Ramapuram and Marco Cuturi},
  year={2025},
  eprint={2512.09106},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2512.09106},
}
```
