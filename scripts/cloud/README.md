# Preemptible-safe launch scripts

These are bash scripts to be run on a Linux / macOS A100 instance (they assume
a POSIX shell and `accelerate launch`). They were designed for **GCP preemptible
A100** nodes, which can be killed at any moment. Key design rules:

1. **Auto-resume.** Every training script loops on the trainer — if the process
   exits for any reason (preemption, OOM, etc.), the loop re-invokes it. The
   trainer picks up the latest checkpoint automatically via train.py's
   `resume_from_checkpoint=True` auto-detect.
2. **Frequent checkpoints.** The 3-way yamls use `save_steps: 50` so the most
   work you lose on a preempt is ≤ 50 steps (~50 min at 1 min/step).
3. **Separate scripts per concern.** One training run per script, one eval per
   script — preempts don't cascade.
4. **Idempotent outputs.** Eval writes to `eval_results/<run_name>/...` keyed
   by checkpoint + seed + temp + sampling_mode, so re-running is a no-op for
   completed configurations.

## Must-do (minimum viable paper)

| Script | What it does | Est. A100-hrs |
|---|---|---|
| `eval_2way_ckpt700_gsm8k.sh`  | Full 1319-problem GSM8K eval at checkpoint 700     | 2–3 |
| `train_3way_alpha0.sh`        | Train 3-way policy, α=0 (resume-safe loop)          | 250 |
| `eval_3way_gsm8k.sh`          | Full GSM8K eval of best 3-way checkpoint            | 2–3 |

## Nice-to-have

| Script | What it does | Est. A100-hrs |
|---|---|---|
| `eval_all_datasets.sh`        | GSM8K + MATH-500 + HumanEval + MBPP, both policies  | ~10 |
| `train_3way_alpha1.sh`        | Train 3-way policy with α=1 efficiency term         | 250 |

## Ordering

Run in this order to derisk:

```
eval_2way_ckpt700_gsm8k.sh             # locks in 2-way number
train_3way_alpha0.sh                   # long-running (critical path)
eval_3way_gsm8k.sh                     # main result
eval_all_datasets.sh                   # after GSM8K result looks good
train_3way_alpha1.sh                   # in parallel if budget allows
```

## Dependencies

All scripts assume:
- `pip install -e .` has been run in the repo root
- `HF_TOKEN` is set (either in env or a `.env` file; see `train/train.py`)
- `WANDB_API_KEY` is set (training scripts report to wandb)
- The repo root is the CWD

Pass these env vars to customize:
- `RUN_NAME` — name the output directory (defaults to a sane value per script)
- `TWO_WAY_CKPT_DIR` — absolute path to the 2-way checkpoint folder (for eval)
- `THREE_WAY_CKPT_DIR` — absolute path to the 3-way checkpoint folder
- `N_GPUS` — number of GPUs (defaults to all visible)
- `SEEDS` — eval seeds (defaults to `42`)
