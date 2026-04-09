# Handoff — Person A baseline run

Short doc to pick up where I left off. For background read [README.md](README.md), [INFRA.md](INFRA.md), [cs288_remasking_policy_plan.md](cs288_remasking_policy_plan.md).

## What's running right now

A full heuristic-baseline sweep is running on the GCP L4 VM, started ~5 AM 2026-04-09.

- **VM**: `dllm-l4` in zone `us-central1-a` (GCP project `cs288-remasking`)
- **Tmux session**: `baselines`
- **Script**: `~/Learned-Remasking-Policy-for-dLLM/run_all_baselines.sh`
- **Sweep**: 3 baselines × 4 datasets × 1 seed
  - `baseline-random-K32`
  - `baseline-low_confidence-K32`
  - `baseline-fastdllm-t0.7`
  - Datasets: gsm8k, math, humaneval, mbpp
  - Seed: 42 only (cheap pass for CP2; more seeds later)
- **Expected wall time**: ~12–15 hr from start → finish between ~5 PM and 8 PM 2026-04-09
- **Expected cost**: ~$11–14 of GCP credits at L4 = $0.88/hr
- **GRPO training is NOT running** — that's deferred to Spot A100 later

## Reconnecting from a new device

```bash
# 1. SSH (need gcloud installed + `gcloud auth login` once on the new device)
gcloud compute ssh dllm-l4 --zone us-central1-a

# 2. If VM was stopped, start it first:
gcloud compute instances start dllm-l4 --zone us-central1-a

# 3. Re-enter the tmux session
tmux attach -t baselines
# detach again with: Ctrl-B then D
```

## Checking progress without attaching tmux

```bash
# Is the GPU busy?  Should show ~16 GB used, ~99% util, python3.12 process.
nvidia-smi

# Is the tmux session still alive?
tmux ls

# Live tail the current baseline's log
ls -lt ~/Learned-Remasking-Policy-for-dLLM/logs_*.txt   # newest = currently running
tail -f ~/Learned-Remasking-Policy-for-dLLM/logs_baseline-random-K32.txt
# Ctrl-C to stop tailing — does NOT kill the script
```

## Is it done?

Look for the literal string `ALL BASELINES DONE` in the most recent log, or in the tmux session output. If you see it, the script finished cleanly.

If it died partway:
- Check the log for the error
- Each baseline writes to its own `eval_results/<NAME>/` directory, so completed baselines are not lost
- You can re-run only the failed one by editing the `for NAME in ...` loop in `run_all_baselines.sh`

## STOP THE VM as soon as the script finishes

This is the single most important post-run step. Idle L4 = $0.88/hr wasted.

```bash
# from inside tmux: Ctrl-B then D to detach first
exit                                                       # leave SSH
gcloud compute instances stop dllm-l4 --zone us-central1-a
```

Or browser console → Compute Engine → VM instances → check `dllm-l4` → STOP.

Stopped ≠ deleted. Files persist. You can start it again any time.

## Where the results live

```
~/Learned-Remasking-Policy-for-dLLM/eval_results/
├── baseline-random-K32/
│   ├── baseline-random-K32/
│   │   └── checkpoint-baseline-random-K32_seed_42_temp_policy_0.0/
│   │       └── <dataset>_..._generations.json
│   ├── detailed_results.csv          ← per-run metrics
│   ├── summary_statistics.csv        ← aggregated by config
│   └── results_report.txt            ← human-readable summary table
├── baseline-low_confidence-K32/
│   └── (same structure)
└── baseline-fastdllm-t0.7/
    └── (same structure)
```

The aggregator runs automatically at the end of each baseline. If for some reason it didn't, run it manually:

```bash
python -m eval.aggregate_results --results_dir eval_results/baseline-random-K32
```

## Interpreting the results for CP2

What you want from each baseline, per dataset:

| Metric | Where to find it | What it means |
|---|---|---|
| **Accuracy** | `summary_statistics.csv` column `accuracy` | Exact-match (gsm8k/math) or pass@1 (humaneval/mbpp) |
| **NFEs** | `summary_statistics.csv` column `avg_nfes` | Number of LLaDA forward passes per problem. ~32 for K32 baselines. Critical x-axis for the Pareto frontier. |
| **Wall time** | `summary_statistics.csv` column `avg_wall_time` | Sec per problem. Sanity check; not for the paper. |

For the **CP2 baseline table**, fill in this layout per dataset:

| Method | NFE | Accuracy |
|---|---|---|
| Random (K=32) | 32 | _from results_report.txt_ |
| Top-K confidence (K=32) | 32 | _from results_report.txt_ |
| Fast-dLLM (t=0.7) | _from results_report.txt_ | _from results_report.txt_ |
| Learned 2-way (Jazbec et al.) | 32 | **TBD — training in progress** |

### Sanity-check the numbers

Compare against the **published numbers** from Jazbec et al. (arXiv:2512.09106). On GSM8K with LLaDA-8B-Instruct, K=32, expect roughly:

- Random: ~30–40%
- Top-K confidence (low_confidence): ~60–65%
- Fast-dLLM (t=0.7): ~55–60% (depends heavily on threshold)

If your numbers are in those ballparks → trust them, write them up. If they're wildly off (e.g., GSM8K top-K at 5% or 95%), something's wrong with the eval setup — flag it before writing the report.

## Pulling results to your local machine

```bash
# from your laptop, NOT the VM
gcloud compute scp --recurse \
  dllm-l4:~/Learned-Remasking-Policy-for-dLLM/eval_results \
  ./eval_results \
  --zone us-central1-a
```

Then commit them to the repo:

```bash
git add eval_results/
git commit -m "Person A: heuristic baseline results (seed 42, all 4 datasets)"
git push
```

## What's next after baselines land

1. **Write CP2 sections** using the table above + content from [cs288_remasking_policy_plan.md](cs288_remasking_policy_plan.md):
   - Intro (lift from §2 + §9)
   - Related Work (lift from §11, expand each bullet)
   - Baseline Setup (this run + INFRA.md content)
   - VESSL credits paragraph (template in earlier conversation)
2. **Person B** starts on the 3-way action head — see [INFRA.md §"Where Person B plugs in the 3-way action"](INFRA.md).
3. **Person C** starts evaluation harness for the 3-way policy once it's training.
4. Eventually: spin up the **Spot A100** (separate VM, see [INFRA.md](INFRA.md)) for GRPO training. L4 stays for eval.

## Common pitfalls

- **`exit` inside tmux kills the script.** Always use Ctrl-B D to leave tmux.
- **Don't forget to stop the VM.** $0.88/hr × 1 night of forgetting = ~$8 wasted. Set a $50 budget alert in GCP Billing as a safety net.
- **MBPP and HumanEval may fail** if the VM doesn't have a working `python` for code execution. If logs show `mbpp` or `humaneval` errors, paste the traceback into a new Claude session — there's a known fix that involves installing `python3` as a system binary or adjusting the eval harness.
- **The aggregator sometimes silently misses files** if the directory layout doesn't match its glob. If `summary_statistics.csv` is empty or missing, run `python -m eval.aggregate_results --results_dir eval_results/<NAME>` manually.
