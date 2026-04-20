"""Aggregate remask-action statistics across many trajectories.

Runs the 3-way policy on a sample of prompts with `record_trajectory=True`, then
computes:
  * remask frequency heatmap: (denoising step bucket) x (position), summed.
  * mean confidence of tokens at the step they were remasked.
  * mean confidence of those same tokens after they were re-predicted
    (i.e. the final step they were committed). Delta = after - before.
  * distribution of "lifetime" — how many steps a token spent masked after a
    remask, before being re-unmasked.

Outputs:
  * {out_dir}/heatmap.png
  * {out_dir}/conf_delta.json
  * {out_dir}/conf_delta.png (histogram)

Usage:
    python -m scripts.analysis.trajectory_stats \
      --policy_config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_3way_alpha0.yaml \
      --policy_ckpt outputs/3way_alpha0/checkpoint-800/model.safetensors \
      --dataset gsm8k --n_prompts 50 \
      --out eval_results/analysis/trajectory_stats_gsm8k
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from trl import TrlParser

from common.config import Config
from common.generation.generation import generate_unified
from common.models.policy import (
    DiTConfidencePolicy,
    DiTHiddenStatePolicy,
    PolicyHFWrapper,
)
from data.loaders.gsm8k import GSM8KDataset
from data.loaders.math500 import MATH500Dataset


DATASETS = {"gsm8k": GSM8KDataset, "math": MATH500Dataset}
MASK_ID_LLADA = 126336


def load_policy(config, model, device):
    if config.policy_type == "dit_hidden":
        core = DiTHiddenStatePolicy(
            dllm=model,
            time_embed_dim=config.policy_time_embed_dim,
            num_blocks=config.policy_num_blocks,
            smart_init=config.policy_smart_init,
            time_period=config.policy_time_period,
            num_actions=config.num_policy_actions,
        ).to(device)
    else:
        core = DiTConfidencePolicy(
            hidden_dim=config.policy_hidden_dim or 128,
            feedforward_dim=config.policy_feedforward_dim or 4 * (config.policy_hidden_dim or 128),
            num_heads=config.policy_num_heads,
            dropout=config.policy_dropout,
            time_embed_dim=config.policy_time_embed_dim,
            smart_init=config.policy_smart_init,
            confidences_top_p=config.confidences_top_p,
            num_blocks=config.policy_num_blocks,
            time_period=config.policy_time_period,
            num_actions=config.num_policy_actions,
        ).to(device)
    return PolicyHFWrapper(core, config.policy_type).to(device)


def analyze_trajectory(trajectory, gen_length):
    """Return per-trajectory stats:
      - remask_grid: (T, L) int count of remasks at each (step, position)
      - unmask_grid: (T, L) int count of unmasks
      - conf_delta records: list of (conf_before_remask, conf_after_recommit, lifetime_steps)
    """
    T = len(trajectory)
    L = gen_length
    remask_grid = np.zeros((T, L), dtype=np.int32)
    unmask_grid = np.zeros((T, L), dtype=np.int32)
    # Track for each position: most recent "remasked at step t with conf c"
    pending_remasks = {}  # pos -> (step_idx, conf_before)
    delta_records = []

    for t, step in enumerate(trajectory):
        unmask = step["unmask"][0].bool().tolist()
        remask = step.get("remask")
        remask = remask[0].bool().tolist() if remask is not None else [False] * L
        conf = step["confidence"][0].tolist()

        for pos in range(L):
            if remask[pos]:
                remask_grid[t, pos] += 1
                pending_remasks[pos] = (t, conf[pos])
            if unmask[pos]:
                unmask_grid[t, pos] += 1
                if pos in pending_remasks:
                    t0, c_before = pending_remasks.pop(pos)
                    delta_records.append({
                        "position": pos,
                        "conf_before": c_before,
                        "conf_after": conf[pos],
                        "delta": conf[pos] - c_before,
                        "lifetime_steps": t - t0,
                    })
    return remask_grid, unmask_grid, delta_records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy_config", required=True)
    p.add_argument("--policy_ckpt", required=True)
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--n_prompts", type=int, default=50)
    p.add_argument("--out", required=True)
    p.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    parser = TrlParser((Config,))
    (config,) = parser.parse_args_and_config(
        args=["--config", args.policy_config], fail_with_unknown_args=False
    )
    assert config.sampling_mode == "three_way"
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    policy = load_policy(config, model, device)
    state = load_file(args.policy_ckpt)
    policy.load_state_dict(state, strict=False)
    policy.eval()

    ds = DATASETS[args.dataset](tokenizer=tokenizer, subsample=-1, num_examples=0, add_reasoning=True)
    if args.n_prompts < len(ds):
        ds = torch.utils.data.Subset(ds, range(args.n_prompts))
    loader = DataLoader(
        ds, batch_size=1,
        collate_fn=ds.dataset.collate_fn if hasattr(ds, "dataset") else ds.collate_fn,
    )

    L = config.max_completion_length
    # Use max-length traj grid (pad individual trajs up)
    T_max = 2 * config.block_length * (L // config.block_length)
    remask_accum = np.zeros((T_max, L), dtype=np.int64)
    unmask_accum = np.zeros((T_max, L), dtype=np.int64)
    all_deltas = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            pid = batch["input_ids"].to(device)
            result = generate_unified(
                model=model,
                prompt=pid,
                remasking="policy",
                policy=policy,
                gen_length=L,
                block_length=config.block_length,
                temperature=0.0,
                mask_id=MASK_ID_LLADA,
                sampling_mode="three_way",
                full_context=config.policy_full_context,
                confidences_top_p=config.confidences_top_p,
                model_type="LLaDA",
                temperature_policy=1.0,
                record_trajectory=True,
            )
            rm, um, deltas = analyze_trajectory(result.trajectory, L)
            T = rm.shape[0]
            remask_accum[:T] += rm
            unmask_accum[:T] += um
            all_deltas.extend(deltas)
            print(f"[{i+1}/{len(loader)}] T={T} remasks={rm.sum()} unmasks={um.sum()}")

    # Save raw arrays + delta records
    np.save(Path(args.out) / "remask_grid.npy", remask_accum)
    np.save(Path(args.out) / "unmask_grid.npy", unmask_accum)
    with open(Path(args.out) / "conf_delta.json", "w") as f:
        json.dump({
            "deltas": all_deltas,
            "summary": {
                "n_remask_events": len(all_deltas),
                "mean_delta": float(np.mean([d["delta"] for d in all_deltas])) if all_deltas else 0.0,
                "median_delta": float(np.median([d["delta"] for d in all_deltas])) if all_deltas else 0.0,
                "mean_lifetime_steps": float(np.mean([d["lifetime_steps"] for d in all_deltas])) if all_deltas else 0.0,
                "pct_delta_positive": float(
                    np.mean([d["delta"] > 0 for d in all_deltas])
                ) if all_deltas else 0.0,
            },
        }, f, indent=2)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(
            remask_accum, aspect="auto", cmap="Reds", interpolation="nearest"
        )
        ax.set_xlabel("Position in sequence")
        ax.set_ylabel("Denoising step (0 = earliest)")
        ax.set_title(f"Remask count heatmap ({args.dataset.upper()}, {args.n_prompts} prompts)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(Path(args.out) / "heatmap.png", dpi=140)
        plt.close(fig)

        if all_deltas:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist([d["delta"] for d in all_deltas], bins=40, color="#28d", alpha=0.8)
            ax.axvline(0, color="k", linestyle="--", linewidth=0.6)
            ax.set_xlabel("conf_after_recommit - conf_before_remask")
            ax.set_ylabel("# remask events")
            ax.set_title(
                f"Confidence delta after remask ({args.dataset.upper()}, "
                f"{len(all_deltas)} events)"
            )
            fig.tight_layout()
            fig.savefig(Path(args.out) / "conf_delta.png", dpi=140)
            plt.close(fig)
        print(f"Wrote plots to {args.out}")
    except ImportError:
        print("matplotlib not available; skipping plots")


if __name__ == "__main__":
    main()
