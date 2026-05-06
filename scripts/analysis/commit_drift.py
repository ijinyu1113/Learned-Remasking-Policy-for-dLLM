"""Commit-drift diagnostic: how often does the base model disagree with its own
earlier commits once full context is available?

For a given policy and dataset, generate completions while recording each commit
event (position, token, denoising-step, p_at_commit_time). After full generation,
run one more forward pass on the COMPLETED sequence (no masks) and read the
model's probability for the committed token at each position. If the model's
agreement with the commit drops sharply (e.g., p_commit > 0.5, p_final < 0.2),
that's a "drift event" — exactly the kind of mistake that a remasking policy
could correct.

Aggregated reporting:
  - Mean drift events per problem
  - Distribution: 0 / 1 / 2+ / 5+ drift events
  - Among WRONG problems: fraction with >= 1 drift event (upper bound on remask
    upside — these are failures that remasking *could* have fixed)
  - Among CORRECT problems: fraction with >= 1 drift event (collateral damage —
    remasking on these would have been harmful if it fired)

This gives a clean upper bound on the maximum benefit a perfect remasking
policy could provide on this dataset, with this base model.

Usage:
    python -m scripts.analysis.commit_drift \
      --policy_config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
      --policy_ckpt outputs/2way_eval/checkpoint-1000/model.safetensors \
      --dataset gsm8k --n_test 100 \
      --out eval_results/analysis/commit_drift_2way_ckpt1000.json
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
from common.parsing.parse_and_get_acc import (
    extract_gsm_answer,
    extract_math_answer,
    check_gsm_correct,
    check_math_correct,
)
from data.loaders.gsm8k import GSM8KDataset
from data.loaders.math500 import MATH500Dataset


DATASETS = {"gsm8k": GSM8KDataset, "math": MATH500Dataset}
EXTRACTORS = {
    "gsm8k": (extract_gsm_answer, check_gsm_correct),
    "math": (extract_math_answer, check_math_correct),
}
MASK_ID_LLADA = 126336

# Drift thresholds (paper-tuneable)
COMMIT_HIGH_CONF = 0.5      # token committed with this much probability or more
FINAL_LOW_CONF = 0.2        # ...but at final-context only this much agreement


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
            feedforward_dim=config.policy_feedforward_dim
            or 4 * (config.policy_hidden_dim or 128),
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


def extract_commits_from_trajectory(trajectory, prompt_L, mask_id):
    """For each unmask event in the trajectory, record (position, token, step, p_commit).

    Uses top-1 confidence at commit time = p(committed_token | partial context),
    since the committed token is exactly the argmax that produced that confidence.
    """
    commits = []
    for step in trajectory:
        unmask = step["unmask"][0].bool()  # (L,)
        x0 = step["x0"][0]                 # (L,) predicted tokens
        conf = step["confidence"][0]       # (L,) top-1 prob at each position
        step_idx = step["step"]
        positions = unmask.nonzero(as_tuple=True)[0]
        for pos in positions.tolist():
            commits.append({
                "position": pos,
                "token": int(x0[pos].item()),
                "step": step_idx,
                "p_commit": float(conf[pos].item()),
            })
    return commits


@torch.no_grad()
def get_p_final_at_committed_tokens(model, sequence, prompt_L, model_type, attention_mask=None):
    """Run a forward pass on the fully-unmasked sequence and return p_final for each
    committed token at its position. Returns a (gen_length,) tensor where index i
    holds probs[i, sequence[prompt_L + i]] under the final full context.
    """
    out = model(sequence, attention_mask=attention_mask)
    if model_type == "Dream":
        logits = out.logits[:, prompt_L - 1 : -1]
    else:
        logits = out.logits[:, prompt_L:]
    probs = torch.softmax(logits.to(torch.float32), dim=-1)  # (1, L, V)
    gen_part = sequence[:, prompt_L:]                        # (1, L)
    p_final = probs.gather(-1, gen_part.unsqueeze(-1)).squeeze(-1)  # (1, L)
    return p_final[0].cpu()  # (L,)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy_config", required=True,
                   help="Path to the yaml config used by the policy")
    p.add_argument("--policy_ckpt", required=True,
                   help="Path to model.safetensors for the policy")
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--n_test", type=int, default=100)
    p.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--commit_high_conf", type=float, default=COMMIT_HIGH_CONF,
                   help="Threshold above which a commit is considered confident")
    p.add_argument("--final_low_conf", type=float, default=FINAL_LOW_CONF,
                   help="Threshold below which the final p(committed) is considered drift")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    parser = TrlParser((Config,))
    (config,) = parser.parse_args_and_config(
        args=["--config", args.policy_config], fail_with_unknown_args=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading base model {args.model_path} on {device}")
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    policy = load_policy(config, model, device)
    state = load_file(args.policy_ckpt)
    policy.load_state_dict(state, strict=False)
    policy.eval()

    # Dataset
    ds = DATASETS[args.dataset](
        tokenizer=tokenizer, subsample=-1, num_examples=0, add_reasoning=True
    )
    if args.n_test < len(ds):
        ds = torch.utils.data.Subset(ds, range(args.n_test))
    loader = DataLoader(
        ds, batch_size=1,
        collate_fn=ds.dataset.collate_fn if hasattr(ds, "dataset") else ds.collate_fn,
    )

    extract_fn, check_fn = EXTRACTORS[args.dataset]
    L = config.max_completion_length
    BL = config.block_length

    records = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            pid = batch["input_ids"].to(device)
            attn = batch["attention_mask"].bool().to(device) if "attention_mask" in batch else None
            gt = batch["answers"][0]
            prompt_L = pid.shape[1]

            # Generate with trajectory recording
            result = generate_unified(
                model=model,
                prompt=pid,
                remasking="policy",
                policy=policy,
                gen_length=L,
                block_length=BL,
                temperature=0.0,
                mask_id=MASK_ID_LLADA,
                sampling_mode=config.sampling_mode,
                full_context=config.policy_full_context,
                confidences_top_p=config.confidences_top_p,
                model_type="LLaDA",
                attention_mask=attn,
                temperature_policy=1.0,
                record_trajectory=True,
            )

            # Extract commits and get p_final
            commits = extract_commits_from_trajectory(
                result.trajectory, prompt_L=prompt_L, mask_id=MASK_ID_LLADA
            )
            # Reconstruct an attention mask aligned to full sequence shape
            seq_attn = torch.ones(
                (1, result.sequences.shape[1]), dtype=torch.float, device=device
            )
            if attn is not None:
                seq_attn[:, :prompt_L] = attn.float()
            seq_attn = seq_attn.to(model.dtype)
            p_final = get_p_final_at_committed_tokens(
                model, result.sequences, prompt_L=prompt_L,
                model_type="LLaDA", attention_mask=seq_attn,
            )  # (gen_length,)

            # Score the answer
            text = tokenizer.decode(
                result.sequences[0, -L:], skip_special_tokens=True
            )
            ans = extract_fn(text)
            is_correct = bool(check_fn(ans, gt))

            # Identify drift events
            drift_events = []
            for c in commits:
                pos = c["position"]
                pc = c["p_commit"]
                pf = float(p_final[pos].item())
                c["p_final"] = pf
                c["drift"] = bool(pc > args.commit_high_conf and pf < args.final_low_conf)
                if c["drift"]:
                    drift_events.append(c)

            records.append({
                "idx": i,
                "ground_truth": gt,
                "is_correct": is_correct,
                "extracted_answer": ans,
                "n_commits": len(commits),
                "n_drift_events": len(drift_events),
                "commits": commits,
                "drift_events": drift_events,
                "generation": text[:1500],
            })
            if (i + 1) % 10 == 0:
                pct_corr = 100.0 * sum(r["is_correct"] for r in records) / len(records)
                pct_drift = 100.0 * sum(r["n_drift_events"] > 0 for r in records) / len(records)
                print(
                    f"[{i+1}/{len(loader)}] acc={pct_corr:.1f}%  "
                    f"problems_with_>=1_drift={pct_drift:.1f}%"
                )

    # Aggregate
    n = len(records)
    correct = [r for r in records if r["is_correct"]]
    wrong = [r for r in records if not r["is_correct"]]

    n_drifts = np.array([r["n_drift_events"] for r in records])

    summary = {
        "dataset": args.dataset,
        "n_problems": n,
        "n_correct": len(correct),
        "n_wrong": len(wrong),
        "accuracy": len(correct) / max(n, 1),
        "thresholds": {
            "commit_high_conf": args.commit_high_conf,
            "final_low_conf": args.final_low_conf,
        },
        "drift_per_problem": {
            "mean": float(n_drifts.mean()) if n else 0.0,
            "median": float(np.median(n_drifts)) if n else 0.0,
            "max": int(n_drifts.max()) if n else 0,
            "p25": float(np.percentile(n_drifts, 25)) if n else 0.0,
            "p75": float(np.percentile(n_drifts, 75)) if n else 0.0,
        },
        "distribution_problems_with": {
            "0_drifts": int((n_drifts == 0).sum()),
            "1_drift": int((n_drifts == 1).sum()),
            "2plus_drifts": int((n_drifts >= 2).sum()),
            "5plus_drifts": int((n_drifts >= 5).sum()),
        },
        "drift_in_wrong_problems": {
            "n_with_at_least_one": sum(r["n_drift_events"] > 0 for r in wrong),
            "frac_with_at_least_one": (
                sum(r["n_drift_events"] > 0 for r in wrong) / max(len(wrong), 1)
            ),
            "mean_drifts": float(np.mean([r["n_drift_events"] for r in wrong])) if wrong else 0.0,
        },
        "drift_in_correct_problems": {
            "n_with_at_least_one": sum(r["n_drift_events"] > 0 for r in correct),
            "frac_with_at_least_one": (
                sum(r["n_drift_events"] > 0 for r in correct) / max(len(correct), 1)
            ),
            "mean_drifts": float(np.mean([r["n_drift_events"] for r in correct])) if correct else 0.0,
        },
        # Headline interpretation hints
        "interpretation": {
            "upper_bound_remask_recoverable_failures": (
                sum(r["n_drift_events"] > 0 for r in wrong) / max(len(wrong), 1)
            ),
            "harmful_remask_risk_on_correct": (
                sum(r["n_drift_events"] > 0 for r in correct) / max(len(correct), 1)
            ),
        },
    }

    os.makedirs(Path(args.out).parent, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
