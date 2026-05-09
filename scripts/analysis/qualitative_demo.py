"""Qualitative demo: 2-way vs 3-way vs counterfactual remask probe.

For each given GSM8K problem (default: hand-picked diverse subset):

  1. Run 2-way policy generation, record trajectory.
  2. Run 3-way warm-started policy generation, record trajectory.
  3. Identify "drift" tokens in the 2-way generation (committed with p > drift_commit_thresh
     at commit time, but base-model agreement falls below drift_final_thresh after full
     context). These are positions where remasking *could* plausibly fix a wrong commit.
  4. For each drift token, perform a counterfactual probe: mask only that single position,
     run one base-model forward pass, and read the new argmax. Report whether the new
     token differs and what it is.
  5. Cross-reference: did the learned 3-way policy actually remask any of these
     positions during its own generation?

Emits a markdown report `{out}.md` showing all three policies side-by-side per problem,
with the drift-and-counterfactual-probe table inlined.

Usage:
    python -m scripts.analysis.qualitative_demo \
      --two_way_ckpt   pretrained_checkpoints/2way_ckpt2000.safetensors \
      --two_way_config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
      --three_way_ckpt   pretrained_checkpoints/3way_v8_ckpt1000.safetensors \
      --three_way_config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_3way_alpha0.yaml \
      --n_problems 5 \
      --out eval_results/analysis/qualitative_demo
"""
import argparse
import json
import os
import random
from pathlib import Path

import torch
from safetensors.torch import load_file
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
    check_gsm_correct,
)
from data.loaders.gsm8k import GSM8KDataset


MASK_ID_LLADA = 126336


def load_policy(config, base_model, device):
    if config.policy_type == "dit_hidden":
        core = DiTHiddenStatePolicy(
            dllm=base_model,
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


def parse_config(yaml_path):
    parser = TrlParser((Config,))
    (config,) = parser.parse_args_and_config(
        args=["--config", yaml_path], fail_with_unknown_args=False
    )
    return config


@torch.no_grad()
def generate_with_trajectory(model, policy, prompt_ids, attn_mask, config, device):
    """Run policy generation with trajectory recording. Returns the full result."""
    return generate_unified(
        model=model,
        prompt=prompt_ids,
        remasking="policy",
        policy=policy,
        gen_length=config.max_completion_length,
        block_length=config.block_length,
        temperature=0.0,
        mask_id=MASK_ID_LLADA,
        sampling_mode=config.sampling_mode,
        full_context=config.policy_full_context,
        confidences_top_p=config.confidences_top_p,
        model_type="LLaDA",
        attention_mask=attn_mask,
        temperature_policy=1.0,
        record_trajectory=True,
        use_committed_token_conf=getattr(config, "use_committed_token_conf", False),
        remask_conf_prior_strength=getattr(config, "remask_conf_prior_strength", 5.0),
    )


def extract_commits(trajectory):
    """For each unmask event, record (position, token_id, denoising_step, p_commit)."""
    commits = []
    for step in trajectory:
        unmask = step["unmask"][0].bool()
        x0 = step["x0"][0]
        conf = step["confidence"][0]
        positions = unmask.nonzero(as_tuple=True)[0]
        for pos in positions.tolist():
            commits.append({
                "position": pos,
                "token": int(x0[pos].item()),
                "step": step["step"],
                "p_commit": float(conf[pos].item()),
            })
    return commits


def extract_remask_positions(trajectory):
    """Positions that the 3-way policy chose to REMASK (and at what step)."""
    events = []
    for step in trajectory:
        remask = step.get("remask")
        if remask is None:
            continue
        rm = remask[0].bool()
        positions = rm.nonzero(as_tuple=True)[0]
        for pos in positions.tolist():
            events.append({"position": pos, "step": step["step"]})
    return events


@torch.no_grad()
def get_pfinal_at_each_pos(model, sequence, prompt_L, attn_mask):
    """Return p_final[i] = probs[i, sequence[prompt_L+i]] under the full-context forward."""
    out = model(sequence, attention_mask=attn_mask)
    logits = out.logits[:, prompt_L:]
    probs = torch.softmax(logits.to(torch.float32), dim=-1)  # (1, L, V)
    gen_part = sequence[:, prompt_L:]  # (1, L)
    p_final = probs.gather(-1, gen_part.unsqueeze(-1)).squeeze(-1)[0].cpu()  # (L,)
    return p_final


@torch.no_grad()
def counterfactual_remask_probe(model, sequence, prompt_L, position, attn_mask):
    """Set position-only mask, forward once, return new top-1 (token_id, prob).

    sequence: (1, L_total) including prompt + completion. position is in [0, gen_length).
    """
    probe_seq = sequence.clone()
    probe_seq[0, prompt_L + position] = MASK_ID_LLADA
    out = model(probe_seq, attention_mask=attn_mask)
    logits = out.logits[:, prompt_L:]
    probs = torch.softmax(logits.to(torch.float32), dim=-1)  # (1, L, V)
    new_top1_id = int(probs[0, position].argmax().item())
    new_top1_prob = float(probs[0, position, new_top1_id].item())
    return new_top1_id, new_top1_prob


def fmt_token(tokenizer, tok_id):
    """Decode a single token, escape newlines for compact display."""
    s = tokenizer.decode([tok_id], skip_special_tokens=False)
    return s.replace("\n", "\\n").replace("\r", "")


def truncate_text(text, max_chars=900):
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--two_way_ckpt", required=True)
    p.add_argument("--two_way_config", required=True)
    p.add_argument("--three_way_ckpt", required=True)
    p.add_argument("--three_way_config", required=True)
    p.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--n_problems", type=int, default=5,
                   help="Number of GSM8K problems to demo")
    p.add_argument("--problem_indices", type=str, default=None,
                   help="Comma-separated list of dataset indices to use; overrides n_problems")
    p.add_argument("--out", required=True,
                   help="Output prefix (writes <out>.md and <out>.json)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drift_commit_thresh", type=float, default=0.5)
    p.add_argument("--drift_final_thresh", type=float, default=0.2)
    p.add_argument("--max_drift_per_problem", type=int, default=5,
                   help="Cap on drift tokens reported per problem (top-N by p_commit-p_final gap)")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading base model on {device} ...")
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    two_way_config = parse_config(args.two_way_config)
    three_way_config = parse_config(args.three_way_config)

    print("Loading 2-way policy...")
    two_way_policy = load_policy(two_way_config, model, device)
    two_way_policy.load_state_dict(load_file(args.two_way_ckpt), strict=False)
    two_way_policy.eval()

    print("Loading 3-way policy...")
    three_way_policy = load_policy(three_way_config, model, device)
    three_way_policy.load_state_dict(load_file(args.three_way_ckpt), strict=False)
    three_way_policy.eval()

    print("Loading dataset...")
    ds = GSM8KDataset(tokenizer=tokenizer, subsample=-1, num_examples=0, add_reasoning=True)

    if args.problem_indices:
        indices = [int(x) for x in args.problem_indices.split(",")]
    else:
        # Spread out indices for diversity
        all_idx = list(range(len(ds)))
        random.shuffle(all_idx)
        indices = all_idx[: args.n_problems]
        indices.sort()
    print(f"Demoing problems: {indices}")

    md_lines = ["# Qualitative Demo — 2-way vs 3-way vs Counterfactual Remask Probe", ""]
    md_lines.append(f"Base model: `{args.model_path}` (frozen, temperature=0).  ")
    md_lines.append(f"2-way policy: `{args.two_way_ckpt}`  ")
    md_lines.append(f"3-way policy: `{args.three_way_ckpt}`  ")
    md_lines.append(f"Drift threshold: `p_commit > {args.drift_commit_thresh}` AND `p_final < {args.drift_final_thresh}`")
    md_lines.append("")

    json_records = []

    for prob_idx, ds_i in enumerate(indices):
        # GSM8KDataset.__getitem__ returns (prompt, question, answer) tuple.
        # Use the dataset's collate_fn to tokenize and produce tensors consistently.
        raw = ds[ds_i]
        batch = ds.collate_fn([raw])
        prompt_text = batch["questions"][0]  # human-readable question text
        gt = batch["answers"][0]
        prompt_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].bool().to(device)
        prompt_L = prompt_ids.shape[1]

        print(f"\n=== Problem {prob_idx+1}/{len(indices)} (dataset idx {ds_i}) ===")

        # 1. 2-way generation
        result_2 = generate_with_trajectory(
            model, two_way_policy, prompt_ids, attn_mask, two_way_config, device
        )
        text_2 = tokenizer.decode(
            result_2.sequences[0, prompt_L:], skip_special_tokens=True
        )
        ans_2 = extract_gsm_answer(text_2)
        ok_2 = bool(check_gsm_correct(ans_2, gt))
        commits_2 = extract_commits(result_2.trajectory)

        # 2. 3-way generation
        result_3 = generate_with_trajectory(
            model, three_way_policy, prompt_ids, attn_mask, three_way_config, device
        )
        text_3 = tokenizer.decode(
            result_3.sequences[0, prompt_L:], skip_special_tokens=True
        )
        ans_3 = extract_gsm_answer(text_3)
        ok_3 = bool(check_gsm_correct(ans_3, gt))
        commits_3 = extract_commits(result_3.trajectory)
        remasks_3 = extract_remask_positions(result_3.trajectory)
        remask_positions_3 = sorted({r["position"] for r in remasks_3})

        # 3. Drift on 2-way's generation: which positions are "suspicious"
        seq_attn = torch.ones(
            (1, result_2.sequences.shape[1]), dtype=torch.float, device=device
        )
        seq_attn[:, :prompt_L] = attn_mask.float()
        seq_attn = seq_attn.to(model.dtype)
        p_final_2 = get_pfinal_at_each_pos(
            model, result_2.sequences, prompt_L, seq_attn
        )

        drift_tokens = []
        for c in commits_2:
            pos = c["position"]
            pc = c["p_commit"]
            pf = float(p_final_2[pos].item())
            if pc > args.drift_commit_thresh and pf < args.drift_final_thresh:
                drift_tokens.append({**c, "p_final": pf, "gap": pc - pf})
        drift_tokens.sort(key=lambda d: -d["gap"])
        drift_tokens = drift_tokens[: args.max_drift_per_problem]

        # 4. Counterfactual remask probe on each drift token
        for d in drift_tokens:
            new_id, new_prob = counterfactual_remask_probe(
                model, result_2.sequences, prompt_L, d["position"], seq_attn
            )
            d["counterfactual_token"] = fmt_token(tokenizer, new_id)
            d["counterfactual_prob"] = new_prob
            d["original_token"] = fmt_token(tokenizer, d["token"])
            d["changes"] = (new_id != d["token"])
            d["remasked_by_3way"] = (d["position"] in remask_positions_3)

        # ---- Markdown writeup ----
        md_lines.append(f"## Problem {prob_idx+1} (dataset idx {ds_i})")
        md_lines.append("")
        md_lines.append("**Question:**")
        md_lines.append("")
        md_lines.append("> " + prompt_text.replace("\n", "\n> "))
        md_lines.append("")
        md_lines.append(f"**Ground truth:** `{gt}`")
        md_lines.append("")
        md_lines.append(f"### 2-way (NFE {result_2.steps_taken[0].item()})  →  {'✅' if ok_2 else '❌'}  predicted `{ans_2}`")
        md_lines.append("```")
        md_lines.append(truncate_text(text_2))
        md_lines.append("```")
        md_lines.append("")
        md_lines.append(f"### 3-way v8 (NFE {result_3.steps_taken[0].item()}, remasks fired: {len(remasks_3)})  →  {'✅' if ok_3 else '❌'}  predicted `{ans_3}`")
        md_lines.append("```")
        md_lines.append(truncate_text(text_3))
        md_lines.append("```")
        md_lines.append("")
        if drift_tokens:
            md_lines.append("### Drift tokens in 2-way's generation + counterfactual remask probe")
            md_lines.append("")
            md_lines.append(
                "| Pos | Step | Original | p_commit | p_final | If we remasked it → | new prob | Changes? | 3-way also remasked? |"
            )
            md_lines.append("|---|---|---|---|---|---|---|---|---|")
            for d in drift_tokens:
                md_lines.append(
                    f"| {d['position']} | {d['step']} | `{d['original_token']}` | "
                    f"{d['p_commit']:.3f} | {d['p_final']:.3f} | "
                    f"`{d['counterfactual_token']}` | {d['counterfactual_prob']:.3f} | "
                    f"{'**yes**' if d['changes'] else 'no'} | "
                    f"{'**yes**' if d['remasked_by_3way'] else 'no'} |"
                )
            md_lines.append("")
        else:
            md_lines.append("_No drift tokens above threshold for this problem._")
            md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

        json_records.append({
            "problem_idx": prob_idx,
            "dataset_idx": ds_i,
            "question": prompt_text,
            "ground_truth": str(gt),
            "two_way": {
                "generation": text_2,
                "extracted_answer": str(ans_2),
                "is_correct": ok_2,
                "n_NFE": int(result_2.steps_taken[0].item()),
                "n_commits": len(commits_2),
            },
            "three_way": {
                "generation": text_3,
                "extracted_answer": str(ans_3),
                "is_correct": ok_3,
                "n_NFE": int(result_3.steps_taken[0].item()),
                "n_remasks": len(remasks_3),
                "remask_positions": remask_positions_3,
            },
            "drift_tokens_in_two_way": drift_tokens,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md_path = out_path.with_suffix(".md")
    json_path = out_path.with_suffix(".json")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    with open(json_path, "w") as f:
        json.dump(json_records, f, indent=2)
    print(f"\nWrote {md_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
