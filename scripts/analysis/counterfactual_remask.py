"""Counterfactual remask ablation.

For each problem, run the SAME 3-way policy twice:
  (a) normal (full 3-way action space)
  (b) REMASK action forbidden (logits[..., 2] = -inf before sampling)

Compare accuracy. If (a) > (b), the remask action is *causally* responsible for
some fraction of the wins — not just a better unmask schedule.

Outputs a JSON with per-problem and aggregate deltas.

Usage:
    python -m scripts.analysis.counterfactual_remask \
      --policy_config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_3way_alpha0.yaml \
      --policy_ckpt outputs/3way_alpha0/checkpoint-800/model.safetensors \
      --dataset gsm8k \
      --n_test 100 \
      --out eval_results/analysis/counterfactual_gsm8k.json
"""
import argparse
import contextlib
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from trl import TrlParser

from common.config import Config
from common.generation import generation as gen_module
from common.generation.generation import generate_unified
from common.generation.sampling import ACTION_REMASK
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


EXTRACTORS = {
    "gsm8k": (extract_gsm_answer, check_gsm_correct),
    "math": (extract_math_answer, check_math_correct),
}
DATASETS = {"gsm8k": GSM8KDataset, "math": MATH500Dataset}
MASK_ID_LLADA = 126336


@contextlib.contextmanager
def disable_remask():
    """Monkey-patch the 3-way decision path so REMASK is never chosen."""
    original = gen_module._policy_three_way_decisions

    def patched(*args, **kwargs):
        unmask, remask, data = original(*args, **kwargs)
        # Nuke any remask actions and clear the sampling_inputs remask channel so the
        # log-prob and entropy logging also reflects the constrained distribution.
        if remask is not None:
            remask = torch.zeros_like(remask)
        si = data.get("sampling_inputs")
        if si is not None and si.dim() >= 1 and si.shape[-1] == 3:
            si = si.clone()
            si[..., ACTION_REMASK] = float("-inf")
            data["sampling_inputs"] = si
        return unmask, remask, data

    gen_module._policy_three_way_decisions = patched
    try:
        yield
    finally:
        gen_module._policy_three_way_decisions = original


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


def run_one(model, policy, prompt_ids, gen_length, block_length, config):
    result = generate_unified(
        model=model,
        prompt=prompt_ids,
        remasking="policy",
        policy=policy,
        gen_length=gen_length,
        block_length=block_length,
        temperature=0.0,
        mask_id=MASK_ID_LLADA,
        sampling_mode=config.sampling_mode,
        full_context=config.policy_full_context,
        confidences_top_p=config.confidences_top_p,
        model_type="LLaDA",
        temperature_policy=1.0,
    )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy_config", required=True)
    p.add_argument("--policy_ckpt", required=True)
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--n_test", type=int, default=50)
    p.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    parser = TrlParser((Config,))
    (config,) = parser.parse_args_and_config(
        args=["--config", args.policy_config], fail_with_unknown_args=False
    )
    assert config.sampling_mode == "three_way", "This script is for 3-way only."

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
    if args.n_test < len(ds):
        ds = torch.utils.data.Subset(ds, range(args.n_test))
    loader = DataLoader(ds, batch_size=1, collate_fn=ds.dataset.collate_fn if hasattr(ds, "dataset") else ds.collate_fn)

    extract_fn, check_fn = EXTRACTORS[args.dataset]
    records = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            pid = batch["input_ids"].to(device)
            gt = batch["answers"][0]

            # Full 3-way
            out_full = run_one(model, policy, pid, config.max_completion_length, config.block_length, config)
            text_full = tokenizer.decode(
                out_full.sequences[0, -config.max_completion_length:], skip_special_tokens=True
            )
            ans_full = extract_fn(text_full)
            correct_full = check_fn(ans_full, gt)
            steps_full = int(out_full.steps_taken.item())

            # Remask disabled
            with disable_remask():
                out_noremask = run_one(
                    model, policy, pid, config.max_completion_length, config.block_length, config
                )
            text_noremask = tokenizer.decode(
                out_noremask.sequences[0, -config.max_completion_length:], skip_special_tokens=True
            )
            ans_noremask = extract_fn(text_noremask)
            correct_noremask = check_fn(ans_noremask, gt)
            steps_noremask = int(out_noremask.steps_taken.item())

            records.append({
                "idx": i,
                "ground_truth": gt,
                "full": {"correct": bool(correct_full), "answer": ans_full,
                         "generation": text_full, "steps": steps_full},
                "noremask": {"correct": bool(correct_noremask), "answer": ans_noremask,
                             "generation": text_noremask, "steps": steps_noremask},
            })
            print(f"[{i+1}/{len(loader)}] full={correct_full} noremask={correct_noremask}")

    acc_full = sum(r["full"]["correct"] for r in records) / max(len(records), 1)
    acc_noremask = sum(r["noremask"]["correct"] for r in records) / max(len(records), 1)
    nfe_full = sum(r["full"]["steps"] for r in records) / max(len(records), 1)
    nfe_noremask = sum(r["noremask"]["steps"] for r in records) / max(len(records), 1)
    summary = {
        "dataset": args.dataset,
        "n": len(records),
        "acc_full_3way": acc_full,
        "acc_remask_disabled": acc_noremask,
        "acc_delta_causal_remask": acc_full - acc_noremask,
        "nfe_full_3way": nfe_full,
        "nfe_remask_disabled": nfe_noremask,
        "n_problems_flip_to_correct_when_remask_enabled": sum(
            (r["full"]["correct"] and not r["noremask"]["correct"]) for r in records
        ),
        "n_problems_flip_to_wrong_when_remask_enabled": sum(
            (not r["full"]["correct"] and r["noremask"]["correct"]) for r in records
        ),
    }
    os.makedirs(Path(args.out).parent, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
