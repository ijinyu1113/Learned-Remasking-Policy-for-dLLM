"""
Visualize denoising trajectories of an LLaDA-style dLLM under different remasking policies.

Runs a single prompt under any combination of {random, low_confidence, fastdllm, policy}
and emits a self-contained HTML file with one column per policy. Each row is a denoising
step; tokens just unmasked at that step are highlighted, masked positions show as ░, and
already-committed tokens render normally. Hover any token to see its confidence.

Usage:
  python -m scripts.visualize_trajectories \
    --model_path GSAI-ML/LLaDA-8B-Instruct \
    --prompt "Natalia sold clips to 48 of her friends in April..." \
    --policies random low_confidence fastdllm \
    --gen_length 128 --block_length 32 \
    --out trajectory.html

To include a learned policy, also pass:
  --policies random low_confidence policy \
    --policy_config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
    --policy_ckpt outputs/<run>/checkpoint-<name>
"""
import argparse
import html
import os

import torch
from transformers import AutoModel, AutoTokenizer

from common.generation.generation import generate_unified

MASK_ID_LLADA = 126336


def run_one(model, tokenizer, prompt_ids, policy_name, gen_length, block_length, mask_id,
            thres=None, steps=None, policy=None, temperature_policy=0.0):
    common = dict(
        model=model,
        prompt=prompt_ids,
        gen_length=gen_length,
        block_length=block_length,
        mask_id=mask_id,
        sampling_mode="bernoulli",
        temperature=0.0,
        record_trajectory=True,
    )
    if policy_name == "random":
        result = generate_unified(remasking="random", steps=gen_length // 1, **common)
    elif policy_name == "low_confidence":
        result = generate_unified(remasking="low_confidence", steps=gen_length, **common)
    elif policy_name == "fastdllm":
        result = generate_unified(remasking="fastdllm", thres=thres if thres is not None else 0.7, **common)
    elif policy_name == "policy":
        if policy is None:
            raise ValueError("--policy_ckpt and --policy_config are required for policy")
        result = generate_unified(
            remasking="policy", policy=policy,
            temperature_policy=temperature_policy, **common,
        )
    else:
        raise ValueError(f"Unknown policy: {policy_name}")
    return result


def render_step_html(tokenizer, step, prev_step, prompt_L, mask_id):
    """Render generation portion of one step as HTML span row."""
    x = step["x"][0]  # (L+P,)
    gen = x[prompt_L:].tolist()
    conf = step["confidence"][0].tolist()  # (L,)
    unmask = step["unmask"][0].tolist()  # (L,)
    x0 = step["x0"][0].tolist()  # (L,)
    remask = step.get("remask")
    remask_list = remask[0].tolist() if remask is not None else [False] * len(gen)

    # Tokens that were masked before this step but committed at this step
    just_committed = set(i for i, u in enumerate(unmask) if u)
    just_remasked = set(i for i, r in enumerate(remask_list) if r)

    parts = []
    for i, tok in enumerate(gen):
        c = conf[i]
        if i in just_remasked:
            # Show the pre-remask token with a red strikethrough marker
            text = tokenizer.decode([tok], skip_special_tokens=False)
            text = html.escape(text).replace("\n", "↵")
            parts.append(
                f'<span class="remasked" title="conf_before={c:.3f}">✗{text}</span>'
            )
        elif i in just_committed:
            # Show the predicted token (x0) — that's what's about to be committed
            text = tokenizer.decode([x0[i]], skip_special_tokens=False)
            text = html.escape(text).replace("\n", "↵")
            parts.append(
                f'<span class="just" title="conf={c:.3f}">{text}</span>'
            )
        elif tok == mask_id:
            parts.append('<span class="mask">░</span>')
        else:
            text = tokenizer.decode([tok], skip_special_tokens=False)
            text = html.escape(text).replace("\n", "↵")
            parts.append(f'<span class="kept" title="conf={c:.3f}">{text}</span>')
    return "".join(parts)


def render_html(tokenizer, results_by_policy, prompt_L, mask_id, prompt_text, out_path):
    css = """
    body { font-family: -apple-system, sans-serif; background:#0e0e10; color:#ddd; padding:20px; }
    h1 { color:#fff; }
    .prompt { background:#1a1a1d; padding:10px; border-radius:6px; margin-bottom:20px; white-space:pre-wrap; }
    .grid { display:flex; gap:16px; align-items:flex-start; }
    .col { flex:1; min-width:0; background:#15151a; padding:10px; border-radius:6px; }
    .col h2 { margin:0 0 8px 0; font-size:14px; color:#9af; text-transform:uppercase; }
    .step { font-family: ui-monospace, monospace; font-size:12px; line-height:1.6; padding:4px 0; border-top:1px solid #222; word-break:break-all; }
    .step-num { color:#666; font-size:10px; display:inline-block; width:32px; }
    .mask { color:#444; }
    .kept { color:#aaa; }
    .just { color:#fff; background:#2a6; padding:0 1px; border-radius:2px; }
    .remasked { color:#fff; background:#c44; padding:0 1px; border-radius:2px; text-decoration:line-through; }
    .final { color:#9af; }
    """
    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<style>{css}</style></head><body>",
        "<h1>Denoising trajectories</h1>",
        f"<div class='prompt'><b>Prompt:</b> {html.escape(prompt_text)}</div>",
        "<div class='grid'>",
    ]
    for name, result in results_by_policy.items():
        traj = result.trajectory
        html_parts.append(f"<div class='col'><h2>{html.escape(name)} &middot; {len(traj)} steps</h2>")
        for step in traj:
            row = render_step_html(tokenizer, step, None, prompt_L, mask_id)
            html_parts.append(
                f"<div class='step'><span class='step-num'>{step['step']:>3}</span>{row}</div>"
            )
        # final row
        final = result.sequences[0, prompt_L:].tolist()
        final_text = tokenizer.decode(final, skip_special_tokens=False)
        final_text = html.escape(final_text).replace("\n", "↵")
        html_parts.append(
            f"<div class='step'><span class='step-num final'>FIN</span><span class='final'>{final_text}</span></div>"
        )
        html_parts.append("</div>")
    html_parts.append("</div></body></html>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))
    print(f"Wrote {out_path}")


def maybe_load_policy(args, model, device):
    if "policy" not in args.policies:
        return None
    if not args.policy_config or not args.policy_ckpt:
        raise ValueError("--policy_config and --policy_ckpt required when 'policy' is in --policies")
    # Lazy import to avoid pulling training deps for the no-policy case
    from omegaconf import OmegaConf
    from common.models.policy import DiTConfidencePolicy, DiTHiddenStatePolicy, PolicyHFWrapper
    cfg = OmegaConf.load(args.policy_config)
    grpo = cfg.grpo if "grpo" in cfg else cfg
    ptype = grpo.policy_type
    num_actions = int(grpo.get("num_policy_actions", 1) or 1)
    if ptype == "dit_confidence":
        hidden_dim = grpo.get("policy_hidden_dim", 128) or 128
        ff = grpo.get("policy_feedforward_dim", None) or 4 * hidden_dim
        core = DiTConfidencePolicy(
            hidden_dim=hidden_dim,
            feedforward_dim=ff,
            num_heads=grpo.policy_num_heads,
            dropout=grpo.get("policy_dropout", 0.0),
            num_blocks=grpo.policy_num_blocks,
            time_embed_dim=grpo.policy_time_embed_dim,
            time_period=grpo.get("policy_time_period", 1.0),
            confidences_top_p=grpo.get("confidences_top_p", 1),
            num_actions=num_actions,
        ).to(device)
    elif ptype == "dit_hidden":
        core = DiTHiddenStatePolicy(
            dllm=model,
            time_embed_dim=grpo.policy_time_embed_dim,
            num_blocks=grpo.policy_num_blocks,
            smart_init=grpo.get("policy_smart_init", None),
            time_period=grpo.get("policy_time_period", 1.0),
            num_actions=num_actions,
        ).to(device)
    else:
        raise ValueError(f"Unknown policy_type: {ptype}")
    policy = PolicyHFWrapper(core, ptype).to(device)
    # Load weights (safetensors or torch)
    ckpt_path = args.policy_ckpt
    if os.path.isdir(ckpt_path):
        for fname in ("model.safetensors", "policy.pt", "pytorch_model.bin"):
            cand = os.path.join(ckpt_path, fname)
            if os.path.exists(cand):
                ckpt_path = cand
                break
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(ckpt_path)
    else:
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
    policy.load_state_dict(state, strict=False)
    policy.eval()
    return policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--prompt", required=True)
    p.add_argument("--policies", nargs="+", default=["random", "low_confidence", "fastdllm"],
                   choices=["random", "low_confidence", "fastdllm", "policy"])
    p.add_argument("--gen_length", type=int, default=128)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--fastdllm_thres", type=float, default=0.7)
    p.add_argument("--policy_config", default=None)
    p.add_argument("--policy_ckpt", default=None)
    p.add_argument("--out", default="trajectory.html")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model {args.model_path} ...")
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    policy = maybe_load_policy(args, model, device)

    # Format prompt with chat template (matches gsm8k loader behavior)
    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    prompt_L = prompt_ids.shape[1]

    results = {}
    for name in args.policies:
        print(f"Running {name} ...")
        results[name] = run_one(
            model, tokenizer, prompt_ids, name,
            gen_length=args.gen_length,
            block_length=args.block_length,
            mask_id=MASK_ID_LLADA,
            thres=args.fastdllm_thres,
            policy=policy,
        )
        print(f"  steps_taken={results[name].steps_taken.tolist()} traj_len={len(results[name].trajectory)}")

    render_html(tokenizer, results, prompt_L, MASK_ID_LLADA, args.prompt, args.out)


if __name__ == "__main__":
    main()
