"""Build a 3-way policy checkpoint warm-started from a trained 2-way checkpoint.

The 3-way policy architecture is identical to 2-way except for the final output
projection (Linear(hidden_dim, 1) vs Linear(hidden_dim, 3)). This script copies
all upstream weights and constructs the new output head so that:

    UNMASK row (0) = 2-way weights + bias   (preserves learned unmasking policy)
    KEEP   row (1) = zeros                  (neutral baseline)
    REMASK row (2) = zero weights + bias=-4 (available but rare at init)

The resulting checkpoint can be loaded as the starting weights for a 3-way
training run — the policy will behave nearly identically to the 2-way baseline
at step 0 (because REMASK is heavily suppressed), providing non-zero reward
signal to GRPO immediately.

Usage:
    python -m scripts.warm_start_3way_from_2way \
        --two_way_ckpt /path/to/2way/checkpoint-700/model.safetensors \
        --out_ckpt outputs/3way_warmstart/checkpoint-0/model.safetensors \
        --remask_bias -4.0
"""
import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--two_way_ckpt", required=True,
                   help="Path to the 2-way model.safetensors to warm-start from")
    p.add_argument("--out_ckpt", required=True,
                   help="Path where the 3-way model.safetensors should be written")
    p.add_argument("--remask_bias", type=float, default=-4.0,
                   help="Initial bias for the REMASK row (logit). "
                        "Lower = rarer at init. Default -4 ≈ P(remask)=0.018 at init.")
    args = p.parse_args()

    two_way = Path(args.two_way_ckpt)
    assert two_way.exists(), f"2-way checkpoint not found: {two_way}"

    state = load_file(str(two_way))
    print(f"Loaded 2-way checkpoint with {len(state)} tensors")

    # Find the 2-way output_proj keys (prefix depends on whether HF wrapper was used)
    weight_keys = [k for k in state if k.endswith("output_proj.weight")]
    bias_keys = [k for k in state if k.endswith("output_proj.bias")]
    assert len(weight_keys) == 1, f"Expected exactly 1 output_proj.weight, found {weight_keys}"
    assert len(bias_keys) == 1, f"Expected exactly 1 output_proj.bias, found {bias_keys}"
    w_key, b_key = weight_keys[0], bias_keys[0]

    w2 = state[w_key]   # shape (1, hidden_dim)
    b2 = state[b_key]   # shape (1,)
    assert w2.shape[0] == 1 and b2.shape[0] == 1, (
        f"2-way ckpt output_proj should have shape (1, H) / (1,), got {w2.shape} / {b2.shape}"
    )
    hidden_dim = w2.shape[1]
    print(f"  UNMASK row copied from 2-way: bias={b2.item():.4f}, weight_norm={w2.norm().item():.4f}")

    # Construct 3-way output projection
    w3 = torch.zeros(3, hidden_dim, dtype=w2.dtype)
    b3 = torch.zeros(3, dtype=b2.dtype)
    w3[0] = w2[0]              # UNMASK: copy learned 2-way weights
    b3[0] = b2[0]              # UNMASK: copy learned 2-way bias
    # row 1 KEEP:   zeros (already)
    # row 2 REMASK: zero weights, bias = remask_bias
    b3[2] = args.remask_bias

    state[w_key] = w3
    state[b_key] = b3
    print(f"  KEEP   row: bias={b3[1].item():.4f} (zeros)")
    print(f"  REMASK row: bias={b3[2].item():.4f} (fresh)")

    out = Path(args.out_ckpt)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(out))
    print(f"Wrote warm-started 3-way checkpoint: {out}")
    print(f"  Total tensors: {len(state)}")


if __name__ == "__main__":
    main()
