"""
Exercise 01 — GRPO advantage computation.

Format: Fill-in-the-blank from the repo's own code.

This function corresponds to train/trainer.py:912-920 — the piece that turns
flat per-completion rewards into per-completion advantages using a per-prompt
group-mean baseline (GRPO).

===== RULES =====
- NO AI. No Copilot. No Claude. No Googling "repeat_interleave."
- NO peeking at train/trainer.py (you already know where the answer lives).
- You MAY consult notes/02_policy_gradient.md (especially §10.3) and use
  the official PyTorch docs (https://pytorch.org/docs/stable/) for any
  single op you want to look up (e.g. `torch.view`, `torch.mean`).
- Target time: 15-25 min. If you're stuck for 10+ min on one line, that's
  the signal — tell Claude and we'll debug together.

===== WHAT YOU'RE IMPLEMENTING =====
Given:
  - rewards: shape (G,), flat tensor of scalar rewards.
    G = num_prompts × group_size (K).
    Layout: completions for the same prompt are CONSECUTIVE.
      rewards = [r(s_0, a_0_0), r(s_0, a_0_1), ..., r(s_0, a_0_{K-1}),
                 r(s_1, a_1_0), r(s_1, a_1_1), ..., r(s_1, a_1_{K-1}),
                 ...]
  - group_size: K, the number of completions per prompt.

Compute:
  - advantages: shape (G,), same layout as rewards. For each completion,
        advantages[i*K + k] = rewards[i*K + k] - (per-prompt mean of rewards[i*K : i*K + K])

i.e. every completion's advantage is "its reward minus its group's mean."

===== HINTS =====
- Three operations: reshape to (N, K), mean over K, subtract from rewards.
- For the subtraction step, you need to broadcast the per-prompt means back
  across K. Two ways to do this:
    (i) use `.repeat_interleave(K, dim=0)` to expand the mean back to (G,);
    (ii) rely on broadcasting by reshaping differently.
  The repo uses (i) — try that first. Look up `torch.Tensor.repeat_interleave`
  if you haven't used it.
- You should NOT need any explicit for-loops.

===== RUN =====
Just run this file directly:

    python notes/exercises/01_grpo_advantages.py

The tests at the bottom will tell you pass or fail.
"""

import torch


def compute_grpo_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """Compute GRPO per-completion advantages using a per-prompt mean baseline.

    :param rewards: (G,) flat tensor of scalar rewards. G = num_prompts * group_size,
        with completions for the same prompt laid out consecutively.
    :param group_size: K, number of completions sampled per prompt.
    :return: (G,) tensor of advantages. Entry i*K+k equals rewards[i*K+k] minus
        the mean of rewards[i*K : i*K + K].
    """
    # Solution — mirrors the repo's approach at train/trainer.py:918-920.
    #
    # Shape flow:
    #   rewards  (G,)
    #     .view(-1, K)           → (N, K)
    #     .mean(dim=1)           → (N,)   per-prompt mean
    #     .repeat_interleave(K)  → (G,)   broadcast mean back, aligned with rewards
    #   rewards - mean_broadcast → (G,)   per-completion advantage
    rewards_2d = rewards.view(-1, group_size)                     # (N, K)
    mean_per_prompt = rewards_2d.mean(dim=1)                      # (N,)
    mean_broadcast = mean_per_prompt.repeat_interleave(group_size)  # (G,)
    return rewards - mean_broadcast                               # (G,)

    # Equivalent alternative using broadcasting instead of repeat_interleave:
    #
    #     rewards_2d = rewards.view(-1, group_size)      # (N, K)
    #     mean = rewards_2d.mean(dim=1, keepdim=True)    # (N, 1) — keepdim is the trick
    #     return (rewards_2d - mean).view(-1)            # (N, K) broadcast, flatten to (G,)

# ==============================================================================
# Tests — run this file to check your implementation.
# Don't modify these tests.
# ==============================================================================

if __name__ == "__main__":
    # Test 1: basic case, K=2, 2 groups
    rewards = torch.tensor([1.0, 3.0, 10.0, 20.0])
    expected = torch.tensor([-1.0, 1.0, -5.0, 5.0])  # group means: 2.0 and 15.0
    actual = compute_grpo_advantages(rewards, group_size=2)
    assert actual.shape == expected.shape, (
        f"Test 1 shape: expected {expected.shape}, got {actual.shape}"
    ) 
    assert torch.allclose(actual, expected), (
        f"Test 1 values: expected {expected.tolist()}, got {actual.tolist()}"
    )
    print("Test 1 passed — basic K=2, 2 groups")

    # Test 2: all-equal rewards → all-zero advantages
    rewards = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    expected = torch.zeros(8)
    actual = compute_grpo_advantages(rewards, group_size=4)
    assert torch.allclose(actual, expected), (
        f"Test 2: expected all zeros, got {actual.tolist()}"
    )
    print("Test 2 passed — all-equal rewards give zero advantages")

    # Test 3: K=3, different scales per group (tests per-prompt centering)
    rewards = torch.tensor([0.1, 0.5, 0.9, 10.0, 20.0, 30.0])
    expected = torch.tensor([-0.4, 0.0, 0.4, -10.0, 0.0, 10.0])
    actual = compute_grpo_advantages(rewards, group_size=3)
    assert torch.allclose(actual, expected), (
        f"Test 3: expected {expected.tolist()}, got {actual.tolist()}"
    )
    print("Test 3 passed — per-prompt centering across different scales")

    # Test 4: single prompt, K=8 (like the actual GRPO config)
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0])  # 4/8 correct
    mean = 0.5
    expected = torch.tensor([0.5, -0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5])
    actual = compute_grpo_advantages(rewards, group_size=8)
    assert torch.allclose(actual, expected), (
        f"Test 4: expected {expected.tolist()}, got {actual.tolist()}"
    )
    print("Test 4 passed — K=8 GRPO-shaped group")

    # Test 5: edge case, G = group_size (one prompt), all different rewards
    rewards = torch.tensor([0.0, 1.0, 2.0, 3.0])
    expected = torch.tensor([-1.5, -0.5, 0.5, 1.5])  # mean = 1.5
    actual = compute_grpo_advantages(rewards, group_size=4)
    assert torch.allclose(actual, expected), (
        f"Test 5: expected {expected.tolist()}, got {actual.tolist()}"
    )
    print("Test 5 passed — single prompt, distinct rewards")

    # Test 6: verify the INSTANCE matches what the repo does (§10.3 bias note).
    # Each entry i in rewards gets advantage r_i - mean_over_group_containing_i.
    # The group for entry i INCLUDES r_i itself — that's the O(1/K) bias we noted.
    rewards = torch.tensor([5.0, 10.0])  # single group, K=2, mean = 7.5
    actual = compute_grpo_advantages(rewards, group_size=2)
    # Note: in leave-one-out, entry 0's baseline would be 10.0 → advantage -5.0.
    # In the non-LOO version (what GRPO paper + this repo uses), the mean
    # INCLUDES the entry itself → baseline 7.5 → advantage -2.5.
    expected = torch.tensor([-2.5, 2.5])
    assert torch.allclose(actual, expected), (
        f"Test 6: expected {expected.tolist()}, got {actual.tolist()}. "
        f"This tests that your baseline INCLUDES the current completion "
        f"(matching the repo's non-leave-one-out choice)."
    )
    print("Test 6 passed — baseline includes current completion (non-LOO)")

    print("\n[OK] All tests passed. Paste your implementation back to Claude for review.")
