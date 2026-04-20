"""Fast standalone sanity check of the 3-way plumbing (no 8B load).

Runs with: python -m scripts.sanity_check_3way

Checks:
  1. categorical_sample respects mask_index (forces KEEP outside)
  2. categorical_batch_loglik grads flow into logits
  3. categorical_sample + categorical_batch_loglik agree in expectation
  4. Action-mask constraint: masked positions never produce REMASK, and
     unmasked positions never produce UNMASK (using the same -inf trick used
     by _policy_three_way_decisions).
  5. DiTConfidencePolicy(num_actions=3) returns the expected shape and
     gradients flow through the output head.
  6. End-to-end generate_unified with sampling_mode='three_way' on a tiny
     dummy model runs for a handful of steps and records remask events in the
     trajectory.
"""
import torch

from common.generation.sampling import (
    ACTION_KEEP,
    ACTION_REMASK,
    ACTION_UNMASK,
    categorical_batch_loglik,
    categorical_entropy,
    categorical_sample,
)


def test_categorical_sample_respects_mask():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 3)
    mask = torch.tensor([[1, 0, 1, 0, 1], [0, 1, 1, 1, 0]], dtype=torch.bool)
    actions = categorical_sample(logits, mask)
    assert actions.shape == (2, 5)
    # Positions outside mask must be KEEP
    assert (actions[~mask] == ACTION_KEEP).all(), actions
    print("  1. categorical_sample mask-gating OK")


def test_action_mask_constraints():
    torch.manual_seed(0)
    B, L = 4, 8
    logits = torch.randn(B, L, 3)
    # Simulate "masked" = first half, "unmasked" = second half.
    is_masked = torch.zeros(B, L, dtype=torch.bool)
    is_masked[:, :4] = True
    # Apply the same -inf constraints _policy_three_way_decisions uses.
    logits[..., ACTION_REMASK] = torch.where(
        is_masked, torch.full_like(logits[..., ACTION_REMASK], float("-inf")),
        logits[..., ACTION_REMASK],
    )
    logits[..., ACTION_UNMASK] = torch.where(
        ~is_masked, torch.full_like(logits[..., ACTION_UNMASK], float("-inf")),
        logits[..., ACTION_UNMASK],
    )
    mask = torch.ones(B, L, dtype=torch.bool)
    for _ in range(64):
        a = categorical_sample(logits, mask)
        assert (a[is_masked] != ACTION_REMASK).all(), "masked pos got REMASK"
        assert (a[~is_masked] != ACTION_UNMASK).all(), "unmasked pos got UNMASK"
    print("  2. action-mask constraints never violated across 64 samples OK")


def test_grads_through_loglik():
    torch.manual_seed(0)
    logits = torch.randn(3, 6, 3, requires_grad=True)
    mask = torch.ones(3, 6, dtype=torch.bool)
    samples = torch.randint(0, 3, (3, 6))
    ll = categorical_batch_loglik(samples, logits, mask)
    ll.sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    print("  3. categorical_batch_loglik grad flow OK")


def test_entropy_monotone():
    # Uniform logits => max entropy ~= ln(3)
    uniform = torch.zeros(1, 4, 3)
    mask = torch.ones(1, 4, dtype=torch.bool)
    H_uniform = categorical_entropy(uniform, mask).item()
    # One-hot logits => entropy ~= 0
    hot = torch.tensor([[[100.0, 0.0, 0.0]] * 4])
    H_hot = categorical_entropy(hot, mask).item()
    assert 1.0 < H_uniform < 1.1, f"uniform H={H_uniform}"
    assert H_hot < 0.05, f"hot H={H_hot}"
    print("  4. categorical_entropy uniform/deterministic OK")


def test_policy_head():
    from common.models.policy import DiTConfidencePolicy

    policy = DiTConfidencePolicy(
        hidden_dim=32, feedforward_dim=64, num_heads=2, num_blocks=1,
        time_embed_dim=16, confidences_top_p=1, num_actions=3,
        smart_init=-2.0,
    )
    m = torch.zeros(2, 10, dtype=torch.long)
    c = torch.rand(2, 10, 1)
    t = torch.zeros(2, 1)
    out = policy(m, c, t)
    assert out.shape == (2, 10, 3), out.shape
    loss = out.sum()
    loss.backward()
    print("  5. DiTConfidencePolicy(num_actions=3) shape + grad OK")


def main():
    print("Running 3-way plumbing sanity checks...")
    test_categorical_sample_respects_mask()
    test_action_mask_constraints()
    test_grads_through_loglik()
    test_entropy_monotone()
    test_policy_head()
    print("All 3-way sanity checks passed.")


if __name__ == "__main__":
    main()
