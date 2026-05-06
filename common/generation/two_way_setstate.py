"""2-way set-state policy: a Bernoulli policy where the per-position action is
*"set this position's mask state to {0=unmasked, 1=masked}"*, with the effect
state-dependent:

    state @ pos | sampled action | effect
    ------------|----------------|--------
    masked      | UNMASKED_STATE | reveal (UNMASK)
    masked      | MASKED_STATE   | no-op  (KEEP MASKED)
    unmasked    | UNMASKED_STATE | no-op  (KEEP UNMASKED)
    unmasked    | MASKED_STATE   | remask (REMASK)

Compared to the 3-way framework:
  - Same Bernoulli output_dim=1 as the upstream Apple 2-way policy.
  - No constraint masking (every action valid at every position) — instead
    "no action" emerges naturally when sampled state == current state.
  - Optional conf-aware prior pushes down P(MASKED_STATE) at high-conf
    unmasked positions (suppressing frivolous remasking).
  - Loads upstream 2-way checkpoints directly with `strict=False` if used with
    `DiTConfidencePCurrentPolicy` (the new params are tiny and trained from
    fresh init).

Returns the same `(unmask, remask, sampling_data)` tuple shape as the existing
3-way decisions function so it drops into the dispatch pattern in
`_policy_unmask_decisions`.
"""
import os

import torch

from common.generation.sampling import bernoulli_sample


def apply_setstate_constraints(
    raw_logits: torch.Tensor,
    is_masked: torch.Tensor,
    conf_top1: torch.Tensor,
    sampling_mask: torch.Tensor,
    remask_conf_prior: float,
) -> torch.Tensor:
    """Construct the constrained Bernoulli logits used for sampling AND for
    loss-time log-prob recomputation.

    This is the SHARED helper between the rollout decisions function and the
    trainer's per-timestep log-prob path. Calling it from both sites eliminates
    the v8-style PPO ratio bias: at training time we re-derive the same
    constrained distribution that sampling drew from, so log P_new and log P_old
    are over identical denominators (modulo updated policy weights).

    Construction:
      1. Conf-aware prior shift: at unmasked positions, add
         `remask_conf_prior * conf_top1` to the logit. This raises the logit
         (so sigmoid-prob of "set to UNMASKED" goes up, equivalently prob of
         "remask this token" goes down) when the base model is confident about
         what's there. Set to 0 to disable; 5.0 mirrors the 3-way REMASK
         prior. Wired through from `grpo_config.setstate_remask_conf_prior`.
      2. DISABLE_REMASK env-var clamp (eval-only): if set, force unmasked
         positions in the active block to logit=+1e9 so they always sample
         "set to UNMASKED" — collapsing the policy to upstream Apple 2-way
         semantics where remasking never happens.

    Constraints depend ONLY on rollout state (mask, conf, sampling_mask), not
    on policy parameters, so they have zero gradient. They do however affect
    the sigmoid (the Bernoulli denominator), so applying them at both call
    sites is what keeps the PPO ratio unbiased.

    :param raw_logits: (..., L) raw Bernoulli logits from the policy
    :param is_masked: (..., L) bool — True at currently-masked positions
    :param conf_top1: (..., L) base-model top-1 confidence per position
    :param sampling_mask: (..., L) bool — positions where the policy is
        permitted to act (current block & not-yet-done batch items)
    :return: (..., L) constrained Bernoulli logits, same shape and dtype as
        raw_logits.
    """
    constrained = raw_logits.clone()
    is_unmasked = ~is_masked

    if remask_conf_prior != 0.0:
        prior_shift = (
            remask_conf_prior
            * conf_top1.to(constrained.dtype)
            * is_unmasked.to(constrained.dtype)
        )
        constrained = constrained + prior_shift

    if os.environ.get("DISABLE_REMASK", "0") == "1":
        force_unmasked = is_unmasked & sampling_mask
        constrained = torch.where(
            force_unmasked,
            torch.full_like(constrained, 1e9),
            constrained,
        )

    return constrained


def _policy_two_way_setstate_decisions(
    mask_index: torch.Tensor,
    block_mask_index: torch.Tensor,
    probs: torch.Tensor,
    steps_taken: torch.Tensor,
    block_slice: slice,
    L: int,
    policy,
    policy_type: str,
    full_context: bool,
    confidences_top_p: int,
    model_output,
    prompt_L: int,
    temperature_policy: float,
    generation_part: torch.Tensor,
    mask_id: int,
    remask_conf_prior: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Set-state decisions over the active block.

    :param probs: (B, L_full, V) base-model softmax distribution
    :param generation_part: (B, L_full) current token ids (mask_id at masked
        positions, real ids at unmasked)
    :returns: (unmask, remask, sampling_data) where unmask and remask are
        (B, L_full) bool tensors of positions that should reveal / hide.
    """
    B, L_full = probs.shape[0], probs.shape[1]

    # active block mask: positions where the policy is allowed to act
    block_positions = torch.zeros(
        (B, L_full), dtype=torch.bool, device=probs.device
    )
    block_positions[:, block_slice] = True
    active = (steps_taken < L).unsqueeze(-1)
    sampling_mask = block_positions & active  # (B, L_full)

    # Build policy inputs.
    # confidence input: top-K over the active region (same convention as
    # upstream `_policy_three_way_decisions`).
    per_batch_timestep = steps_taken.unsqueeze(-1) * (1 / L)
    topk_result = probs.topk(confidences_top_p, dim=-1)
    if full_context:
        c_max_input = topk_result.values
        policy_mask = mask_index
    else:
        c_max_input = topk_result.values[:, block_slice]
        policy_mask = block_mask_index

    # New input: probability the base model assigns to the actual current token
    # at each position. At masked positions this is p(mask_id|context) ≈ 0; the
    # policy's `masked_p_embedding` handles those via a learnable swap-in.
    p_current_full = probs.gather(
        -1, generation_part.unsqueeze(-1)
    ).squeeze(-1)  # (B, L_full)
    if full_context:
        p_current_input = p_current_full
    else:
        p_current_input = p_current_full[:, block_slice]

    if policy_type != "dit_confidence_pcurrent":
        raise ValueError(
            f"two_way_setstate sampling expects policy_type='dit_confidence_pcurrent', "
            f"got {policy_type}"
        )
    policy_inputs = (policy_mask, c_max_input, per_batch_timestep, p_current_input)
    policy_logits = policy(*policy_inputs)  # (B, L_active) Bernoulli logits

    # Bernoulli logits are scalar per position. Convention: sigmoid(logit)
    # = P(set state to UNMASKED). So a HIGH logit means "leave/move to
    # unmasked"; a LOW logit means "leave/move to masked."
    if temperature_policy != 1.0:
        policy_logits = policy_logits / temperature_policy

    # Promote block-only output to full-L tensor for downstream uniformity.
    if not full_context:
        full_logits = torch.zeros(
            (B, L_full), dtype=policy_logits.dtype, device=policy_logits.device,
        )
        full_logits[:, block_slice] = policy_logits
        policy_logits_full = full_logits
    else:
        policy_logits_full = policy_logits

    # Apply the SHARED constraint helper. Used identically at loss time by the
    # trainer (`_apply_two_way_setstate_constraints`) so the PPO ratio is
    # unbiased — the v8 PPO bug pattern but for this new sampling mode.
    is_masked = (generation_part == mask_id)
    is_unmasked = ~is_masked
    conf_max_full = probs.max(dim=-1).values  # (B, L_full) base-model top-1
    constrained_logits = apply_setstate_constraints(
        raw_logits=policy_logits_full,
        is_masked=is_masked,
        conf_top1=conf_max_full,
        sampling_mask=sampling_mask,
        remask_conf_prior=remask_conf_prior,
    )

    # Sample. `bernoulli_sample` returns True where sampled action = "set to
    # the high-prob outcome under the logit." Here that's "set to unmasked."
    sampled_set_to_unmasked = bernoulli_sample(
        constrained_logits, mask_index=sampling_mask
    )  # (B, L_full) bool: True if sampled state = UNMASKED

    # Outside the sampling_mask: no transition (keep current state).
    # Inside: derive transitions from (sampled state, current state).
    unmask = is_masked & sampled_set_to_unmasked & sampling_mask  # masked → unmasked
    remask = is_unmasked & (~sampled_set_to_unmasked) & sampling_mask  # unmasked → masked

    sampling_data = {
        "sampling_inputs": constrained_logits.detach(),  # (B, L_full)
        "samples": sampled_set_to_unmasked.detach(),  # (B, L_full) bool
        "sampling_masks": sampling_mask.detach(),  # (B, L_full) bool
        "policy_inputs": tuple(
            pi.detach() if isinstance(pi, torch.Tensor) else pi for pi in policy_inputs
        ),
    }
    return unmask, remask, sampling_data
