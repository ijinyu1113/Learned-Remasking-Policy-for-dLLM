#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
import os
from typing import NamedTuple

import torch
import torch.nn.functional as F

from common.generation.sampling import ACTION_KEEP
from common.generation.sampling import ACTION_REMASK
from common.generation.sampling import ACTION_UNMASK
from common.generation.sampling import bernoulli_sample
from common.generation.sampling import categorical_sample
from common.generation.sampling import dpls_sample


class GenerationResult(NamedTuple):
    sequences: torch.Tensor  # Generated sequences (B, prompt_L + gen_L)
    steps_taken: torch.Tensor  # Steps taken per batch item

    # Policy training data (None for non-policy modes)
    sampling_inputs: torch.Tensor | None = None  # (B, T, BL)
    samples: torch.Tensor | None = None  # (B, T, BL)
    sampling_masks: torch.Tensor | None = None  # (B, T, BL)
    policy_inputs: tuple[torch.Tensor, ...] | None = None
    still_masked: torch.Tensor | None = None  # (B,)

    # Trajectory capture (None unless record_trajectory=True)
    trajectory: list | None = None  # list of per-step dicts, see generate_unified docstring


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0.0:
        return logits
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


@torch.no_grad()
def generate_unified(
    model,
    prompt: torch.Tensor,
    remasking: str,
    policy=None,
    thres: float | torch.Tensor | None = None,
    steps: int | None = None,
    gen_length: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    mask_id: int = 126336,
    sampling_mode: str = "bernoulli",
    dpls_stop_logit: float = 0.0,
    model_type: str | None = None,
    attention_mask: torch.Tensor | None = None,
    temperature_policy: float = 1.0,
    full_context: bool = False,
    confidences_top_p: int = 1,
    record_trajectory: bool = False,
    setstate_remask_conf_prior: float = 0.0,
) -> GenerationResult:
    """
    If record_trajectory=True, GenerationResult.trajectory is a list of dicts, one per
    denoising step, each with keys: step (int), block (int), x (LongTensor B,L+P clone of
    sequence BEFORE this step's unmask), x0 (LongTensor B,L predicted tokens), unmask
    (BoolTensor B,L positions chosen to commit), confidence (FloatTensor B,L max prob).
    """
    if remasking == "policy":
        if policy is None:
            raise ValueError("policy must be provided for remasking='policy'")
    elif remasking == "fastdllm":
        if thres is None:
            raise ValueError("thres must be provided for remasking='fastdllm'")
    elif remasking in ["low_confidence", "random"]:
        if steps is None:
            raise ValueError(f"steps must be provided for remasking='{remasking}'")
    else:
        raise ValueError(f"Unknown remasking strategy: {remasking}")

    B, prompt_L = prompt.shape
    L = gen_length
    x = torch.full((B, L + prompt_L), mask_id, dtype=torch.long, device=prompt.device)
    x[:, :prompt_L] = prompt
    steps_taken = torch.zeros((B,), dtype=torch.int32, device=x.device)
    num_blocks = L // block_length

    if attention_mask is not None:
        _attn_mask = torch.ones((B, L + prompt_L), dtype=torch.float, device=x.device)
        _attn_mask[:, :prompt_L] = attention_mask.float()
        if model_type == "Dream":
            _attn_mask = _attn_mask.unsqueeze(1).unsqueeze(-2) * _attn_mask.unsqueeze(
                1
            ).unsqueeze(-1)
        # Handle DDP-wrapped models
        model_dtype = model.module.dtype if hasattr(model, "module") else model.dtype
        _attn_mask = _attn_mask.to(model_dtype)
    else:
        _attn_mask = None

    # Strategy-specific state
    record_policy_data = policy is not None
    sampling_history = [] if record_policy_data else None
    trajectory = [] if record_trajectory else None
    global_step = 0

    max_steps = L
    if remasking in ["low_confidence", "random"]:
        assert steps is not None and steps <= L
        tokens_per_step = L // steps
        max_steps = steps

    policy_type = None
    if policy is not None:
        policy_type = (
            policy.module.policy_type
            if hasattr(policy, "module")
            else policy.policy_type
        )

    for num_block in range(num_blocks):
        start_idx = num_block * block_length
        end_idx = start_idx + block_length
        block_slice = slice(start_idx, end_idx)
        block_index = torch.zeros(L, dtype=torch.bool, device=x.device)
        block_index[start_idx:end_idx] = True

        # 3-way allows extra iterations because remasked tokens must be re-predicted.
        # We cap the inner loop at 2*block_length for 3-way to bound total NFE.
        inner_loop_cap = (
            2 * block_length if sampling_mode == "three_way" else block_length
        )
        # Track whether last policy step did anything. Used by the 3-way break
        # condition so the policy always gets one pass over the fully-unmasked
        # state (the only time remasking makes sense).
        last_step_idle = False
        for _ in range(inner_loop_cap):
            generation_part = x[:, prompt_L:]
            mask_index = (generation_part == mask_id) & (
                steps_taken < max_steps
            ).unsqueeze(-1)
            block_mask_index = mask_index[:, block_index]  # (B, BL)

            if sampling_mode == "three_way":
                # Break only when the block is FULLY unmasked AND the policy
                # voted no-op last pass. This mirrors 2-way's "keep going until
                # everything is unmasked" guarantee, while still giving the policy
                # the option to remask a completed block and then decide it's done.
                # Prevents the early-termination failure mode where a KEEP-heavy
                # policy at t≈1.0 would idle-break before any tokens unmasked.
                if (~block_mask_index).all() and last_step_idle:
                    break
            else:
                if (~block_mask_index).all():
                    break

            model_output = model(
                x,
                attention_mask=_attn_mask,
                output_hidden_states=(policy_type == "dit_hidden"),
            )

            # Handle Dream model logit shifting
            # Dream: logits at position i predict token i+1
            # For generated tokens at [P, P+1, ..., P+L-1], we need logits at [P-1, P, ..., P+L-2]
            if model_type == "Dream":
                logits = model_output.logits[
                    :, prompt_L - 1 : -1
                ]  # Include last prompt pos, exclude last gen pos
            else:
                logits = model_output.logits[
                    :, prompt_L:
                ]  # Just slice to generation portion

            # Apply Gumbel noise
            logits_with_noise = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            # Compute softmax once (needed by all strategies)
            probs = F.softmax(logits, dim=-1)

            # Get unmask decisions based on strategy
            remask = None
            if remasking == "policy":
                unmask, remask, sampling_data = _policy_unmask_decisions(
                    mask_index,
                    block_mask_index,
                    probs,
                    x0,
                    steps_taken,
                    block_slice,
                    L,
                    policy,
                    policy_type,
                    sampling_mode,
                    full_context,
                    confidences_top_p,
                    model_output,
                    prompt_L,
                    dpls_stop_logit,
                    temperature_policy,
                    generation_part=generation_part,
                    mask_id=mask_id,
                    setstate_remask_conf_prior=setstate_remask_conf_prior,
                )
                sampling_history.append(sampling_data)

            elif remasking == "fastdllm":
                unmask = _confidence_threshold_unmask(
                    block_mask_index, probs, block_slice, thres
                )
                if policy is not None:
                    sampling_data = _record_policy_data(
                        mask_index,
                        block_mask_index,
                        probs,
                        steps_taken,
                        block_slice,
                        L,
                        policy,
                        policy_type,
                        full_context,
                        confidences_top_p,
                        model_output,
                        prompt_L,
                        temperature_policy,
                        unmask,
                    )
                    sampling_history.append(sampling_data)

            elif remasking in ["low_confidence", "random"]:
                unmask = _fixed_step_unmask_decisions(
                    block_mask_index,
                    probs,
                    x0,
                    block_slice,
                    tokens_per_step,
                    remasking,
                )

            # Capture trajectory BEFORE applying unmask, so x reflects pre-step state
            if record_trajectory:
                trajectory.append({
                    "step": global_step,
                    "block": num_block,
                    "x": x.detach().clone().cpu(),
                    "x0": x0.detach().clone().cpu(),
                    "unmask": unmask.detach().clone().cpu(),
                    "remask": (
                        remask.detach().clone().cpu()
                        if remask is not None
                        else torch.zeros_like(unmask).cpu()
                    ),
                    "confidence": probs.max(dim=-1).values.detach().clone().cpu(),
                })
                global_step += 1

            # Apply unmasking
            new_gen = torch.where(unmask, x0, generation_part)
            # Apply remasking (3-way only): unmasked tokens → mask_id
            step_did_remask = False
            if remask is not None and remask.any():
                new_gen = torch.where(
                    remask, torch.full_like(new_gen, mask_id), new_gen
                )
                step_did_remask = True
            x[:, prompt_L:] = new_gen
            # An "idle" step is one where the policy neither unmasked nor remasked.
            last_step_idle = not (unmask.any() or step_did_remask)

            # Update steps taken: only count steps for batch elements that had work to do
            did_work = block_mask_index.any(dim=-1)
            if remask is not None:
                did_work = did_work | remask[:, block_slice].any(dim=-1)
            steps_taken += did_work.int()

    # Prepare metadata for gradient steps/loss computation
    if record_policy_data:
        generation_part = x[:, prompt_L:]
        still_masked = (generation_part == mask_id).any(dim=-1)

        if sampling_history:
            # Stack all sampling data for training
            sampling_inputs = torch.stack(
                [h["sampling_inputs"] for h in sampling_history], dim=1
            )
            samples = torch.stack([h["samples"] for h in sampling_history], dim=1)
            sampling_masks = torch.stack(
                [h["sampling_masks"] for h in sampling_history], dim=1
            )

            # Stack policy inputs
            policy_input_columns = zip(*[h["policy_inputs"] for h in sampling_history])
            policy_inputs_result = tuple(
                torch.stack(col, dim=1) for col in policy_input_columns
            )
        else:
            sampling_inputs = samples = sampling_masks = None
            policy_inputs_result = None

        return GenerationResult(
            sequences=x,
            steps_taken=steps_taken,
            sampling_inputs=sampling_inputs,
            samples=samples,
            sampling_masks=sampling_masks,
            policy_inputs=policy_inputs_result,
            still_masked=still_masked,
            trajectory=trajectory,
        )
    else:
        return GenerationResult(
            sequences=x,
            steps_taken=steps_taken,
            trajectory=trajectory,
        )


def _get_masks(
    mask_index: torch.Tensor,
    block_mask_index: torch.Tensor,
    block_slice: slice,
    full_context: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    policy_mask = mask_index if full_context else block_mask_index

    if full_context:
        # Policy sees full sequence (B, L), but we only sample in current block
        sampling_mask = torch.zeros_like(mask_index)
        sampling_mask[:, block_slice] = block_mask_index
    else:
        # Policy sees only block (B, BL), sample from same positions
        sampling_mask = policy_mask

    return policy_mask, sampling_mask


def _compute_policy_logits(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
    """Compute policy logits and masks.

    :return: (policy_logits, policy_mask, sampling_mask, policy_inputs)
    """
    per_batch_timestep = steps_taken.unsqueeze(-1) * (1 / L)
    policy_mask, sampling_mask = _get_masks(
        mask_index, block_mask_index, block_slice, full_context
    )

    topk_result = probs.topk(confidences_top_p, dim=-1)
    c_max_input = (
        topk_result.values if full_context else topk_result.values[:, block_slice]
    )

    if policy_type == "dit_hidden":
        hidden_states = model_output.hidden_states[-1]
        hidden_states_input = (
            hidden_states[:, prompt_L:, :]
            if full_context
            else hidden_states[
                :, prompt_L + block_slice.start : prompt_L + block_slice.stop, :
            ]
        )
        policy_inputs = (policy_mask, hidden_states_input, per_batch_timestep)
    elif policy_type == "dit_confidence":
        policy_inputs = (policy_mask, c_max_input, per_batch_timestep)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    policy_logits = policy(*policy_inputs)

    # Apply temperature scaling
    if temperature_policy != 1.0:
        policy_logits = policy_logits / temperature_policy

    return policy_logits, policy_mask, sampling_mask, policy_inputs


def _policy_unmask_decisions(
    mask_index: torch.Tensor,
    block_mask_index: torch.Tensor,
    probs: torch.Tensor,
    x0: torch.Tensor,
    steps_taken: torch.Tensor,
    block_slice: slice,
    L: int,
    policy,
    policy_type: str,
    sampling_mode: str,
    full_context: bool,
    confidences_top_p: int,
    model_output,
    prompt_L: int,
    dpls_stop_logit: float = 0.0,
    temperature_policy: float = 1.0,
    generation_part: torch.Tensor | None = None,
    mask_id: int | None = None,
    setstate_remask_conf_prior: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
    if sampling_mode == "three_way":
        return _policy_three_way_decisions(
            mask_index=mask_index,
            block_mask_index=block_mask_index,
            probs=probs,
            steps_taken=steps_taken,
            block_slice=block_slice,
            L=L,
            policy=policy,
            policy_type=policy_type,
            full_context=full_context,
            confidences_top_p=confidences_top_p,
            model_output=model_output,
            prompt_L=prompt_L,
            temperature_policy=temperature_policy,
            generation_part=generation_part,
            mask_id=mask_id,
        )

    if sampling_mode == "two_way_setstate":
        from common.generation.two_way_setstate import _policy_two_way_setstate_decisions
        return _policy_two_way_setstate_decisions(
            mask_index=mask_index,
            block_mask_index=block_mask_index,
            probs=probs,
            steps_taken=steps_taken,
            block_slice=block_slice,
            L=L,
            policy=policy,
            policy_type=policy_type,
            full_context=full_context,
            confidences_top_p=confidences_top_p,
            model_output=model_output,
            prompt_L=prompt_L,
            temperature_policy=temperature_policy,
            generation_part=generation_part,
            mask_id=mask_id,
            remask_conf_prior=setstate_remask_conf_prior,
        )

    policy_logits, _, sampling_mask, policy_inputs = _compute_policy_logits(
        mask_index,
        block_mask_index,
        probs,
        steps_taken,
        block_slice,
        L,
        policy,
        policy_type,
        full_context,
        confidences_top_p,
        model_output,
        prompt_L,
        temperature_policy,
    )

    # Sample based on mode (using sampling_mask which is gated to current block)
    if sampling_mode == "bernoulli":
        b = bernoulli_sample(utilities=policy_logits, mask_index=sampling_mask)
        samples_for_loglik = b
    elif sampling_mode == "bernoulli-argmax":
        b = bernoulli_sample(utilities=policy_logits, mask_index=sampling_mask)
        # For batch items where nothing was selected, force unmask at argmax
        no_selection = b.sum(dim=-1) == 0
        if no_selection.any():
            masked_logits = policy_logits.clone()
            masked_logits[~sampling_mask] = -torch.inf
            force_idx = torch.argmax(masked_logits, dim=-1)
            batch_indices = torch.arange(b.shape[0], device=b.device)[no_selection]
            b[batch_indices, force_idx[no_selection]] = True
        samples_for_loglik = b
    elif sampling_mode == "dpls":
        dpls_sequences, b = dpls_sample(
            utilities=policy_logits,
            stop_logit=dpls_stop_logit,
            mask_index=sampling_mask,
        )
        samples_for_loglik = dpls_sequences
    else:
        raise ValueError(f"Unknown sampling mode: {sampling_mode}")

    # Convert to sequence-level (always gate to current block)
    unmask = torch.zeros(
        (probs.shape[0], probs.shape[1]), dtype=torch.bool, device=probs.device
    )
    if full_context:
        unmask[:, block_slice] = b[:, block_slice]
    else:
        unmask[:, block_slice] = b

    sampling_data = {
        "sampling_inputs": policy_logits.detach(),
        "samples": samples_for_loglik.detach(),
        "sampling_masks": sampling_mask.detach(),
        "policy_inputs": tuple(
            pi.detach() if isinstance(pi, torch.Tensor) else pi for pi in policy_inputs
        ),
    }

    return unmask, None, sampling_data


def _policy_three_way_decisions(
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
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """3-way (unmask/keep/remask) policy decision path.

    The sampling_mask here spans ALL in-block positions (masked and unmasked) — the policy
    can act on already-revealed tokens too. Invalid (action, position) pairs get -inf
    logits so the categorical distribution is automatically constrained.
    """
    B, L_full = probs.shape[0], probs.shape[1]
    # sampling_mask: all positions in the current block (so the policy can
    # potentially remask a previously unmasked token). We do NOT gate on
    # mask_index here — that's the 2-way convention.
    block_positions = torch.zeros(
        (B, L_full), dtype=torch.bool, device=probs.device
    )
    block_positions[:, block_slice] = True
    # Gate by NFE budget so batch items that are "done" stop acting
    active = (steps_taken < L).unsqueeze(-1)
    sampling_mask = block_positions & active  # (B, L)

    # Compute policy logits. Note: for 3-way, policy output is (*B, L_policy, 3)
    # where L_policy is L if full_context else BL.
    per_batch_timestep = steps_taken.unsqueeze(-1) * (1 / L)
    topk_result = probs.topk(confidences_top_p, dim=-1)
    if full_context:
        c_max_input = topk_result.values
        policy_mask = mask_index  # policy's "what-is-masked" input sees full sequence
    else:
        c_max_input = topk_result.values[:, block_slice]
        policy_mask = block_mask_index

    if policy_type == "dit_hidden":
        hidden_states = model_output.hidden_states[-1]
        hidden_input = (
            hidden_states[:, prompt_L:, :]
            if full_context
            else hidden_states[
                :, prompt_L + block_slice.start : prompt_L + block_slice.stop, :
            ]
        )
        policy_inputs = (policy_mask, hidden_input, per_batch_timestep)
    elif policy_type == "dit_confidence":
        policy_inputs = (policy_mask, c_max_input, per_batch_timestep)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    policy_logits = policy(*policy_inputs)  # (*B, L_policy, 3)
    if temperature_policy != 1.0:
        policy_logits = policy_logits / temperature_policy

    assert policy_logits.shape[-1] == 3, (
        f"Three-way sampling expects policy output with last dim 3, got {policy_logits.shape}"
    )

    # Promote block-only policy output to full-L tensor so downstream tensors
    # always have shape (B, L, 3).
    if not full_context:
        full_logits = torch.full(
            (B, L_full, 3), float("-inf"), dtype=policy_logits.dtype,
            device=policy_logits.device,
        )
        full_logits[:, block_slice, :] = policy_logits
        policy_logits_full = full_logits
    else:
        policy_logits_full = policy_logits

    # Build per-position action validity mask:
    #   - Masked positions: {UNMASK, KEEP} valid, REMASK invalid
    #   - Unmasked positions: {REMASK, KEEP} valid, UNMASK invalid
    #   - Outside sampling_mask: force KEEP
    is_masked = (generation_part == mask_id)  # (B, L)
    # Clone to avoid mutating caller tensors
    constrained_logits = policy_logits_full.clone()

    # Confidence-aware REMASK prior: suppress remasking of tokens that the base
    # model is confident about. At an unmasked position, we add -REMASK_CONF_PRIOR
    # * (top-1 confidence) to the REMASK logit. A token with conf=1.0 gets its
    # remask logit shifted down by REMASK_CONF_PRIOR; a token with conf=0 is
    # unaffected. This is a structural prior that encodes the assumption
    # "high base-model confidence => token is likely correct => don't throw it
    # away." The policy can still learn to override this prior via its own
    # features; this just stops a randomly-initialized remask head from
    # indiscriminately destroying coherent outputs before reward signal arrives.
    REMASK_CONF_PRIOR = 5.0
    conf_max_full = probs.max(dim=-1).values  # (B, L_full) model top-1 confidence
    constrained_logits[..., ACTION_REMASK] = (
        constrained_logits[..., ACTION_REMASK] - REMASK_CONF_PRIOR * conf_max_full
    )
    # At masked positions, REMASK invalid
    constrained_logits[..., ACTION_REMASK] = torch.where(
        is_masked, torch.full_like(constrained_logits[..., ACTION_REMASK], float("-inf")),
        constrained_logits[..., ACTION_REMASK],
    )
    # At unmasked positions, UNMASK invalid
    constrained_logits[..., ACTION_UNMASK] = torch.where(
        ~is_masked, torch.full_like(constrained_logits[..., ACTION_UNMASK], float("-inf")),
        constrained_logits[..., ACTION_UNMASK],
    )
    # Outside sampling_mask: force deterministic KEEP
    outside = ~sampling_mask
    constrained_logits[..., ACTION_UNMASK] = torch.where(
        outside, torch.full_like(constrained_logits[..., ACTION_UNMASK], float("-inf")),
        constrained_logits[..., ACTION_UNMASK],
    )
    constrained_logits[..., ACTION_REMASK] = torch.where(
        outside, torch.full_like(constrained_logits[..., ACTION_REMASK], float("-inf")),
        constrained_logits[..., ACTION_REMASK],
    )

    # Eval-only knob: forces the policy into 2-way behavior by zeroing the
    # REMASK action's probability mass. Set DISABLE_REMASK=1 in the env to
    # enable. Used to test whether a learned 3-way policy actually exploits
    # REMASK or whether remasking is ornamental at inference time.
    if os.environ.get("DISABLE_REMASK", "0") == "1":
        constrained_logits[..., ACTION_REMASK] = float("-inf")

    actions = categorical_sample(constrained_logits, mask_index=sampling_mask)
    unmask = (actions == ACTION_UNMASK) & sampling_mask
    remask = (actions == ACTION_REMASK) & sampling_mask

    sampling_data = {
        "sampling_inputs": constrained_logits.detach(),  # (B, L, 3)
        "samples": actions.detach(),  # (B, L) int64
        "sampling_masks": sampling_mask.detach(),  # (B, L) bool
        "policy_inputs": tuple(
            pi.detach() if isinstance(pi, torch.Tensor) else pi for pi in policy_inputs
        ),
    }
    return unmask, remask, sampling_data


def _confidence_threshold_unmask(
    block_mask_index: torch.Tensor,
    probs: torch.Tensor,
    block_slice: slice,
    thres: float | torch.Tensor,
) -> torch.Tensor:
    confidence = probs.max(dim=-1).values

    # Only consider masked positions in current block
    confidence_masked = confidence[:, block_slice].clone()
    confidence_masked[~block_mask_index] = -torch.inf

    unmask_local = confidence_masked > thres
    if not unmask_local.any():
        force_idx = torch.argmax(confidence_masked, dim=-1)
        unmask_local.scatter_(1, force_idx.unsqueeze(-1), True)

    unmask = torch.zeros(
        (probs.shape[0], probs.shape[1]), dtype=torch.bool, device=probs.device
    )
    unmask[:, block_slice] = unmask_local
    return unmask


def _record_policy_data(
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
    unmask: torch.Tensor,
) -> dict:
    policy_logits, policy_mask, _, policy_inputs = _compute_policy_logits(
        mask_index,
        block_mask_index,
        probs,
        steps_taken,
        block_slice,
        L,
        policy,
        policy_type,
        full_context,
        confidences_top_p,
        model_output,
        prompt_L,
        temperature_policy,
    )

    samples = unmask if full_context else unmask[:, block_slice]

    # ES (Expert Steering) special behavior: save policy_mask (not sampling_mask) so the
    # model learns to mimic the confidence thresholding in a block-agnostic way
    return {
        "sampling_inputs": policy_logits.detach().clone(),
        "samples": samples.detach().clone(),
        "sampling_masks": policy_mask.detach().clone(),
        "policy_inputs": tuple(
            pi.detach().clone() if isinstance(pi, torch.Tensor) else pi
            for pi in policy_inputs
        ),
    }


def _fixed_step_unmask_decisions(
    block_mask_index: torch.Tensor,
    probs: torch.Tensor,
    x0: torch.Tensor,
    block_slice: slice,
    tokens_per_step: int,
    mode: str,
) -> torch.Tensor:
    B = block_mask_index.shape[0]
    block_size = block_slice.stop - block_slice.start

    if mode == "low_confidence":
        confidence_block = torch.gather(
            probs[:, block_slice], dim=-1, index=x0[:, block_slice].unsqueeze(-1)
        ).squeeze(-1)
    elif mode == "random":
        confidence_block = torch.rand((B, block_size), device=x0.device)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    confidence_masked = torch.where(block_mask_index, confidence_block, -torch.inf)

    num_masked_block = block_mask_index.sum(dim=-1)
    k = torch.clamp(num_masked_block, max=tokens_per_step)
    max_k = k.max().item()

    if max_k == 0:
        unmask = torch.zeros(
            (probs.shape[0], probs.shape[1]), dtype=torch.bool, device=probs.device
        )
        return unmask

    _, topk_indices = torch.topk(confidence_masked, k=max_k, dim=-1)

    positions = torch.arange(max_k, device=x0.device).unsqueeze(0).expand(B, -1)
    valid_mask = positions < k.unsqueeze(-1)

    unmask_local = torch.zeros_like(block_mask_index, dtype=torch.bool)
    unmask_local.scatter_(1, topk_indices, valid_mask)

    unmask = torch.zeros(
        (probs.shape[0], probs.shape[1]), dtype=torch.bool, device=probs.device
    )
    unmask[:, block_slice] = unmask_local
    return unmask
