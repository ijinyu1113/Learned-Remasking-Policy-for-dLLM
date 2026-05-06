# pretraining utils for the remasking policy
# generates synthetic corrupted sequences and computes per-token rewards

import torch
import torch.nn.functional as F
from typing import NamedTuple, Optional


class PretrainBatch(NamedTuple):
    """Container for pretraining batch data"""
    sequences: torch.Tensor  # (B, L) original reference completions
    corrupted_sequences: torch.Tensor  # (B, L) sequences with some tokens replaced
    corrupted_mask: torch.BoolTensor  # (B, L) which positions are corrupted
    masked_mask: torch.BoolTensor  # (B, L) which positions are masked
    confidences: torch.Tensor  # (B, L) per-token confidence from dLLM
    unmask_reward: torch.Tensor  # (B, L) reward for unmasking each position
    remask_reward: torch.Tensor  # (B, L) reward for remasking corrupted tokens
    probs: torch.Tensor  # (B, L, V) full probability distribution
    p_current_token: Optional[torch.Tensor]  # (B, L) prob of current token (used only by 2-way)


def corrupt_and_mask_sequences(
    sequences: torch.Tensor,
    dllm,
    mask_id: int,
    mask_prob: float = 0.3,
    corrupt_prob: float = 0.3,
    exclusion_k: int = 10,
    top_p: float = 0.90,
    temperature: float = 1.0,
    confidence_threshold: float = 0.5,
    unmask_reward_value: float = 1.0,
    remask_reward_value: float = 1.0,
    num_actions: int = 3,
) -> PretrainBatch:
    """
    create corrupted/masked sequences for pretraining.

    Args:
        sequences: (B, L) token IDs from reference completions
        dllm: frozen diffusion LLM
        mask_id: token ID for [MASK]
        mask_prob: probability to mask each position
        corrupt_prob: probability to corrupt each position
        temperature: temperature for sampling from 1-p(x)
        confidence_threshold: threshold for high-confidence unmask reward (e.g., 0.9)
        unmask_reward_value: reward for unmasking high-confidence tokens
        remask_reward_value: reward for remasking corrupted tokens
        num_actions: 1: 2-way set-state, 3: 3-way

    Returns:
        PretrainBatch with sequences, masks, and confidences
    """
    assert num_actions in [1, 3], f"num_actions must be 1 (2-way) or 3 (3-way), got {num_actions}"
    B, L = sequences.shape
    device = sequences.device

    # TODO: should corruption sampling be unform across all tokens, then disjoint with masked, or uniform across all non-masked?
    # currently then probability that any token gets corrupted is (1-mask_prob)*corrupt_prob
    mask_decisions = torch.rand(B, L, device=device) < mask_prob # mask each token with probability mask_prob
    corrupt_decisions = (torch.rand(B, L, device=device) < corrupt_prob) & (~mask_decisions) # attempt to currupt tokens w/ corrupt_prob, ignore if token is masked

    # create corrupted sequence
    corrupted_sequences = sequences.clone()
    corrupted_sequences[mask_decisions] = mask_id

    # sample replacement tokens
    with torch.no_grad():
        # get model logits
        output = dllm(corrupted_sequences)
        logits = output.logits
        probs = F.softmax(logits, dim=-1)

        # sample from top_p(exclude_top_k(p(x)))
        if corrupt_decisions.any():
            corrupt_indices = torch.where(corrupt_decisions)

            for b, i in zip(corrupt_indices[0], corrupt_indices[1]):
                original_token = sequences[b, i]

                # create a distribution excluding original token
                corrupt_logits = probs[b, i].clone()
                corrupt_logits[original_token] = float("-inf")

                # exclude top k
                if exclusion_k > 0:
                    _, indices_to_exclude = torch.topk(corrupt_logits, exclusion_k)
                    corrupt_logits[indices_to_exclude] = float('-inf')

                corrupt_logits = corrupt_logits / temperature
                
                # keep tokens in top p of probability mass
                sorted_logits, sorted_indices = torch.sort(corrupt_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                corrupt_logits[indices_to_remove] = float('-inf')

                corrupt_probs = F.softmax(corrupt_logits, dim=-1)

                sampled_token = torch.multinomial(corrupt_probs, num_samples=1).item()
                corrupted_sequences[b, i] = sampled_token

        # get per-position confidences
        max_probs = probs.max(dim=-1)[0]  # (B, L)
        current_token_probs = probs.gather(-1, corrupted_sequences.unsqueeze(-1)).squeeze(-1)  # (B, L)

        # confidences: max prob for masked, token prob for unmasked
        confidences = torch.where(mask_decisions, max_probs, current_token_probs)  # (B, L)

        # for 2-way: p(current_token) is 0 for masked, token prob for unmasked
        if num_actions == 1:
            p_current_token = torch.where(mask_decisions, torch.zeros_like(max_probs), current_token_probs)
        else:
            p_current_token = None

    # compute per-token rewards
    unmask_reward = torch.zeros(B, L, device=device)
    remask_reward = torch.zeros(B, L, device=device)

    # reward for unmasking any tokens above confidence theshold
    unmask_reward[mask_decisions] = (
        (confidences[mask_decisions] > confidence_threshold).float() * unmask_reward_value
    )

    # reward for remasking corrupted tokens with low confidence
    remask_reward[corrupt_decisions] = remask_reward_value

    return PretrainBatch(
        sequences=sequences,
        corrupted_sequences=corrupted_sequences,
        corrupted_mask=corrupt_decisions,
        masked_mask=mask_decisions,
        confidences=confidences,
        unmask_reward=unmask_reward,
        remask_reward=remask_reward,
        probs=probs,
        p_current_token=p_current_token,
    )


def compute_pretrain_loss(
    policy_logits: torch.Tensor,
    masked_mask: torch.BoolTensor,
    corrupted_mask: torch.BoolTensor,
    unmask_reward: torch.Tensor,
    remask_reward: torch.Tensor,
    num_actions: int = 3,
) -> torch.Tensor:
    """
    compute supervised learning loss for pretraining.

    For 3-way: outputs (B, L, 3) logits (keep=0, unmask=1, remask=2).
    For 2-way: outputs (B, L) or (B, L, 1) Bernoulli logits where sigmoid(logit) = P(set state to UNMASKED).

    both policies train on the same synthetic task: unmask high-confidence tokens, remask corrupted tokens.
    loss adapts the target representation to each policy's action space.

    Args:
        policy_logits: (B, L, num_actions) logits from policy
        masked_mask: (B, L) which positions are masked
        corrupted_mask: (B, L) which positions are corrupted
        unmask_reward: (B, L) reward for unmasking
        remask_reward: (B, L) reward for remasking
        num_actions: 1:2-way set-state, 3:3-way

    Returns:
        Scalar loss (mean across batch and sequence)
    """
    B, L = masked_mask.shape
    device = policy_logits.device

    if num_actions == 3:
        return _compute_3way_loss(policy_logits, masked_mask, corrupted_mask, unmask_reward, remask_reward)
    elif num_actions == 1:
        return _compute_2way_loss(policy_logits, masked_mask, corrupted_mask, unmask_reward, remask_reward)
    else:
        raise ValueError(f"num_actions must be 1 or 3, got {num_actions}")


def _compute_3way_loss(
    policy_logits: torch.Tensor,
    masked_mask: torch.BoolTensor,
    corrupted_mask: torch.BoolTensor,
    unmask_reward: torch.Tensor,
    remask_reward: torch.Tensor,
) -> torch.Tensor:
    """Compute loss for 3-way policy."""
    B, L, A = policy_logits.shape
    device = policy_logits.device
    assert A == 3, f"Expected 3 actions for 3-way, got {A}"

    # create target labels and reward weights
    targets = torch.zeros(B, L, dtype=torch.long, device=device)
    weights = torch.zeros(B, L, device=device)

    # for masked positions: unmask if unmask_reward > 0, else keep
    masked_idx = masked_mask.nonzero(as_tuple=True)
    targets[masked_idx] = (unmask_reward[masked_idx] > 0).long()  # 0=keep, 1=unmask
    weights[masked_idx] = unmask_reward[masked_idx].abs().clamp(min=0.1)

    # for corrupted positions: remask action
    corrupted_idx = corrupted_mask.nonzero(as_tuple=True)
    targets[corrupted_idx] = 2  # remask action
    weights[corrupted_idx] = remask_reward[corrupted_idx].abs().clamp(min=0.1)

    # for correct unmasked positions: keep action (target=0) to prevent over-remasking
    correct_unmasked = ~(masked_mask | corrupted_mask)
    targets[correct_unmasked] = 0  # keep action
    weights[correct_unmasked] = 0.15  # regularize keep behavior

    # compute weighted cross-entropy
    flat_logits = policy_logits.view(-1, A)  # (B*L, 3)
    flat_targets = targets.view(-1)  # (B*L,)
    flat_weights = weights.view(-1)  # (B*L,)

    ce_loss = F.cross_entropy(flat_logits, flat_targets, reduction='none')
    weighted_loss = (ce_loss * flat_weights).sum() / flat_weights.sum().clamp(min=1e-8)

    return weighted_loss


def compute_pretrain_metrics(
    policy_logits: torch.Tensor,
    masked_mask: torch.BoolTensor,
    confidences: torch.Tensor,
    num_actions: int = 3,
) -> dict:
    """compute pretraining metrics based on policy decisions.

    Args:
        policy_logits: (B, L, num_actions) or (B, L) logits from policy
        masked_mask: (B, L) which positions are masked
        confidences: (B, L) per-token confidence scores
        num_actions: 1 (2-way) or 3 (3-way policy)

    Returns:
        dict with keys for action rates and mean confidences per action
    """
    B, L = masked_mask.shape
    device = policy_logits.device
    is_masked = masked_mask
    is_unmasked = ~masked_mask

    def _metrics_from_mask(keep_action, unmask_action, remask_action, total_positions):
        keep_rate = keep_action.float().sum().item() / total_positions
        unmask_rate = unmask_action.float().sum().item() / total_positions
        remask_rate = remask_action.float().sum().item() / total_positions

        # mean confidences at positions where policy took each action
        keep_conf_mean = confidences[keep_action].mean().item() if keep_action.any() else 0.0
        unmask_conf_mean = confidences[unmask_action].mean().item() if unmask_action.any() else 0.0
        remask_conf_mean = confidences[remask_action].mean().item() if remask_action.any() else 0.0

        return {
            "keep_rate": keep_rate,
            "unmask_rate": unmask_rate,
            "remask_rate": remask_rate,
            "keep_conf_mean": keep_conf_mean,
            "unmask_conf_mean": unmask_conf_mean,
            "remask_conf_mean": remask_conf_mean,
        }

    if num_actions == 3:
        # get argmax action (0=keep, 1=unmask, 2=remask)
        actions = policy_logits.argmax(dim=-1)  # (B, L)

        keep_mask = (actions == 0)
        unmask_mask = (actions == 1)
        remask_mask = (actions == 2)

        return _metrics_from_mask(keep_mask, unmask_mask, remask_mask, B*L)

    elif num_actions == 1:
        # 2-way set-state: Bernoulli logits
        if policy_logits.dim() == 3:
            logits = policy_logits.squeeze(-1)  # (B, L)
        else:
            logits = policy_logits  # (B, L)

        # logit > 0 means set_unmasked (sigmoid > 0.5)
        set_unmasked = (logits > 0).bool()  # (B, L)

        # action effects depend on current state:
        # masked & set_unmasked → UNMASK (reveal token)
        # unmasked & ~set_unmasked → REMASK (hide token)
        # otherwise → KEEP (no state change)
        unmask_action = is_masked & set_unmasked
        remask_action = is_unmasked & ~set_unmasked
        keep_action = ~(unmask_action | remask_action)

        return _metrics_from_mask(keep_action, unmask_action, remask_action, B*L)

    else:
        raise ValueError(f"num_actions must be 1 or 3, got {num_actions}")


def _compute_2way_loss(
    policy_logits: torch.Tensor,
    masked_mask: torch.BoolTensor,
    corrupted_mask: torch.BoolTensor,
    unmask_reward: torch.Tensor,
    remask_reward: torch.Tensor,
) -> torch.Tensor:
    """
    compute loss for 2-way policy (Bernoulli action).

    targets: 0 = set-masked, 1 = set-unmasked.
    """
    B, L = masked_mask.shape
    device = policy_logits.device

    # flatten logits: (B*L,) or (B*L, 1)
    if policy_logits.dim() == 3:
        # (B, L, 1) -> (B*L,)
        flat_logits = policy_logits.squeeze(-1).view(-1)
    else:
        # (B, L) -> (B*L,)
        flat_logits = policy_logits.view(-1)

    # create binary targets and weights
    targets = torch.zeros(B, L, dtype=torch.float, device=device)
    weights = torch.zeros(B, L, device=device)

    # for masked positions: set-unmasked (target=1) if high reward, else set-masked (target=0)
    masked_idx = masked_mask.nonzero(as_tuple=True)
    targets[masked_idx] = (unmask_reward[masked_idx] > 0).float()
    weights[masked_idx] = unmask_reward[masked_idx].abs().clamp(min=0.1)

    # for corrupted positions: set-masked (target=0)
    corrupted_idx = corrupted_mask.nonzero(as_tuple=True)
    targets[corrupted_idx] = 0.0  # set-masked
    weights[corrupted_idx] = remask_reward[corrupted_idx].abs().clamp(min=0.1)

    # for correct unmasked positions: set-unmasked (target=1) to prevent over-remasking
    correct_unmasked = ~(masked_mask | corrupted_mask)
    targets[correct_unmasked] = 1.0
    weights[correct_unmasked] = 0.15  # regularize keep behavior

    # compute weighted binary cross-entropy
    flat_targets = targets.view(-1)
    flat_weights = weights.view(-1)

    bce_loss = F.binary_cross_entropy_with_logits(flat_logits, flat_targets, reduction='none')
    weighted_loss = (bce_loss * flat_weights).sum() / flat_weights.sum().clamp(min=1e-8)

    return weighted_loss
