#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
### Adapted from https://github.com/dllm-reasoning/d1 (Apache 2.0)
import gc
import json
import os
import warnings
from collections import deque
from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

import numpy as np
import torch
import wandb
from accelerate.utils import gather
from accelerate.utils import gather_object
from datasets import Dataset
from datasets import IterableDataset
from torch import nn
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerBase
from transformers import Trainer as HFTrainer
from transformers import TrainerCallback
from trl.data_utils import is_conversational
from trl.data_utils import maybe_apply_chat_template
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.utils import print_prompt_completions_sample

from common.generation.generation import generate_unified
from common.generation.sampling import ACTION_REMASK
from common.generation.sampling import ACTION_UNMASK
from common.generation.sampling import bernoulli_batch_loglik
from common.generation.sampling import categorical_batch_loglik
from common.generation.sampling import categorical_entropy
from common.generation.sampling import dpls_batch_loglik
from common.s3 import S3UploadCallback

try:
    import rich  # noqa: F401

    _rich_available = True
except ImportError:
    _rich_available = False

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class Trainer(GRPOTrainer):
    def __init__(
        self,
        model,
        dllm: nn.Module,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (
            None,
            None,
        ),
    ):
        # Initialize the parent class
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=None,
        )
        torch._dynamo.config.capture_scalar_outputs = True
        self.dllm = torch.compile(dllm)
        self.dllm.eval()

        # Initialize buffering for multi-iteration training
        self._buffered_inputs = None
        self._step = 0

        if self.args.remasking == "policy":
            assert self.beta == 0.0, "Beta must be 0.0 for policy-based remasking"

        # Gradient accumulation not supported with current buffering logic
        assert self.args.gradient_accumulation_steps == 1, (
            "gradient_accumulation_steps must be 1 (current buffering does not support gradient accumulation)"
        )

        # Track recent rewards for best checkpoint saving
        self.train_reward_queue = deque(
            maxlen=10 * self.args.gradient_accumulation_steps
        )
        self.train_reward_best = -float("inf")
        self.train_reward_best_step = 0
        self.effective_steps = 0
        self.s3_callback = None
        for callback in callbacks:
            if isinstance(callback, S3UploadCallback):
                self.s3_callback = callback
                break

    def train(self, *args, **kwargs):
        """Override train to save final checkpoint at end of training."""
        output = super().train(*args, **kwargs)

        if self.accelerator.is_main_process:
            final_step = self.state.global_step
            checkpoint_dir = os.path.join(
                self.args.output_dir, f"checkpoint-{final_step}"
            )

            print(f"\nSaving final checkpoint at step {final_step}")
            unwrapped_model = self.accelerator.unwrap_model(self.model_wrapped)
            unwrapped_model.save_pretrained(checkpoint_dir)
            self.state.save_to_json(os.path.join(checkpoint_dir, "trainer_state.json"))

            if self.s3_callback is not None:
                print(
                    f"Uploading checkpoint-{final_step} to s3: {self.args.output_dir}"
                )
                self.s3_callback.on_save(self.args, self.state, self.control)

        return output

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        """Override training_step to skip optimizer step when advantages are zero."""
        # Check if all advantages are zero (no learning signal)
        if "advantages" in inputs and torch.abs(inputs["advantages"]).max() < 1e-6:
            # Skip expensive forward/backward passes - no learning signal
            return torch.tensor(0.0, device=inputs["advantages"].device)

        # Track effective training steps (non-zero advantages)
        self.effective_steps += 1

        # Normal training step for non-zero advantages
        return super().training_step(model, inputs, num_items_in_batch)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        model.train()

        # Note that the output of the policy
        # is a list containing pointers to tensors of size  (B, T, BL)
        # although if you sum the B dim you get the group size G, these can
        # in general not be stacked because the T
        # can vary between different batches within the group.
        # Therefore, we process it as a list of batches for as long as necessary.
        policy_outputs: list[dict[str, torch.Tensor | tuple[torch.Tensor]]] = inputs[
            "policy_outputs"
        ]

        # ensure the group batches add up to the whole group size (ie sum_i B_i = G)
        group_batch_sizes = [
            policy_output["sampling_masks"].size(0) for policy_output in policy_outputs
        ]
        assert sum(group_batch_sizes) == inputs["advantages"].size(0)

        # Check if ES (Expert Steering) is enabled and compute mixture distribution weights
        has_es = (
            self.args.es_thresholds is not None and len(self.args.es_thresholds) > 0
        )
        if has_es:
            num_es = len(self.args.es_thresholds)
            total_group_size = inputs["advantages"].size(0)
            num_regular = total_group_size - num_es
            device = inputs["advantages"].device
            # Mixture weights: (G/(G+E)) * pi_theta + (1/(G+E)) * dirac
            # Note that we assume a sample can only come from one particular ES dirac
            # (hence why the second factor is not E/(G+E))
            log_weight_theta = torch.log(
                torch.tensor(num_regular / total_group_size, device=device)
            )
            log_weight_dirac = torch.log(
                torch.tensor(1.0 / total_group_size, device=device)
            )

        # Accumulate the loss over the batches in the group
        batch_index_start = 0
        loss_acummulator = 0
        entropy_accumulator = []
        for batch_idx, batch_policy_output in enumerate(policy_outputs):
            batch_sampling_masks = batch_policy_output[
                "sampling_masks"
            ]  # will need these later

            batch_index_end = batch_index_start + batch_sampling_masks.size(0)

            # Detect if this is an ES batch (last batch in the group)
            is_es_batch = has_es and batch_idx == len(policy_outputs) - 1

            B, T, _ = batch_sampling_masks.shape

            time_step_loss_accumulator = torch.zeros(
                B, device=batch_sampling_masks.device
            )  # (B,)
            timestep_bs = (
                self.args.timestep_batch_size
                if self.args.timestep_batch_size is not None
                else T
            )
            for time_step_idx in range(0, T, timestep_bs):
                # Get this batch's data
                time_step_batch_sampling_masks = batch_sampling_masks[
                    :, time_step_idx : time_step_idx + timestep_bs, :
                ]
                time_step_batch_samples = batch_policy_output["samples"][
                    :, time_step_idx : time_step_idx + timestep_bs, :
                ]

                ### Prepare time-batched inputs
                time_step_batch_policy_inputs = []
                for ptdi in batch_policy_output["policy_inputs"]:
                    if isinstance(ptdi, torch.Tensor):
                        # Find time dim and slice only at that dim
                        time_dim = 1
                        assert ptdi.size(time_dim) == T, (
                            f"ptdi of shape {ptdi.shape=} did not match {T=} "
                            f"at expected {time_dim=}"
                        )
                        slices = [slice(None)] * ptdi.ndim  # by default get everything
                        slices[time_dim] = slice(
                            time_step_idx, time_step_idx + timestep_bs
                        )  # at time_dim get only what belongs to this time-batch

                        time_step_batch_policy_inputs.append(ptdi[tuple(slices)])
                    else:
                        # Non-tensors just get propagated
                        time_step_batch_policy_inputs.append(ptdi)

                logps_timestep = self._get_per_timestep_logps_block(
                    model=model,
                    samples=time_step_batch_samples,
                    sampling_masks=time_step_batch_sampling_masks,
                    policy_inputs=time_step_batch_policy_inputs,
                    sampling_mode=self.args.sampling_mode,
                    return_entropy=True,
                )  # (B, timestep_bs), entropy (scalar)

                if isinstance(logps_timestep, tuple):
                    logps_timestep, entropy = logps_timestep
                    entropy_accumulator.append(entropy)
                else:
                    # Backward compatibility if return_entropy=False
                    pass

                # For ES batches, adjust NEW log probabilities with mixture distribution
                if is_es_batch:
                    logps_timestep = torch.logaddexp(
                        log_weight_theta + logps_timestep, log_weight_dirac
                    )

                # Get old log probabilities
                old_logps_slice = batch_policy_output["old_per_timestep_logps"][
                    :, time_step_idx : time_step_idx + timestep_bs
                ].detach()

                # For ES batches, adjust OLD log probabilities with mixture distribution
                if is_es_batch:
                    old_logps_slice = torch.logaddexp(
                        log_weight_theta + old_logps_slice, log_weight_dirac
                    )

                coeff_1 = torch.exp(
                    logps_timestep - old_logps_slice
                )  # (B, timestep_bs)
                coeff_2 = torch.clamp(
                    coeff_1, 1 - self.args.epsilon, 1 + self.args.epsilon
                )

                # Get the advantages corresponding to this batch.
                # Note that the "advantages" are flat of shape (G,),
                # so we do some indexing to keep track of where the
                # current batch is.
                batch_advantages = (
                    inputs["advantages"][batch_index_start:batch_index_end]
                    .detach()
                    .view((-1,) + (1,) * (coeff_1.ndim - 1))
                )  # (B, 1)

                per_timestep_loss1 = coeff_1 * batch_advantages
                per_timestep_loss2 = coeff_2 * batch_advantages
                per_timestep_loss = torch.min(per_timestep_loss1, per_timestep_loss2)

                # Only include the loss for the active timesteps
                per_timestep_loss *= time_step_batch_sampling_masks.any(dim=-1).to(
                    per_timestep_loss.dtype
                )
                time_step_loss_accumulator += per_timestep_loss.sum(dim=-1)

                del (
                    logps_timestep,
                    coeff_1,
                    coeff_2,
                    per_timestep_loss1,
                    per_timestep_loss2,
                    per_timestep_loss,
                )
                torch.cuda.empty_cache()

            num_active_steps = batch_sampling_masks.any(dim=-1).sum(dim=-1)  # (B,)
            assert (num_active_steps > 0).all(), (
                "At least one batch element was active for < 1 steps?"
            )
            batch_loss = time_step_loss_accumulator / num_active_steps

            # The accumulated loss is updated per the sum of -batch_loss
            # (we later turn the sum into a mean by dividing by the group size)
            loss_acummulator -= batch_loss.sum()

            # Next batch starts where the current one left off
            batch_index_start = batch_index_end

        assert self.beta == 0.0, (
            f"TODO non-zero {self.beta=} not supported at this time"
        )

        # final loss is average over the group
        loss = loss_acummulator / sum(group_batch_sizes)

        # Log entropy if collected; optionally add entropy bonus to the loss.
        if entropy_accumulator:
            mean_entropy = torch.stack(entropy_accumulator).mean()
            self._metrics["train"]["entropy"].append(
                self.accelerator.gather_for_metrics(mean_entropy).mean().item()
            )
            entropy_coef = getattr(self.args, "entropy_coef", 0.0)
            if entropy_coef and entropy_coef > 0.0:
                # Subtract because loss is minimized but entropy should be maximized.
                loss = loss - entropy_coef * mean_entropy

        return loss

    def _get_per_timestep_logps_block(
        self,
        model,
        samples,
        sampling_masks,
        policy_inputs,
        sampling_mode="bernoulli",
        return_entropy=False,
    ):
        """Compute log-probabilities for sampled actions under the policy.

        :param model: policy model
        :param samples: sampled actions
        :param sampling_masks: mask indicating valid positions
        :param policy_inputs: inputs to policy model
        :param sampling_mode: sampling mode ("bernoulli", "dpls")
        :param return_entropy: whether to compute and return entropy
        :return: (log_probs, entropy) or just log_probs if return_entropy=False
        """
        with torch.amp.autocast("cuda", enabled=self.args.fp16):
            logits = model(*policy_inputs)  # (B, T, BL)

        # For 3-way, re-apply the SAME constraints used at sampling time so that
        # the new log-prob is computed on the same distribution as the stored old
        # log-prob. Without this, the PPO importance ratio is biased and gradients
        # point in the wrong direction.
        if sampling_mode == "three_way":
            constrained_logits = self._apply_three_way_constraints(
                logits, policy_inputs, sampling_masks
            )
        elif sampling_mode == "two_way_setstate":
            constrained_logits = self._apply_two_way_setstate_constraints(
                logits, policy_inputs, sampling_masks
            )
        else:
            constrained_logits = logits

        # Calculate corresponding log-likelihoods under the model
        if sampling_mode == "dpls":
            lls = dpls_batch_loglik(
                samples=samples,
                utilities=constrained_logits,
                stop_logit=self.args.dpls_stop_logit,
                mask_index=sampling_masks,
                dtype=self.args.loglikelihood_dtype,
            )
        elif sampling_mode in ["bernoulli", "bernoulli-argmax", "two_way_setstate"]:
            lls = bernoulli_batch_loglik(
                samples,
                constrained_logits,
                mask_index=sampling_masks,
                dtype=self.args.loglikelihood_dtype,
            )
        elif sampling_mode == "three_way":
            lls = categorical_batch_loglik(
                samples=samples,
                utilities=constrained_logits,
                mask_index=sampling_masks,
                dtype=self.args.loglikelihood_dtype,
            )
        else:
            raise ValueError(f"Unexpected {sampling_mode=}")

        # Compute entropy if requested
        entropy = None
        if return_entropy:
            if sampling_mode in ["bernoulli", "bernoulli-argmax", "two_way_setstate"]:
                # For Bernoulli: H = -p*log(p) - (1-p)*log(1-p)
                probs_clamped = torch.sigmoid(constrained_logits).clamp(1e-8, 1 - 1e-8)
                entropy = -(
                    probs_clamped * torch.log(probs_clamped)
                    + (1 - probs_clamped) * torch.log(1 - probs_clamped)
                )  # (B, timestep_bs, BL)
                # Average over active positions only
                active_mask = sampling_masks.float()
                entropy = (entropy * active_mask).sum() / active_mask.sum()
            elif sampling_mode == "dpls":
                masked_logits = constrained_logits.masked_fill(~sampling_masks, float("-inf"))
                probs = torch.softmax(masked_logits, dim=-1)
                probs_clamped = probs.clamp(1e-8, 1.0)
                entropy = -(probs * torch.log(probs_clamped)).sum(dim=-1)
                active_mask = sampling_masks.any(dim=-1).float()
                entropy = (entropy * active_mask).sum() / active_mask.sum()
            elif sampling_mode == "three_way":
                entropy = categorical_entropy(constrained_logits, mask_index=sampling_masks)

        if constrained_logits is not logits:
            del constrained_logits
        del logits
        torch.cuda.empty_cache()

        if return_entropy:
            return lls, entropy
        return lls

    def _apply_three_way_constraints(
        self,
        logits: torch.Tensor,
        policy_inputs,
        sampling_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct the constrained 3-way logits used at sampling time.

        Mirrors the constraint construction in
        ``common.generation.generation._policy_three_way_decisions``:
          1. Subtract the confidence-aware REMASK prior (-5 * conf_top1).
          2. Force REMASK invalid (-inf) at masked positions.
          3. Force UNMASK invalid (-inf) at unmasked positions.
          4. Force UNMASK and REMASK invalid (-inf) outside the sampling mask.

        These constraints depend only on the rollout state, NOT on the policy
        parameters, so they have zero gradient. But they DO affect the softmax
        denominator used by `categorical_batch_loglik`, so they must be applied
        consistently between sampling-time (old log-prob) and training-time
        (new log-prob) for the PPO importance ratio to be unbiased.

        :param logits: (B, T, L, 3) raw policy logits from a fresh forward pass.
        :param policy_inputs: tuple stored at sampling time.
            For ``dit_confidence``: (policy_mask (B,T,L), c_max_input (B,T,L,top_p), timestep).
            For ``dit_hidden``: (policy_mask, hidden_states, timestep) -- no conf info,
                so the conf prior is skipped.
        :param sampling_masks: (B, T, L) bool — positions where the policy acted.
        :return: constrained (B, T, L, 3) logits with same shape as input.
        """
        constrained = logits.clone()

        # policy_inputs[0] is the mask info passed to the policy at sampling time.
        # For full_context=True (the only case currently supported), this equals
        # is_masked (which positions had [MASK] tokens) at the corresponding step.
        is_masked = policy_inputs[0].bool()  # (B, T, L)

        neg_inf_unmask = torch.full_like(constrained[..., ACTION_UNMASK], float("-inf"))
        neg_inf_remask = torch.full_like(constrained[..., ACTION_REMASK], float("-inf"))

        # 1. Confidence-aware REMASK prior (only for dit_confidence path).
        # policy_inputs[1] for dit_confidence is c_max_input shape (B, T, L, top_p).
        REMASK_CONF_PRIOR = 5.0
        c_input = policy_inputs[1] if len(policy_inputs) >= 2 else None
        if (
            isinstance(c_input, torch.Tensor)
            and c_input.dim() == 4
            and c_input.shape[:3] == constrained.shape[:3]
            and c_input.shape[-1] >= 1
        ):
            conf_max = c_input[..., 0]  # (B, T, L)
            constrained[..., ACTION_REMASK] = (
                constrained[..., ACTION_REMASK] - REMASK_CONF_PRIOR * conf_max
            )

        # 2. REMASK invalid at masked positions.
        constrained[..., ACTION_REMASK] = torch.where(
            is_masked, neg_inf_remask, constrained[..., ACTION_REMASK]
        )
        # 3. UNMASK invalid at unmasked positions.
        constrained[..., ACTION_UNMASK] = torch.where(
            ~is_masked, neg_inf_unmask, constrained[..., ACTION_UNMASK]
        )
        # 4. Outside sampling_mask, only KEEP is valid.
        outside = ~sampling_masks
        constrained[..., ACTION_UNMASK] = torch.where(
            outside, neg_inf_unmask, constrained[..., ACTION_UNMASK]
        )
        constrained[..., ACTION_REMASK] = torch.where(
            outside, neg_inf_remask, constrained[..., ACTION_REMASK]
        )
        return constrained

    def _apply_two_way_setstate_constraints(
        self,
        logits: torch.Tensor,
        policy_inputs,
        sampling_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct the constrained Bernoulli logits used at sampling time
        for sampling_mode='two_way_setstate'. Mirrors the construction in
        ``common.generation.two_way_setstate.apply_setstate_constraints`` so
        that loss-time log-prob and sampling-time log-prob are computed over
        the same Bernoulli denominators (PPO ratio unbiased).

        :param logits: (B, T, L) raw Bernoulli logits from a fresh policy forward
        :param policy_inputs: tuple stored at sampling time. For the
            ``dit_confidence_pcurrent`` policy used with this sampling mode it is
            ``(policy_mask (B,T,L), c_max_input (B,T,L,top_p), timestep (B,T,1),
            p_current_input (B,T,L))``.
        :param sampling_masks: (B, T, L) bool — positions where the policy acted.
        :return: (B, T, L) constrained Bernoulli logits, same shape as ``logits``.
        """
        from common.generation.two_way_setstate import apply_setstate_constraints

        is_masked = policy_inputs[0].bool()  # (B, T, L)
        c_input = policy_inputs[1]  # (B, T, L, top_p)
        # Top-1 conf is the first slot of the top-K conf input.
        if not (
            isinstance(c_input, torch.Tensor)
            and c_input.dim() == 4
            and c_input.shape[:3] == logits.shape
            and c_input.shape[-1] >= 1
        ):
            raise ValueError(
                f"two_way_setstate expects c_max_input shape (B,T,L,top_p>=1), "
                f"got {None if not isinstance(c_input, torch.Tensor) else c_input.shape}"
            )
        conf_top1 = c_input[..., 0]  # (B, T, L)

        return apply_setstate_constraints(
            raw_logits=logits,
            is_masked=is_masked,
            conf_top1=conf_top1,
            sampling_mask=sampling_masks,
            remask_conf_prior=self.args.setstate_remask_conf_prior,
        )

    def _log_three_way_metrics(
        self,
        policy_outputs_all: list[dict[str, Any]],
        num_steps: torch.Tensor,
        post_gathering_policy_only_index,
        mode: str,
        gen_length: int,
    ) -> None:
        """Compute + log training-time metrics specific to the 3-way policy.

        Logs:
          * action rates (unmask / keep / remask) averaged over active positions
          * early/mid/late remask rates (by relative denoising progress)
          * mean confidence of tokens the policy chose to remask (sanity check:
            should be lower than keep/unmask confidences)
          * NFE mean/std/min/max (already logged below for 2-way)
        """
        # Gather + cast to float once (incoming num_steps is int32 and not gathered yet)
        num_steps = self.accelerator.gather_for_metrics(num_steps).float()
        per_action_counts = torch.zeros(3, device=num_steps.device)
        active_total = torch.zeros(1, device=num_steps.device)
        early_remask = torch.zeros(1, device=num_steps.device)
        early_active = torch.zeros(1, device=num_steps.device)
        mid_remask = torch.zeros(1, device=num_steps.device)
        mid_active = torch.zeros(1, device=num_steps.device)
        late_remask = torch.zeros(1, device=num_steps.device)
        late_active = torch.zeros(1, device=num_steps.device)
        remask_conf_sum = torch.zeros(1, device=num_steps.device)
        remask_conf_count = torch.zeros(1, device=num_steps.device)

        for i in range(
            len(policy_outputs_all) - (1 if self.args.es_thresholds else 0)
        ):
            samples = policy_outputs_all[i]["samples"]  # (B, T, L) int64
            ms = policy_outputs_all[i]["sampling_masks"]  # (B, T, L) bool
            policy_inputs = policy_outputs_all[i]["policy_inputs"]

            B, T, L = samples.shape
            active = ms  # (B, T, L)
            active_f = active.float()

            # Action counts
            for a in range(3):
                per_action_counts[a] += ((samples == a) & active).float().sum()
            active_total += active_f.sum()

            # Early/mid/late buckets by timestep index within the trajectory
            t_idx = torch.arange(T, device=samples.device).view(1, T, 1).expand_as(samples)
            early_buk = t_idx < (T // 3)
            late_buk = t_idx >= (2 * T // 3)
            mid_buk = ~(early_buk | late_buk)

            remask_mask = (samples == ACTION_REMASK) & active
            early_remask += (remask_mask & early_buk).float().sum()
            early_active += (active & early_buk).float().sum()
            mid_remask += (remask_mask & mid_buk).float().sum()
            mid_active += (active & mid_buk).float().sum()
            late_remask += (remask_mask & late_buk).float().sum()
            late_active += (active & late_buk).float().sum()

            # Mean confidence of remasked tokens (uses policy confidence input if present).
            # For dit_confidence policy, policy_inputs is (mask, c_input (B, T, L, topp), time).
            # For dit_hidden we skip this metric.
            if len(policy_inputs) >= 2 and isinstance(policy_inputs[1], torch.Tensor):
                c_input = policy_inputs[1]
                if c_input.dim() == 4 and c_input.shape[-1] >= 1:
                    c_top1 = c_input[..., 0]  # (B, T, L)
                    if c_top1.shape == samples.shape:
                        sel = remask_mask
                        if sel.any():
                            remask_conf_sum += c_top1[sel].float().sum()
                            remask_conf_count += sel.float().sum()

        def _safe_div(n, d):
            return (n / d.clamp(min=1.0)).item()

        action_total = per_action_counts.sum().clamp(min=1.0)
        self._metrics[mode]["action_rate/unmask"].append(
            (per_action_counts[ACTION_UNMASK] / action_total).item()
        )
        self._metrics[mode]["action_rate/keep"].append(
            (per_action_counts[1] / action_total).item()
        )
        self._metrics[mode]["action_rate/remask"].append(
            (per_action_counts[ACTION_REMASK] / action_total).item()
        )
        self._metrics[mode]["remask_rate/early"].append(
            _safe_div(early_remask, early_active)
        )
        self._metrics[mode]["remask_rate/mid"].append(_safe_div(mid_remask, mid_active))
        self._metrics[mode]["remask_rate/late"].append(
            _safe_div(late_remask, late_active)
        )
        if remask_conf_count.item() > 0:
            self._metrics[mode]["remask_conf_mean"].append(
                _safe_div(remask_conf_sum, remask_conf_count)
            )

        # NFE stats (shared with 2-way block below, but we log here for 3-way).
        num_steps_policy = num_steps[post_gathering_policy_only_index]
        self._metrics[mode]["num_steps_mean"].append(num_steps_policy.mean().item())
        self._metrics[mode]["num_steps_std"].append(num_steps_policy.std().item())
        self._metrics[mode]["num_steps_min"].append(num_steps_policy.min().item())
        self._metrics[mode]["num_steps_max"].append(num_steps_policy.max().item())
        # Normalized NFE (vs gen_length — useful when comparing alpha sweep runs)
        self._metrics[mode]["nfe_per_token"].append(
            (num_steps_policy.float().mean() / max(gen_length, 1)).item()
        )

    def _log_two_way_setstate_metrics(
        self,
        policy_outputs_all: list[dict[str, Any]],
        num_steps: torch.Tensor,
        post_gathering_policy_only_index,
        mode: str,
        gen_length: int,
    ) -> None:
        """Compute + log training-time metrics specific to the 2-way set-state policy.

        Bernoulli sample = "set state to UNMASKED" (True) vs "set state to MASKED"
        (False). Action effect depends on current state, so we decompose by
        (current_state, sampled_state):
          * action_rate/unmask  = masked & sampled True  (reveal)
          * action_rate/remask  = unmasked & sampled False (the new behavior)
          * action_rate/keep    = either (masked & sampled False) | (unmasked & sampled True)

        Metric names match the 3-way logging where semantics align, so existing
        W&B dashboards' action-rate charts work for both runs.

        Also logs:
          * remask_rate/early/mid/late — by relative denoising progress
          * remask_conf_mean — base-model conf at positions chosen for remasking
            (with conf prior at 0 we expect this to be lower than the average
            unmasked-token conf if the policy learns to target uncertain tokens)
          * NFE mean/std/min/max
        """
        num_steps = self.accelerator.gather_for_metrics(num_steps).float()
        unmask_count = torch.zeros(1, device=num_steps.device)
        remask_count = torch.zeros(1, device=num_steps.device)
        keep_count = torch.zeros(1, device=num_steps.device)
        active_total = torch.zeros(1, device=num_steps.device)
        early_remask = torch.zeros(1, device=num_steps.device)
        early_active = torch.zeros(1, device=num_steps.device)
        mid_remask = torch.zeros(1, device=num_steps.device)
        mid_active = torch.zeros(1, device=num_steps.device)
        late_remask = torch.zeros(1, device=num_steps.device)
        late_active = torch.zeros(1, device=num_steps.device)
        remask_conf_sum = torch.zeros(1, device=num_steps.device)
        remask_conf_count = torch.zeros(1, device=num_steps.device)

        for i in range(
            len(policy_outputs_all) - (1 if self.args.es_thresholds else 0)
        ):
            samples = policy_outputs_all[i]["samples"].bool()  # (B, T, L) — True = "set to UNMASKED"
            ms = policy_outputs_all[i]["sampling_masks"]  # (B, T, L) bool
            policy_inputs = policy_outputs_all[i]["policy_inputs"]
            policy_mask = policy_inputs[0].bool()  # (B, T, L) — True at currently-masked positions

            B, T, L = samples.shape
            active = ms
            is_masked_active = active & policy_mask
            is_unmasked_active = active & ~policy_mask

            active_total += active.float().sum()
            unmask_count += (is_masked_active & samples).float().sum()
            remask_count += (is_unmasked_active & ~samples).float().sum()
            keep_count += (
                (is_masked_active & ~samples).float().sum()
                + (is_unmasked_active & samples).float().sum()
            )

            t_idx = torch.arange(T, device=samples.device).view(1, T, 1).expand_as(samples)
            early_buk = t_idx < (T // 3)
            late_buk = t_idx >= (2 * T // 3)
            mid_buk = ~(early_buk | late_buk)

            remask_mask = is_unmasked_active & ~samples
            early_remask += (remask_mask & early_buk).float().sum()
            early_active += (active & early_buk).float().sum()
            mid_remask += (remask_mask & mid_buk).float().sum()
            mid_active += (active & mid_buk).float().sum()
            late_remask += (remask_mask & late_buk).float().sum()
            late_active += (active & late_buk).float().sum()

            # Base-model top-1 conf at positions the policy chose to remask.
            # policy_inputs[1] for dit_confidence_pcurrent is c_max_input (B,T,L,top_p).
            if len(policy_inputs) >= 2 and isinstance(policy_inputs[1], torch.Tensor):
                c_input = policy_inputs[1]
                if c_input.dim() == 4 and c_input.shape[-1] >= 1:
                    c_top1 = c_input[..., 0]
                    if c_top1.shape == samples.shape and remask_mask.any():
                        remask_conf_sum += c_top1[remask_mask].float().sum()
                        remask_conf_count += remask_mask.float().sum()

        def _safe_div(n, d):
            return (n / d.clamp(min=1.0)).item()

        action_total = active_total.clamp(min=1.0)
        self._metrics[mode]["action_rate/unmask"].append((unmask_count / action_total).item())
        self._metrics[mode]["action_rate/remask"].append((remask_count / action_total).item())
        self._metrics[mode]["action_rate/keep"].append((keep_count / action_total).item())
        self._metrics[mode]["remask_rate/early"].append(_safe_div(early_remask, early_active))
        self._metrics[mode]["remask_rate/mid"].append(_safe_div(mid_remask, mid_active))
        self._metrics[mode]["remask_rate/late"].append(_safe_div(late_remask, late_active))
        if remask_conf_count.item() > 0:
            self._metrics[mode]["remask_conf_mean"].append(
                _safe_div(remask_conf_sum, remask_conf_count)
            )

        num_steps_policy = num_steps[post_gathering_policy_only_index]
        self._metrics[mode]["num_steps_mean"].append(num_steps_policy.mean().item())
        self._metrics[mode]["num_steps_std"].append(num_steps_policy.std().item())
        self._metrics[mode]["num_steps_min"].append(num_steps_policy.min().item())
        self._metrics[mode]["num_steps_max"].append(num_steps_policy.max().item())
        self._metrics[mode]["nfe_per_token"].append(
            (num_steps_policy.float().mean() / max(gen_length, 1)).item()
        )

    def _compute_mask_loglikelihood(
        self,
        samples: torch.Tensor,
        sampling_inputs: torch.Tensor,
        sampling_masks: torch.Tensor,
    ) -> torch.Tensor:
        if self.args.sampling_mode in ["bernoulli", "bernoulli-argmax", "two_way_setstate"]:
            return bernoulli_batch_loglik(
                samples=samples,
                utilities=sampling_inputs,
                mask_index=sampling_masks,
                dtype=self.args.loglikelihood_dtype,
            )
        elif self.args.sampling_mode == "dpls":
            return dpls_batch_loglik(
                samples=samples,
                utilities=sampling_inputs,
                stop_logit=self.args.dpls_stop_logit,
                mask_index=sampling_masks,
                dtype=self.args.loglikelihood_dtype,
            )
        elif self.args.sampling_mode == "three_way":
            return categorical_batch_loglik(
                samples=samples,
                utilities=sampling_inputs,
                mask_index=sampling_masks,
                dtype=self.args.loglikelihood_dtype,
            )
        else:
            raise ValueError(f"Unknown sampling mode: {self.args.sampling_mode}")

    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                # Generate new completions
                inputs = self._generate_and_score_completions(inputs)
                # Store for reuse in next num_iterations-1 steps
                self._buffered_inputs = inputs
            else:
                # Reuse buffered completions
                inputs = self._buffered_inputs
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]

        # Remove assertion - we now support mixed dataset types in a batch
        # assert len(set([x["dataset_type"] for x in inputs])) == 1

        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]

        # Need to add the gen_prefix to the prompt for KodCode (per-sample check)
        for i, example in enumerate(inputs):
            if example["dataset_type"] == "kodcode":
                prompts_text[i] = prompts_text[i] + example["gen_prefix"]

        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = HFTrainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Configuration for the diffusion generation
        gen_length = self.args.max_completion_length
        block_length = self.args.block_length
        temperature = self.args.temperature or 0.0

        with unwrap_model_for_generation(
            self.model_wrapped, self.accelerator
        ) as unwrapped_model:
            generation_batch_size = self.args.generation_batch_size
            prompt_completion_ids_all = []
            num_steps_all = []
            force_es_thresholds = None
            if self.args.es_thresholds:
                # TODO: For now we are hardcoding BL=32 for ES samples.
                force_es_thresholds = torch.tensor(
                    self.args.es_thresholds,
                    dtype=unwrapped_model.dtype,
                    device=unwrapped_model.device,
                ).unsqueeze(-1)
            if self.args.remasking == "policy":
                policy_outputs_all = []
                still_masked_all = []
                for i in range(0, prompt_ids.size(0), generation_batch_size):
                    end_idx = min(i + generation_batch_size, prompt_ids.size(0))
                    batch_prompt_ids = prompt_ids[i:end_idx]
                    batch_prompt_mask = prompt_mask[i:end_idx]

                    result = generate_unified(
                        model=self.dllm,
                        prompt=batch_prompt_ids,
                        remasking="policy",
                        policy=unwrapped_model,
                        gen_length=gen_length,
                        block_length=block_length,
                        temperature=temperature,
                        mask_id=self.args.mask_id,
                        sampling_mode=self.args.sampling_mode,
                        dpls_stop_logit=self.args.dpls_stop_logit,
                        full_context=self.args.policy_full_context,
                        confidences_top_p=self.args.confidences_top_p,
                        model_type=self.args.model_type,
                        attention_mask=batch_prompt_mask,
                        setstate_remask_conf_prior=self.args.setstate_remask_conf_prior,
                    )

                    # Extract values from NamedTuple
                    batch_prompt_completion_ids = result.sequences
                    batch_sampling_inputs = result.sampling_inputs
                    batch_samples = result.samples
                    batch_sampling_masks = result.sampling_masks
                    num_steps = result.steps_taken
                    batch_policy_inputs = result.policy_inputs
                    still_masked = result.still_masked

                    # Compute log-likelihood based on sampling mode
                    mask_ll = self._compute_mask_loglikelihood(
                        samples=batch_samples,
                        sampling_inputs=batch_sampling_inputs,
                        sampling_masks=batch_sampling_masks,
                    )

                    policy_outputs_all.append(
                        {
                            "samples": batch_samples,
                            "sampling_masks": batch_sampling_masks,
                            "old_per_timestep_logps": mask_ll,
                            "prompt_length": batch_prompt_ids.shape[1],
                            "sampling_inputs": batch_sampling_inputs,
                            "policy_inputs": batch_policy_inputs,
                        }
                    )
                    num_steps_all.append(num_steps)
                    still_masked_all.append(still_masked)
                    prompt_completion_ids_all.append(batch_prompt_completion_ids)
                    # Removed gc.collect() and empty_cache() from inner loop for better GPU utilization

                if force_es_thresholds is not None:
                    es_prompt_ids = (
                        prompt_ids[0:1]
                        .expand(
                            force_es_thresholds.size(0),
                            *[-1] * (prompt_ids.ndim - 1),
                        )
                        .contiguous()
                    )
                    es_prompt_mask = (
                        prompt_mask[0:1]
                        .expand(
                            force_es_thresholds.size(0),
                            *[-1] * (prompt_mask.ndim - 1),
                        )
                        .contiguous()
                    )
                    result = generate_unified(
                        model=self.dllm,
                        prompt=es_prompt_ids,
                        remasking="fastdllm",
                        thres=force_es_thresholds,
                        policy=unwrapped_model,
                        gen_length=gen_length,
                        block_length=32,  # TODO: in the future we may not want to hardcode this
                        temperature=temperature,
                        mask_id=self.args.mask_id,
                        full_context=self.args.policy_full_context,
                        confidences_top_p=self.args.confidences_top_p,
                        temperature_policy=1.0,
                        model_type=self.args.model_type,
                        attention_mask=es_prompt_mask,
                    )

                    # Extract values from NamedTuple
                    es_prompt_completion_ids = result.sequences
                    es_sampling_inputs = result.sampling_inputs
                    es_samples = result.samples
                    es_sampling_masks = result.sampling_masks
                    es_num_steps = result.steps_taken
                    es_policy_inputs = result.policy_inputs
                    # still_masked ignored for ES

                    # Compute log-likelihood based on sampling mode
                    es_mask_ll = self._compute_mask_loglikelihood(
                        samples=es_samples,
                        sampling_inputs=es_sampling_inputs,
                        sampling_masks=es_sampling_masks,
                    )

                    policy_outputs_all.append(
                        {
                            "samples": es_samples,
                            "sampling_masks": es_sampling_masks,
                            "old_per_timestep_logps": es_mask_ll,
                            "prompt_length": es_prompt_ids.shape[1],
                            "sampling_inputs": es_sampling_inputs,
                            "policy_inputs": es_policy_inputs,
                        }
                    )
                    num_steps_all.append(es_num_steps)
                    prompt_completion_ids_all.append(es_prompt_completion_ids)
                    # Removed gc.collect() and empty_cache() from inner loop for better GPU utilization

                prompt_completion_ids = torch.cat(prompt_completion_ids_all, dim=0)
                num_steps = torch.cat(num_steps_all, dim=0)
                still_masked = torch.cat(still_masked_all, dim=0)

        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        if self.args.es_thresholds:
            # Add copies for ES, if needed
            # TODO: assuming we can reuse inputs[0],
            # should be safe since all inputs should be the same on the gpu anyway
            num_es_samples = len(self.args.es_thresholds)
            inputs.extend([inputs[0]] * num_es_samples)
            prompts.extend([inputs[0]["prompt"]] * num_es_samples)
        assert len(completion_ids) == len(prompts) == len(inputs), (
            f"{len(completion_ids)=} ?!= {len(prompts)=} ?!= {len(inputs)=}"
        )

        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = (
                    prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                )
                completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}]
                )
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = (
                    f"reward {reward_func.config._name_or_path.split('/')[-1]}"
                )
            else:
                reward_func_name = reward_func.__name__
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

            if reward_func_name in [
                "lm_eval_flex_mult_reward",
                "lm_eval_flex_add_reward",
                "xml_mult_reward",
                "xml_add_reward",
                "math_correctness_mult_reward",
                "mixed_correctness_mult_reward_func",
                "mixed_correctness_add_reward_func",
                "mixed_correctness_reward_func",
                "kodcode_correctness_mult_reward",
            ]:
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    n_steps=num_steps,
                    L=gen_length,
                    alpha=self.args.alpha_compute_reward,
                    step=self._step,
                    run_name=self.args.output_dir,
                    pos_reward=self.args.alpha_correctness_reward,
                    **reward_kwargs,
                )
            elif reward_func_name == "lm_eval_harness_flexible_match_reward_func":
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    step=self._step,
                    run_name=self.args.output_dir,
                    pos_reward=self.args.alpha_correctness_reward,
                    **reward_kwargs,
                )
            else:
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    step=self._step,
                    run_name=self.args.output_dir,
                    **reward_kwargs,
                )

            assert len(output_reward_func) == len(prompts) == len(completion_ids), (
                f"{len(output_reward_func)=} != {len(prompts)=} = {len(completion_ids)=}"
            )
            # Convert None values to NaN
            output_reward_func = [
                reward if reward is not None else torch.nan
                for reward in output_reward_func
            ]
            assert len(output_reward_func) == len(prompts), (
                f"{len(output_reward_func)=} != {len(prompts)=}"
            )

            rewards_per_func[:, i] = torch.tensor(
                output_reward_func, dtype=torch.float32, device=device
            )

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = (
                torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            )
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items()
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        rewards_per_func = gather(rewards_per_func)
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)

        # Compute grouped-wise rewards
        group_size = self.num_generations + len(self.args.es_thresholds or [])
        mean_grouped_rewards = rewards.view(-1, group_size).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, group_size).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(group_size, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(group_size, dim=0)
        advantages = rewards - mean_grouped_rewards
        # Count prompts with zero std deviation (policy only for metrics)
        zero_std_count = (std_grouped_rewards < 1e-6).sum().item()
        total_prompts = std_grouped_rewards.size(0)
        zero_std_ratio = zero_std_count / total_prompts if total_prompts > 0 else 0.0

        # Slice out this process's advantages
        # Each process has per_device_train_batch_size items (potentially multiple groups)
        items_per_process = self.args.per_device_train_batch_size + (
            len(self.args.es_thresholds) if self.args.es_thresholds else 0
        )
        process_slice = slice(
            self.accelerator.process_index * items_per_process,
            (self.accelerator.process_index + 1) * items_per_process,
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(
            completion_mask.sum(1)
        ).float()

        # For rewards and other metrics (eg completion length) inferred from the prompt_completion_ids,
        # we need to slice out ES samples (if any) so as to not pollute the logs
        num_es = len(self.args.es_thresholds) if self.args.es_thresholds else 0
        if num_es > 0:
            num_processes = self.accelerator.num_processes
            policy_samples_per_process = group_size - num_es
            post_gathering_policy_only_index = torch.ones(
                completion_length.shape[0],
                dtype=torch.bool,
                device=completion_length.device,
            )
            for proc_idx in range(num_processes):
                start_idx = proc_idx * group_size
                es_start = start_idx + policy_samples_per_process
                es_end = start_idx + group_size
                post_gathering_policy_only_index[es_start:es_end] = False
        else:
            post_gathering_policy_only_index = slice(None)

        completion_length = (
            completion_length[post_gathering_policy_only_index].mean().item()
        )
        self._metrics[mode]["completion_length"].append(completion_length)
        self._metrics[mode]["zero_std_ratio"].append(zero_std_ratio)
        self._metrics[mode]["effective_steps"].append(self.effective_steps)

        still_masked = gather(still_masked)  # (N_GPUs * G,)
        still_masked = (still_masked.float() / gen_length).mean()
        self._metrics[mode]["still_masked"].append(still_masked.item())

        # Metrics: Calculate mean reward per function, but only for samples where the function was applied
        # and the sample actually came from the policy (not ES)
        rewards_per_func_policy = rewards_per_func[post_gathering_policy_only_index]
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func_policy[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)

        rewards_policy = rewards[post_gathering_policy_only_index]
        self._metrics[mode]["reward"].append(rewards_policy.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        # Log ES advantages if present
        if num_es > 0:
            # Gather all advantages across processes
            advantages_gathered = gather(advantages)
            # Extract ES advantages (inverse of policy-only index)
            advantages_es = advantages_gathered[~post_gathering_policy_only_index]
            self._metrics[mode]["es_advantage_mean"].append(advantages_es.mean().item())

        if (
            self.args.save_best_checkpoint
            and mode == "train"
            and self.accelerator.is_main_process
            and self.state.global_step % self.num_iterations == 0
        ):
            self.train_reward_queue.append(rewards_policy.mean().item())
            if np.mean(self.train_reward_queue) > self.train_reward_best:
                self.train_reward_best = np.mean(self.train_reward_queue)
                self.train_reward_best_step = self.state.global_step

                _output_dir = os.path.join(self.args.output_dir, "checkpoint-best")
                unwrapped_model = self.accelerator.unwrap_model(self.model_wrapped)
                unwrapped_model.save_pretrained(_output_dir)
                self.state.save_to_json(os.path.join(_output_dir, "trainer_state.json"))
                with open(
                    os.path.join(_output_dir, "best_train_reward.json"), "w"
                ) as f:
                    json.dump(
                        {
                            "best_train_reward": self.train_reward_best,
                            "best_train_reward_step": self.train_reward_best_step,
                        },
                        f,
                    )

                print(
                    f"Saved checkpoint-best at step {self.train_reward_best_step} with train reward {self.train_reward_best}"
                )
                if self.s3_callback is not None:
                    # use the callback to push the checkpoint to s3
                    print(f"Uploading checkpoint-best to s3: {self.args.output_dir}")
                    self.s3_callback.on_save(
                        self.args, self.state, self.control, best=True
                    )

        # 3-way categorical policy has a different sample/logits shape; log
        # three-action metrics and skip the bernoulli-specific aggregations.
        if self.args.sampling_mode == "three_way":
            self._log_three_way_metrics(
                policy_outputs_all=policy_outputs_all,
                num_steps=num_steps,
                post_gathering_policy_only_index=post_gathering_policy_only_index,
                mode=mode,
                gen_length=gen_length,
            )
        elif self.args.sampling_mode == "two_way_setstate":
            self._log_two_way_setstate_metrics(
                policy_outputs_all=policy_outputs_all,
                num_steps=num_steps,
                post_gathering_policy_only_index=post_gathering_policy_only_index,
                mode=mode,
                gen_length=gen_length,
            )
            if (
                self.log_completions
                and self.state.global_step % self.args.logging_steps == 0
            ):
                prompts_to_log = gather_object(prompts_text)
                completions_to_log = gather_object(completions_text)
                if self.accelerator.is_main_process:
                    if _rich_available:
                        print_prompt_completions_sample(
                            prompts_to_log,
                            completions_to_log,
                            rewards.tolist(),
                            self.state.global_step,
                        )
                    if (
                        self.args.report_to
                        and "wandb" in self.args.report_to
                        and wandb.run is not None
                    ):
                        import pandas as pd

                        table = {
                            "step": [str(self.state.global_step)] * len(rewards),
                            "prompt": prompts_to_log,
                            "completion": completions_to_log,
                            "reward": rewards.tolist(),
                        }
                        df = pd.DataFrame(table)
                        wandb.log({"completions": wandb.Table(dataframe=df)})

            gc.collect()
            torch.cuda.empty_cache()

            return {
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "completion_ids": completion_ids,
                "completion_mask": completion_mask,
                "advantages": advantages,
                "policy_outputs": policy_outputs_all,
            }

        # Log metrics to detect the collapse to 0 policy
        avg_us_all = []
        max_us_all = []
        non_zero_active_us_all = []
        non_zero_bs_timesteps_all = []
        for i in range(
            # Drop last batch, corresponding to ES samples, if present
            len(policy_outputs_all) - (1 if self.args.es_thresholds else 0)
        ):
            sampling_inputs = policy_outputs_all[i][
                "sampling_inputs"
            ]  # Contains logits for both Bernoulli and DPLS
            samples = policy_outputs_all[i][
                "samples"
            ]  # One-hot bernoulli outcomes for Bernoulli, ordered indices for PL
            ms = policy_outputs_all[i]["sampling_masks"]

            # Convert to probabilities for consistent logging across sampling modes
            if self.args.sampling_mode == "dpls":
                # For DPLS, sampling_inputs contains logits, convert to probabilities
                # Do not normalize over unmasked tokens
                sampling_inputs = torch.where(
                    ms.any(dim=-1).unsqueeze(-1),
                    sampling_inputs.masked_fill(~ms, float("-inf")),
                    torch.zeros_like(sampling_inputs),
                )
                us = torch.softmax(sampling_inputs, dim=-1, dtype=torch.float32)
                # Similarly samples need to be converted to one-hot
                # (do not care about their ordering for logging)
                # Since samples vector may contain padding (-1), we clamp to valid indices
                # but then dynamically set the value to True/False - making the scatter
                # a no-op at the padded indices
                bs = (
                    torch.zeros_like(us, dtype=torch.int)
                    .scatter_add(-1, samples.clamp(min=0), (samples >= 0).int())
                    .bool()
                )
            else:
                # For Bernoulli, sampling_inputs contains logits, convert to probabilities with sigmoid
                us = torch.sigmoid(sampling_inputs)
                # And samples contains the sampled unmasking indices (one-hot)
                bs = samples

            # For average unmask probability as well as for proportion of non-zero unmask probability,
            # we average both over time and the block dimension
            avg_us = (us * ms).sum(dim=(-1, -2)) / ms.sum(dim=(-1, -2))
            eps = 0.001
            non_zero_active_us = ((us * ms) > eps).sum(dim=(-1, -2)) / ms.sum(
                dim=(-1, -2)
            )
            avg_us_all.append(avg_us)
            non_zero_active_us_all.append(non_zero_active_us)

            active_timesteps = ms.any(dim=-1)  # (B, T)
            # For max unmask probability, we aggregate over the block dimension only
            # and then avg over the active timesteps
            max_us = torch.amax(us * ms, dim=(-1))
            max_us = (max_us * active_timesteps).sum(dim=-1) / active_timesteps.sum(
                dim=-1
            )
            max_us_all.append(max_us)
            # For non-zero bs, we take any over the block dimension and
            # then avg over the active timesteps
            non_zero_bs_timesteps = bs.any(dim=-1)  # (B, T)
            non_zero_bs_timesteps = (non_zero_bs_timesteps * active_timesteps).sum(
                dim=-1
            ) / active_timesteps.sum(dim=-1)
            non_zero_bs_timesteps_all.append(non_zero_bs_timesteps)

        avg_us_all = torch.cat(avg_us_all, dim=0)
        max_us_all = torch.cat(max_us_all, dim=0)
        non_zero_active_us_all = torch.cat(non_zero_active_us_all, dim=0)
        non_zero_bs_timesteps_all = torch.cat(non_zero_bs_timesteps_all, dim=0)

        avg_us_all = self.accelerator.gather_for_metrics(avg_us_all)
        non_zero_active_us_all = self.accelerator.gather_for_metrics(
            non_zero_active_us_all
        )
        max_us_all = self.accelerator.gather_for_metrics(max_us_all)

        num_steps = self.accelerator.gather_for_metrics(num_steps).float()

        num_steps = num_steps[post_gathering_policy_only_index]

        non_zero_bs_timesteps_all = self.accelerator.gather_for_metrics(
            non_zero_bs_timesteps_all
        )

        self._metrics[mode]["mean_unmask_prob"].append(avg_us_all.mean().item())
        self._metrics[mode]["non_zero_unmask_prob"].append(
            non_zero_active_us_all.mean().item()
        )
        self._metrics[mode]["max_unmask_prob"].append(max_us_all.mean().item())

        self._metrics[mode]["num_steps_mean"].append(num_steps.mean().item())
        self._metrics[mode]["num_steps_std"].append(num_steps.std().item())
        self._metrics[mode]["num_steps_min"].append(num_steps.min().item())
        self._metrics[mode]["num_steps_max"].append(num_steps.max().item())

        self._metrics[mode]["non_zero_bs_timesteps_mean"].append(
            non_zero_bs_timesteps_all.mean().item()
        )
        self._metrics[mode]["non_zero_bs_timesteps_std"].append(
            non_zero_bs_timesteps_all.std().item()
        )
        self._metrics[mode]["non_zero_bs_timesteps_min"].append(
            non_zero_bs_timesteps_all.min().item()
        )
        self._metrics[mode]["non_zero_bs_timesteps_max"].append(
            non_zero_bs_timesteps_all.max().item()
        )

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
        ):
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if _rich_available:
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                if (
                    self.args.report_to
                    and "wandb" in self.args.report_to
                    and wandb.run is not None
                ):
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        # clear cuda memory
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "policy_outputs": policy_outputs_all,
        }
