"""DiT-confidence policy extension that adds p(current_token) as an input feature.

Motivation: the upstream `DiTConfidencePolicy` takes only top-K confidence values
(`probs.topk(K)`), which does not let the policy distinguish "model agrees with
what's at this position" from "model would predict something else here." For
3-way / set-state generation that needs to decide WHEN to remask, the latter
case is exactly the signal we want — but the upstream policy is blind to it.

This class adds:
  - `p_current`: the model's probability of the actual token at this position.
    Computed as `probs.gather(-1, x.unsqueeze(-1)).squeeze(-1)` — one gather on
    tensors already in memory, ~zero compute cost.
  - A learned `Linear(1, hidden_dim)` projection for `p_current` — kept
    independent of the existing `confidence_proj` so the two features have
    distinct embedding manifolds. The two projections are summed (along with
    mask + time conditioning) before transformer blocks, the same combine
    pattern used for timestep/mask in the upstream policy.
  - A learnable `masked_p_embedding` parameter (shape `hidden_dim`, matching
    the output of the p_current projection) that is swapped in at masked
    positions, where `p_current` reads `p(mask_id|context) ≈ 0` and is
    uninformative.

Layer name compatibility: every layer that exists in upstream
`DiTConfidencePolicy` keeps the same name and shape here, so an upstream 2-way
checkpoint loads via `strict=False` into this class without conversion. The new
parameters (`p_current_proj.*`, `masked_p_embedding`) are reported as missing
keys and left at fresh init; they're ~257 new params on top of ~331K upstream.
"""
import torch
import torch.nn as nn

from common.models.policy_layers import RoPEDiTBlock
from common.models.policy_layers import sinusoidal_time_embedding


class DiTConfidencePCurrentPolicy(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        feedforward_dim: int = 512,
        num_heads: int = 1,
        dropout: float = 0.0,
        time_embed_dim: int = 128,
        confidences_top_p: int = 1,
        smart_init: float | None = None,
        num_blocks: int = 1,
        time_period: float = 1,
        num_actions: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.time_period = time_period
        self.num_blocks = num_blocks
        self.confidences_top_p = confidences_top_p
        self.num_actions = num_actions

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # confidence_proj: identical to upstream `DiTConfidencePolicy`. Takes
        # raw top-K confidence scalars (B, L, K) and projects to hidden_dim.
        # Loads cleanly from Apple's 2-way checkpoint via strict=False.
        self.confidence_proj = nn.Linear(confidences_top_p, self.hidden_dim)

        # p_current_proj: new. Takes raw scalar p(current_token | context)
        # (B, L, 1) and projects to hidden_dim. Trained from fresh init when
        # warmstarting from an upstream 2-way ckpt (those don't have this
        # layer). The downstream transformer learns to use this signal.
        self.p_current_proj = nn.Linear(1, self.hidden_dim)

        # Learnable embedding swapped in at masked positions for the p_current
        # feature, where the actual probability is degenerate (≈0). Shape
        # matches the output of `p_current_proj` so it lives in the same
        # post-projection space.
        self.masked_p_embedding = nn.Parameter(
            torch.randn(self.hidden_dim) * 0.02
        )

        self.mask_embedding = nn.Embedding(2, self.hidden_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                RoPEDiTBlock(
                    d_model=self.hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=feedforward_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, num_actions)

        if smart_init is not None:
            self.apply_smart_init(smart_init)

    def apply_smart_init(self, target_logit: float):
        """Match upstream `DiTConfidencePolicy.apply_smart_init` semantics."""
        with torch.no_grad():
            for block in self.transformer_blocks:
                block.ada_conditioning.weight.data.zero_()
                block.ada_conditioning.bias.data.zero_()

            if self.num_actions == 1:
                self.output_proj.bias.data.fill_(target_logit)
            elif self.num_actions == 3:
                self.output_proj.bias.data = torch.tensor(
                    [target_logit, 0.0, -4.0],
                    dtype=self.output_proj.bias.dtype,
                    device=self.output_proj.bias.device,
                )
            else:
                raise ValueError(f"Unsupported num_actions={self.num_actions}")

    def forward(
        self,
        m: torch.Tensor,
        c: torch.Tensor,
        timestep: torch.Tensor,
        p_current: torch.Tensor,
    ) -> torch.Tensor:
        """:param m: (*B,L) mask with 1=masked, 0=unmasked
        :param c: (*B,L,confidences_top_p) top-K confidence values in [0,1]
        :param timestep: (*B,1) diffusion timestep in [0, 1]
        :param p_current: (*B,L) model's probability of the actual current token at
            this position (≈0 at masked positions; the masked_p_embedding handles
            those via a learnable swap-in).
        :return: (*B,L) unmasking logits if num_actions==1, else (*B,L,num_actions)
        """
        *B, L, _ = c.shape
        assert m.shape == (*B, L)
        assert p_current.shape == (*B, L), (
            f"Expected p_current shape {(*B, L)}, got {p_current.shape}"
        )
        assert timestep.shape == (*B, 1)

        # Mask + time conditioning (same as upstream)
        cond = self.mask_embedding(m.int())
        time_embed = sinusoidal_time_embedding(
            timestep, self.time_embed_dim, max_period=self.time_period
        )
        time_embed = time_embed.to(cond.dtype)
        time_embed = time_embed.expand((*([-1] * len(B)), L, -1))
        time_embed = self.time_mlp(time_embed)
        cond = cond + time_embed

        # Top-K confidence projection — identical to upstream `DiTConfidencePolicy`.
        x = self.confidence_proj(c)

        # p_current projection — new. Add to x (same combine pattern as
        # mask + time conditioning), then swap in the learnable masked
        # embedding at masked positions where p_current is degenerate.
        p_feature = self.p_current_proj(
            p_current.unsqueeze(-1).to(x.dtype)
        )  # (*B, L, hidden_dim)
        is_masked = m.bool()
        p_feature = torch.where(
            is_masked.unsqueeze(-1),
            self.masked_p_embedding.to(p_feature.dtype).expand_as(p_feature),
            p_feature,
        )
        x = x + p_feature

        # Transformer + AdaLN-style conditioning (same as upstream)
        original_shape = x.shape
        x_flat = x.view(-1, L, self.hidden_dim)
        cond_flat = cond.view(-1, L, self.hidden_dim)
        for block in self.transformer_blocks:
            x_flat = block(x_flat, cond_flat)
        x = x_flat.view(original_shape)

        x = self.final_norm(x)
        raw_logits = self.output_proj(x)
        if self.num_actions == 1:
            return raw_logits.squeeze(-1)
        return raw_logits
