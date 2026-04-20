"""Sanity-check the full 3-way generation loop end-to-end with a tiny dummy model.

We stub out the 8B LLaDA with a random-logit MLP so the whole pipeline
(mask -> policy -> categorical sample -> apply unmask/remask -> step) runs.

Run: python -m scripts.sanity_check_3way_generation
"""
import torch
import torch.nn as nn

from common.generation.generation import generate_unified
from common.models.policy import DiTConfidencePolicy


VOCAB = 64
MASK_ID = 63
HIDDEN = 32


class DummyOutput:
    def __init__(self, logits, hidden):
        self.logits = logits
        self.hidden_states = (hidden,)


class DummyLLaDA(nn.Module):
    """Produces uniform-ish logits; just enough to drive the pipeline."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.to_logits = nn.Linear(HIDDEN, VOCAB)
        self.dtype = torch.float32

    def forward(self, x, attention_mask=None, output_hidden_states=False):
        h = self.embed(x)
        logits = self.to_logits(h)
        return DummyOutput(logits=logits, hidden=h)


def main():
    torch.manual_seed(0)
    model = DummyLLaDA()
    policy = DiTConfidencePolicy(
        hidden_dim=32, feedforward_dim=64, num_heads=2, num_blocks=1,
        time_embed_dim=16, confidences_top_p=1, num_actions=3, smart_init=-2.0,
    )
    # Force aggressive unmasking then noticeable remasking so both code paths fire.
    with torch.no_grad():
        policy.output_proj.bias.data[0] = 3.0   # strong UNMASK bias
        policy.output_proj.bias.data[1] = -2.0  # mild KEEP suppression
        policy.output_proj.bias.data[2] = 1.5   # strong REMASK bias at unmasked positions
    policy.policy_type = "dit_confidence"

    prompt = torch.tensor([[1, 2, 3, 4]])  # 4-token prompt
    result = generate_unified(
        model=model,
        prompt=prompt,
        remasking="policy",
        policy=policy,
        gen_length=8,
        block_length=4,
        temperature=0.0,
        mask_id=MASK_ID,
        sampling_mode="three_way",
        full_context=True,
        confidences_top_p=1,
        model_type="LLaDA",
        temperature_policy=1.0,
        record_trajectory=True,
    )
    assert result.sequences.shape == (1, 12), result.sequences.shape
    assert result.sampling_inputs is not None, "sampling_inputs missing"
    assert result.sampling_inputs.shape[-1] == 3, result.sampling_inputs.shape
    # samples should be an int tensor
    assert result.samples.dtype in (torch.int64, torch.long)
    # Trajectory should include remask flags
    assert result.trajectory and "remask" in result.trajectory[0]
    total_remasks = sum(int(step["remask"].sum().item()) for step in result.trajectory)
    total_unmasks = sum(int(step["unmask"].sum().item()) for step in result.trajectory)
    final_mask_count = int((result.sequences[:, -8:] == MASK_ID).sum().item())
    print("steps recorded:", len(result.trajectory))
    print("total unmasks:", total_unmasks, " total remasks:", total_remasks)
    print("final # masked tokens (should eventually be 0 or close):", final_mask_count)
    assert total_unmasks > 0, "No tokens ever got unmasked?"
    assert total_remasks > 0, "No tokens ever got remasked (with strong remask bias)?"
    print("End-to-end 3-way generation sanity OK.")


if __name__ == "__main__":
    main()
