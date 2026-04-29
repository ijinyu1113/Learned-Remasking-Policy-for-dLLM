import os

import wandb
from transformers import TrainerCallback


class WandbCheckpointArtifact(TrainerCallback):
    """Pushes each saved checkpoint to W&B as a versioned artifact.

    Uploads only the inference-relevant files (model.safetensors + config.json),
    not optimizer / scheduler / rng / tokenizer state. A 3-way policy checkpoint
    is ~1.3MB; per-step uploads are cheap.

    Failure modes are swallowed so a W&B outage cannot break training.
    """

    def __init__(self, run_name: str | None = None, files: tuple[str, ...] = ("model.safetensors", "config.json")):
        self.run_name = run_name
        self.files = files

    def on_save(self, args, state, control, **kwargs):
        if wandb.run is None:
            return
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            return
        try:
            name = f"{self.run_name or wandb.run.name}-ckpt{state.global_step}"
            artifact = wandb.Artifact(name=name, type="model",
                                      metadata={"global_step": state.global_step})
            for fname in self.files:
                fpath = os.path.join(ckpt_dir, fname)
                if os.path.isfile(fpath):
                    artifact.add_file(fpath)
            wandb.log_artifact(artifact)
        except Exception as e:
            print(f"[WandbCheckpointArtifact] upload failed at step {state.global_step}: {e}")
