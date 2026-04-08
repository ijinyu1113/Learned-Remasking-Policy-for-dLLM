"""
Modal wrapper for train.py — runs GRPO policy training on a remote GPU.

Usage:
    # Default training run with the standard config
    modal run modal_train.py

    # Custom config
    modal run modal_train.py --config configs/experiment_configs/my_config.yaml

    # With wandb logging (set WANDB_API_KEY in the "wandb-secret" Modal secret)
    modal run modal_train.py --wandb-project my-project --wandb-run my-run
"""

import modal

# Modal image
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git")
    .pip_install(
        "torch==2.4.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "packaging",
        "ninja",
        "numpy>=1.26.0",
    )
    .run_commands("pip install wheel setuptools && pip install flash-attn --no-build-isolation")
    .add_local_dir(".", remote_path="/root/repo", copy=True, ignore=["*.pyc", "__pycache__", ".git", "*.egg-info", "outputs", "eval_results"])
    .run_commands("cd /root/repo && pip install --no-deps -e .")
    .pip_install(
        "transformers==4.53.0",
        "accelerate==1.4.0",
        "trl==0.19.1",
        "peft==0.15.1",
        "deepspeed==0.16.4",
        "datasets==4.0.0",
        "tiktoken==0.9.0",
        "safetensors",
        "bitsandbytes==0.45.3",
        "evaluate",
        "sentencepiece",
        "scipy",
        "scikit-learn",
        "s3fs",
        "pandas",
        "matplotlib",
        "tqdm",
        "regex",
        "wandb>=0.16.0",
        "python-dotenv",
        "protobuf",
    )
)

# Persistent volumes
hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
checkpoints_vol = modal.Volume.from_name("train-checkpoints", create_if_missing=True)

app = modal.App("llada-train", image=image)

TIMEOUT_SECONDS = 24 * 3600  # 24 hours max for training

@app.function(
    gpu="A100-80GB",
    timeout=TIMEOUT_SECONDS,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/checkpoints": checkpoints_vol,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),    # HF_TOKEN
        modal.Secret.from_name("wandb-secret")
    ],
)
def run_train(
    config: str = "configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml",
    output_dir: str = "/checkpoints/policy_run",
    extra_args: list[str] | None = None,
    wandb_project: str | None = None,
):
    """Run train.py on the remote GPU."""
    import os
    import subprocess
    import json
    from datetime import datetime, timezone

    os.chdir("/root/repo")

    # Ensure output directory exists on the persistent volume
    if not output_dir.startswith("s3://"):
        os.makedirs(output_dir, exist_ok=True)

    # Build the command
    cmd = [
        "python", "-m", "train.train",
        "--config", config,
        "--output_dir", output_dir,
    ]

    # Append any extra CLI args the user passed through
    if extra_args:
        cmd += extra_args
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project

    # Log the command to a JSON file alongside the checkpoints
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": cmd,
        "config": config,
        "output_dir": output_dir,
        "extra_args": extra_args,
    }
    log_path = os.path.join(output_dir, "commands.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"\n{'='*60}")
    print(f"Running training")
    print(f"Config: {config}")
    print(f"Output dir: {output_dir}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd="/root/repo")
    if result.returncode != 0:
        raise RuntimeError(f"train.py exited with code {result.returncode}")

    # Commit checkpoints volume so they persist after the container exits
    checkpoints_vol.commit()
    print(f"\nCheckpoints saved to volume at {output_dir}")


# CLI entrypoint for `modal run modal_train.py`
@app.local_entrypoint()
def main(
    config: str = "configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml",
    output_dir: str = "/checkpoints/policy_run",
    wandb_project: str = None,
    wandb_run: str = None,
):
    extra_args = []
    if wandb_project:
        extra_args += ["--report_to", "wandb"]
    if wandb_run:
        extra_args += ["--run_name", wandb_run]

    run_train.remote(
        config=config,
        output_dir=output_dir,
        extra_args=extra_args if extra_args else None,
        wandb_project=wandb_project
    )