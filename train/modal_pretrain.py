"""
Modal wrapper for pretrain.py — runs policy pretraining on a remote GPU.

Setup (optional, for wandb logging):
    Create a Modal secret with your wandb credentials:
    modal secret create wandb-secret \
        --env-file <(echo -e "WANDB_API_KEY=your_api_key\nWANDB_PROJECT=your_project_name\nWANDB_ENTITY=your_entity")

    Note: WANDB_ENTITY is optional (username or team name). If omitted, defaults to your user account.

Usage:
    # Run with defaults (3-way policy, gsm8k, 3 epochs, wandb logging enabled)
    modal run modal_pretrain.py

    # Custom pretraining config with checkpointing
    modal run -d train/modal_pretrain.py \
        --policy-type dit_confidence_pcurrent \
        --dataset gsm8k \
        --num-epochs 5 \
        --batch-size 16 \
        --checkpoint-interval 1

    # Disable wandb logging if not configured
    modal run modal_pretrain.py --use-wandb false
"""

import modal

# Modal image
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git")
    .pip_install(
        "torch==2.6.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "packaging",
        "ninja",
        "numpy>=1.26.0",
    )
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
pretrain_vol = modal.Volume.from_name("pretrain-checkpoints", create_if_missing=True)

app = modal.App("llada-pretrain", image=image)

TIMEOUT_SECONDS = 12 * 3600  # 12 hours max for pretraining

@app.function(
    gpu="A100-80GB",
    timeout=TIMEOUT_SECONDS,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/checkpoints": pretrain_vol,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),    # HF_TOKEN
        modal.Secret.from_name("wandb-secret")           # WANDB_API_KEY
    ],
)
def run_pretrain(
    policy_type: str = "dit_confidence",
    dataset: str = "gsm8k",
    model_path: str = "GSAI-ML/LLaDA-8B-Instruct",
    batch_size: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 5e-5,
    mask_prob: float = 0.4,
    corrupt_prob: float = 0.2,
    exclusion_k: int = 5,
    top_p: float = 0.70,
    temperature: float = 1.0,
    confidence_threshold: float = 0.9,
    unmask_reward: float = 1.0,
    remask_reward: float = 1.0,
    checkpoint_interval: int = 1,
    output_dir: str = "/checkpoints/pretrained_policy",
    use_wandb: bool = True,
):
    """Run pretrain.py on the remote GPU."""
    import os
    import sys
    from argparse import Namespace
    from datetime import datetime, timezone
    import json

    os.chdir("/root/repo")
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Import after chdir
    from common.config import Config
    from train.pretrain import main

    # Log the command to a JSON file
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "policy_type": policy_type,
        "dataset": dataset,
        "model_path": model_path,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "mask_prob": mask_prob,
        "corrupt_prob": corrupt_prob,
        "output_dir": output_dir,
    }
    log_path = os.path.join(output_dir, "pretrain_commands.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"\n{'='*60}")
    print(f"Running pretraining")
    print(f"Policy type: {policy_type}")
    print(f"Dataset: {dataset}")
    print(f"Batch size: {batch_size}")
    print(f"Num epochs: {num_epochs}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*60}\n")

    # Build config objects
    config = Config(
        model_path=model_path,
        dataset=dataset,
        policy_type=policy_type,
        policy_num_blocks=1,
        policy_hidden_dim=128,
        policy_num_heads=2,
        policy_dropout=0.0,
        policy_feedforward_dim=512,
        policy_time_embed_dim=128,
        policy_full_context=True,
        policy_smart_init=-2.0,
        confidences_top_p=1,
        num_policy_actions=1 if policy_type == "dit_confidence_pcurrent" else 3,
    )

    model_config = Namespace()

    pretrain_config = Namespace(
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        mask_prob=mask_prob,
        corrupt_prob=corrupt_prob,
        exclusion_k=exclusion_k,
        top_p=top_p,
        temperature=temperature,
        confidence_threshold=confidence_threshold,
        unmask_reward=unmask_reward,
        remask_reward=remask_reward,
        checkpoint_interval=checkpoint_interval,
        output_dir=output_dir,
        log_interval=10,
        use_wandb=use_wandb,
    )

    # Run pretraining
    try:
        main(config, model_config, pretrain_config)
        print(f"\n✓ Pretraining completed successfully")
    except Exception as e:
        print(f"\n✗ Pretraining failed: {e}")
        raise

    # Commit volume so checkpoints persist
    pretrain_vol.commit()
    print(f"✓ Checkpoints saved to volume at {output_dir}")


# CLI entrypoint for `modal run modal_pretrain.py`
@app.local_entrypoint()
def main(
    policy_type: str = "dit_confidence",
    dataset: str = "gsm8k",
    model_path: str = "GSAI-ML/LLaDA-8B-Instruct",
    batch_size: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 5e-5,
    mask_prob: float = 0.4,
    corrupt_prob: float = 0.2,
    exclusion_k: int = 5,
    top_p: float = 0.70,
    temperature: float = 1.0,
    confidence_threshold: float = 0.9,
    unmask_reward: float = 1.0,
    remask_reward: float = 1.0,
    checkpoint_interval: int = 1,
    output_dir: str = "/checkpoints/pretrained",
    use_wandb: bool = True,
):
    """CLI entrypoint for modal pretraining."""
    run_pretrain.remote(
        policy_type=policy_type,
        dataset=dataset,
        model_path=model_path,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        mask_prob=mask_prob,
        corrupt_prob=corrupt_prob,
        exclusion_k=exclusion_k,
        top_p=top_p,
        temperature=temperature,
        confidence_threshold=confidence_threshold,
        unmask_reward=unmask_reward,
        remask_reward=remask_reward,
        checkpoint_interval=checkpoint_interval,
        output_dir=output_dir,
        use_wandb=use_wandb,
    )
