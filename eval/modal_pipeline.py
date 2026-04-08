"""
Modal wrapper for eval_pipeline.py — runs the full evaluation pipeline on a remote GPU.

This orchestrates checkpoint download, multi-seed/multi-dataset evaluation,
and result aggregation in a single remote call.

Usage:
    # Evaluate a local checkpoint directory on gsm8k
    modal run modal_eval_pipeline.py \
        --run-paths /checkpoints/my_run \
        --config-path configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
        --checkpoints last \
        --datasets gsm8k

    # Evaluate multiple checkpoints across datasets with multiple seeds
    modal run modal_eval_pipeline.py \
        --run-paths /checkpoints/my_run \
        --config-path configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
        --checkpoints 500,1000,last \
        --datasets gsm8k,math \
        --seeds 42,43,44

    # Baseline evaluation
    modal run modal_eval_pipeline.py \
        --run-paths /checkpoints/baseline-low_confidence-K32 \
        --config-path configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
        --checkpoints self \
        --datasets gsm8k

    # Skip aggregation
    modal run modal_eval_pipeline.py \
        --run-paths /checkpoints/my_run \
        --config-path configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
        --checkpoints last \
        --no-aggregate
"""

import modal

# Modal
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.4.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
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
        "numpy>=1.26.0",
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
results_vol = modal.Volume.from_name("eval-results", create_if_missing=True)

app = modal.App("llada-eval-pipeline", image=image)

TIMEOUT_SECONDS = 12 * 3600  # 12 hours max


@app.function(
    gpu="A100-40GB",
    timeout=TIMEOUT_SECONDS,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/checkpoints": checkpoints_vol,
        "/results": results_vol,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),  # HF_TOKEN
    ],
)
def run_pipeline(
    run_paths: str,
    config_path: str,
    datasets: str = "gsm8k",
    temperatures: str = "1.0",
    sampling_mode: str | None = None,
    checkpoints: str = "",
    seeds: str = "42,43,44",
    block_length: int | None = None,
    gen_length: int | None = None,
    model_path: str = "GSAI-ML/LLaDA-8B-Instruct",
    save_path: str = "/results/pipeline",
    n_test: int | None = None,
    no_aggregate: bool = False,
):
    """Run eval_pipeline.py on the remote GPU."""
    import json
    import os
    import subprocess
    from datetime import datetime, timezone
    from pathlib import Path

    os.chdir("/root/repo")

    cmd = [
        "python", "-m", "eval.pipeline",
        run_paths,
        config_path,
        "--datasets", datasets,
        "--temperatures", temperatures,
        "--checkpoints", checkpoints,
        "--seeds", seeds,
        "--model_path", model_path,
        "--save_path", save_path,
    ]

    # Create baseline marker if needed
    for run_path in run_paths.split(","):
        name = run_path.strip().split("/")[-1]
        if name.startswith("baseline-"):
            ckpt_dir = Path(run_path) / f"checkpoint-{name}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / ".baseline_marker").touch()

    if sampling_mode is not None:
        cmd += ["--sampling_mode", sampling_mode]
    if block_length is not None:
        cmd += ["--block_length", str(block_length)]
    if gen_length is not None:
        cmd += ["--gen_length", str(gen_length)]
    if n_test is not None:
        cmd += ["--n_test", str(n_test)]
    if no_aggregate:
        cmd += ["--no_aggregate"]

    # Log the command
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": cmd,
        "run_paths": run_paths,
        "config_path": config_path,
        "datasets": datasets,
        "checkpoints": checkpoints,
        "seeds": seeds,
        "save_path": save_path,
    }
    os.makedirs(save_path, exist_ok=True)
    log_path = os.path.join(save_path, "commands.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"\n{'='*60}")
    print(f"Running eval pipeline")
    print(f"Run paths: {run_paths}")
    print(f"Config: {config_path}")
    print(f"Datasets: {datasets}")
    print(f"Checkpoints: {checkpoints}")
    print(f"Seeds: {seeds}")
    print(f"Save path: {save_path}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd="/root/repo")
    if result.returncode != 0:
        raise RuntimeError(f"eval_pipeline.py exited with code {result.returncode}")

    results_vol.commit()
    print(f"\nResults saved to volume at {save_path}")


@app.local_entrypoint()
def main(
    run_paths: str = "",
    config_path: str = "configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml",
    datasets: str = "gsm8k",
    temperatures: str = "1.0",
    sampling_mode: str = None,
    checkpoints: str = "",
    seeds: str = "42,43,44",
    block_length: int = None,
    gen_length: int = None,
    model_path: str = "GSAI-ML/LLaDA-8B-Instruct",
    save_path: str = "/results/pipeline",
    n_test: int = None,
    no_aggregate: bool = False,
):
    run_pipeline.remote(
        run_paths=run_paths,
        config_path=config_path,
        datasets=datasets,
        temperatures=temperatures,
        sampling_mode=sampling_mode,
        checkpoints=checkpoints,
        seeds=seeds,
        block_length=block_length,
        gen_length=gen_length,
        model_path=model_path,
        save_path=save_path,
        n_test=n_test,
        no_aggregate=no_aggregate,
    )