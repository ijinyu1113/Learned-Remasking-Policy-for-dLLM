"""
Modal wrapper for eval.py — runs LLaDA evaluation on a remote GPU.

Usage:
    # Full baseline eval
    modal run modal_eval.py --baseline low_confidence --dataset gsm8k

    # Learned 2-way policy eval (provide a policy checkpoint path)
    modal run modal_eval.py --policy-path /outputs/policy2way/checkpoint-500/model.safetensors \
        --dataset gsm8k --temperature 1.0 --sampling-mode bernoulli-argmax

    # Fast-dLLM baseline
    modal run modal_eval.py --baseline fastdllm --thres 0.7 --dataset gsm8k
"""

import modal

# Modal image: installs everything the repo needs
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
results_vol = modal.Volume.from_name("eval-results", create_if_missing=True)

app = modal.App("llada-eval", image=image)

TIMEOUT_SECONDS = 6 * 3600  # 6 hours max

@app.function(
    gpu="A100-40GB",
    timeout=TIMEOUT_SECONDS,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/results": results_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")], # HF_TOKEN
)
def run_eval(
    baseline: str | None = None,
    dataset: str = "gsm8k",
    config_path: str = "configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml",
    diffusion_steps: int = 32,
    thres: float = 0.7,
    seeds: str = "42",
    temperature: float = 0.0,
    temperature_policy: float = 1.0,
    sampling_mode: str | None = None,
    policy_path: str | None = None,
    n_test: int | None = None,
    batch_size: int = 1,
    gen_length: int | None = None,
    block_length: int | None = None,
    save_path: str | None = None,
    few_shot: int = -1,
    model_path: str | None = None,
    suffix: str = "",
    dont_save: bool = False,
):
    """Run eval.py on the remote GPU."""
    import os
    import subprocess
    import json
    from datetime import datetime, timezone

    os.chdir("/root/repo")

    seed_list = [int(s.strip()) for s in seeds.split(",")]

    for seed in seed_list:
        # Build the checkpoint / output directory structure
        remasking = baseline if baseline else "policy"
        _policy_path = policy_path

        _save_path = save_path or f"/results/{dataset}_{remasking}_seed{seed}"

        cmd = [
            "python", "-m", "eval.eval",
            "--config", config_path,
            "--dataset", dataset,
            "--seed", str(seed),
            "--temperature", str(temperature),
            "--temperature_policy", str(temperature_policy),
            "--batch_size", str(batch_size),
            "--remasking", remasking,
            "--output_dir", _save_path,
        ]

        if _policy_path:
            cmd += ["--policy_path", _policy_path]
        if n_test is not None:
            cmd += ["--n_test", str(n_test)]
        if gen_length is not None:
            cmd += ["--gen_length", str(gen_length)]
        if block_length is not None:
            cmd += ["--block_length", str(block_length)]
        if sampling_mode is not None:
            cmd += ["--sampling_mode", sampling_mode]
        if remasking in ("random", "low_confidence"):
            cmd += ["--diffusion_steps", str(diffusion_steps)]
        if remasking == "fastdllm":
            cmd += ["--thres", str(thres)]
        if few_shot != -1:
            cmd += ["--few_shot", str(few_shot)]
        if model_path is not None:
            cmd += ["--model_path", model_path]
        if suffix:
            cmd += ["--suffix", suffix]
        if dont_save:
            cmd += ["--dont_save"]

        # Log the command
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "command": cmd,
            "seed": seed,
            "dataset": dataset,
            "remasking": remasking,
            "save_path": _save_path,
        }
        os.makedirs(_save_path, exist_ok=True)
        log_path = os.path.join(_save_path, "commands.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


        print(f"\n{'='*60}")
        print(f"Running eval: seed={seed}, dataset={dataset}, remasking={remasking}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}\n")

        result = subprocess.run(cmd, cwd="/root/repo")
        if result.returncode != 0:
            raise RuntimeError(f"eval.py exited with code {result.returncode}")

    # Commit the results volume so files persist
    results_vol.commit()
    print("\nResults saved to /results volume.")


# CLI entrypoint for `modal run modal_eval.py`
@app.local_entrypoint()
def main(
    baseline: str = None,
    dataset: str = "gsm8k",
    config_path: str = "configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml",
    diffusion_steps: int = 32,
    thres: float = 0.7,
    seeds: str = "42",
    temperature: float = 0.0,
    temperature_policy: float = 1.0,
    sampling_mode: str = None,
    policy_path: str = None,
    n_test: int = None,
    batch_size: int = 1,
    gen_length: int = None,
    block_length: int = None,
    save_path: str = None,
    few_shot: int = -1,
    model_path: str | None = None,
    suffix: str = "",
    dont_save: bool = False,
):
    run_eval.remote(
        baseline=baseline,
        dataset=dataset,
        config_path=config_path,
        diffusion_steps=diffusion_steps,
        thres=thres,
        seeds=seeds,
        temperature=temperature,
        temperature_policy=temperature_policy,
        sampling_mode=sampling_mode,
        policy_path=policy_path,
        n_test=n_test,
        batch_size=batch_size,
        gen_length=gen_length,
        block_length=block_length,
        save_path=save_path,
        few_shot=few_shot,
        model_path=model_path,
        suffix=suffix,
        dont_save=dont_save,
    )