# pretraining script for 3-way and 2-way set-state policies
#
# creates synthetic tasks to pretrain the policy before RL:
# 1. randomly mask some tokens in reference completions
# 2. randomly corrupt (replace with "bad" samples) other tokens
# 3. run frozen dLLM on corrupted sequence to get per-token confidences
# 4. train policy with supervised learning:
#    - positive reward for unmasking high-confidence tokens
#    - positive reward for remasking corrupted tokens
#    - zero reward for keeping tokens in correct state

import os
import torch
import wandb
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from common.config import Config
from common.models.policy import DiTConfidencePolicy, DiTHiddenStatePolicy, PolicyHFWrapper
from common.models.policy_pcurrent import DiTConfidencePCurrentPolicy
from common.s3 import S3UploadCallback
from data.data_utils import get_gsm8k_questions, get_math_questions, set_random_seed
from train.pretrain_utils import corrupt_and_mask_sequences, compute_pretrain_loss, compute_pretrain_metrics

load_dotenv()
token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("Hugging Face token not found in environment variables. Please set HF_TOKEN.")

torch.set_float32_matmul_precision("high")

MASK_TOKENS_MAP = {"LLaDA": 126336, "Dream": 151666}


def main(config: Config, model_config, pretrain_config):
    """
    main pretraining loop.

    args:
        config: Base GRPO config
        model_config: Model loading config
        pretrain_config: pretraining-specific config (batch_size, lr, epochs...)
    """
    set_random_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model and tokenizer
    bnb_config = None
    if getattr(model_config, 'load_in_4bit', False):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    if "LLaDA" in config.model_path:
        dllm = AutoModel.from_pretrained(
            config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            token=token,
        ).to(device)
        config.mask_id = MASK_TOKENS_MAP["LLaDA"]
        config.model_type = "LLaDA"
    else:
        raise ValueError(f"Model path {config.model_path} not supported")

    # freeze base model
    dllm.eval()
    dllm = torch.compile(dllm)
    for param in dllm.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True, token=token)
    tokenizer.pad_token = tokenizer.eos_token

    # load dataset
    if config.dataset == "gsm8k":
        dataset = get_gsm8k_questions("train")
    elif config.dataset == "math":
        dataset = get_math_questions("train")
    else:
        raise ValueError(f"Dataset {config.dataset} not supported for pretraining")

    dataset = dataset.shuffle(seed=config.seed)

    # create policy
    if config.policy_type == "dit_hidden":
        policy_core = DiTHiddenStatePolicy(
            dllm=dllm,
            time_embed_dim=config.policy_time_embed_dim,
            num_blocks=config.policy_num_blocks,
            smart_init=config.policy_smart_init,
            time_period=config.policy_time_period,
            num_actions=config.num_policy_actions,
        ).to(device).to(torch.bfloat16)
    elif config.policy_type == "dit_confidence":
        hidden_dim = config.policy_hidden_dim or 128
        feedforward_dim = config.policy_feedforward_dim or (4 * hidden_dim)
        policy_core = DiTConfidencePolicy(
            hidden_dim=hidden_dim,
            feedforward_dim=feedforward_dim,
            num_heads=config.policy_num_heads,
            dropout=config.policy_dropout,
            time_embed_dim=config.policy_time_embed_dim,
            smart_init=config.policy_smart_init,
            confidences_top_p=config.confidences_top_p,
            num_blocks=config.policy_num_blocks,
            time_period=config.policy_time_period,
            num_actions=config.num_policy_actions,
        ).to(device).to(torch.bfloat16)
    elif config.policy_type == "dit_confidence_pcurrent":
        hidden_dim = config.policy_hidden_dim or 128
        feedforward_dim = config.policy_feedforward_dim or (4 * hidden_dim)
        policy_core = DiTConfidencePCurrentPolicy(
            hidden_dim=hidden_dim,
            feedforward_dim=feedforward_dim,
            num_heads=config.policy_num_heads,
            dropout=config.policy_dropout,
            time_embed_dim=config.policy_time_embed_dim,
            smart_init=config.policy_smart_init,
            confidences_top_p=config.confidences_top_p,
            num_blocks=config.policy_num_blocks,
            time_period=config.policy_time_period,
            num_actions=config.num_policy_actions,
        ).to(device).to(torch.bfloat16)
    else:
        raise ValueError(f"Policy type {config.policy_type} not supported")

    policy = PolicyHFWrapper(policy_core, config.policy_type)

    total_params = sum(p.numel() for p in policy_core.parameters())
    trainable_params = sum(p.numel() for p in policy_core.parameters() if p.requires_grad)
    print(f"Policy type: {config.policy_type}")
    print(f"Total policy parameters: {total_params:,}")
    print(f"Trainable policy parameters: {trainable_params:,}")

    # setup optimizer and scheduler
    optimizer = AdamW(policy_core.parameters(), lr=pretrain_config.learning_rate)
    num_epochs = pretrain_config.num_epochs
    num_batches_per_epoch = max(1, len(dataset) // pretrain_config.batch_size)
    total_steps = num_epochs * num_batches_per_epoch
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # setup wandb
    if pretrain_config.use_wandb:
        wandb_project = os.getenv("WANDB_PROJECT")
        if not wandb_project:
            raise ValueError("WANDB_PROJECT environment variable not set.")
        wandb_entity = os.getenv("WANDB_ENTITY")
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=f"pretrain-{config.policy_type}-{config.dataset}",
            config={
                "policy_type": config.policy_type,
                "dataset": config.dataset,
                "num_epochs": num_epochs,
                "batch_size": pretrain_config.batch_size,
                "learning_rate": pretrain_config.learning_rate,
                "mask_prob": pretrain_config.mask_prob,
                "corrupt_prob": pretrain_config.corrupt_prob,
            },
        )
        wandb.log({
            "policy/total_parameters": total_params,
            "policy/trainable_parameters": trainable_params,
        })

    # determine completion field name
    sample = dataset[0]
    if "completion" in sample:
        completion_field = "completion"
    elif "text" in sample:
        completion_field = "text"
    elif "answer" in sample:
        completion_field = "answer"
    else:
        raise ValueError(f"Unknown completion field in dataset. Keys: {sample.keys()}")

    # pretraining loop
    policy_core.train()
    global_step = 0
    checkpoint_interval = getattr(pretrain_config, 'checkpoint_interval', 1)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        epoch_metrics = {
            "remask_rate": 0.0,
            "unmask_rate": 0.0,
            "keep_rate": 0.0,
            "remask_conf_mean": 0.0,
            "unmask_conf_mean": 0.0,
            "keep_conf_mean": 0.0,
        }

        # shuffle dataset for each epoch
        shuffled_indices = torch.randperm(len(dataset)).tolist()

        for batch_start in range(0, len(dataset), pretrain_config.batch_size):
            batch_end = min(batch_start + pretrain_config.batch_size, len(dataset))
            batch_indices = shuffled_indices[batch_start:batch_end]
            batch_data = dataset[batch_indices]

            # tokenize completions
            if isinstance(batch_data[completion_field], str):
                completions = [batch_data[completion_field]]
            else:
                completions = batch_data[completion_field]

            tokens = tokenizer(
                completions,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"].to(device)
            B, L = input_ids.shape

            # generate corrupted/masked sequences
            with torch.no_grad():
                pretrain_batch = corrupt_and_mask_sequences(
                    input_ids,
                    dllm,
                    mask_id=config.mask_id,
                    mask_prob=pretrain_config.mask_prob,
                    corrupt_prob=pretrain_config.corrupt_prob,
                    exclusion_k=pretrain_config.exclusion_k,
                    top_p=pretrain_config.top_p,
                    temperature=pretrain_config.temperature,
                    confidence_threshold=pretrain_config.confidence_threshold,
                    unmask_reward_value=pretrain_config.unmask_reward,
                    remask_reward_value=pretrain_config.remask_reward,
                    num_actions=config.num_policy_actions,
                )

            confidences = pretrain_batch.confidences # (B, L)

            mask_tensor = pretrain_batch.masked_mask.long()  # (B, L)

            # call policy (depends on policy type/action space)
            if config.policy_type == "dit_confidence":
                # policy expects: m (B,L), c (B,L,confidences_top_p), timestep (B,1)
                policy_input_c = confidences.unsqueeze(-1)  # (B, L, 1)
                timestep = torch.zeros((B, 1), device=device)  # normalized to [0, 1]
                policy_logits = policy_core(mask_tensor, policy_input_c, timestep)
            elif config.policy_type == "dit_confidence_pcurrent":
                # 2-way policy expects separate confidence and p_current inputs
                if pretrain_batch.p_current_token is None:
                    raise RuntimeError(
                        "dit_confidence_pcurrent policy requires p_current_token, but it's None. "
                        "This happens when num_policy_actions != 1 (should be 1 for 2-way)."
                    )
                policy_input_c = confidences.unsqueeze(-1)  # (B, L, 1)
                p_current = pretrain_batch.p_current_token  # (B, L)
                timestep = torch.zeros((B, 1), device=device)
                policy_logits = policy_core(mask_tensor, policy_input_c, timestep, p_current)
            elif config.policy_type == "dit_hidden":
                # dit_hidden uses hidden states - needs full forward pass
                output = dllm(pretrain_batch.corrupted_sequences, output_hidden_states=True)
                hidden_states = output.hidden_states[-1]  # Last layer
                timestep = torch.zeros((B, 1), device=device)
                policy_logits = policy_core(mask_tensor, hidden_states, timestep)
            else:
                raise ValueError(f"Unknown policy type: {config.policy_type}")

            # compute batch metrics based on policy decisions
            batch_metrics = compute_pretrain_metrics(
                policy_logits,
                pretrain_batch.masked_mask,
                confidences,
                num_actions=config.num_policy_actions,
            )

            # compute loss
            loss = compute_pretrain_loss(
                policy_logits,
                pretrain_batch.masked_mask,
                pretrain_batch.corrupted_mask,
                pretrain_batch.unmask_reward,
                pretrain_batch.remask_reward,
                num_actions=config.num_policy_actions,
            )

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_core.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            # accumulate epoch metrics
            for key in epoch_metrics:
                epoch_metrics[key] += batch_metrics[key]

            if (batch_start // pretrain_config.batch_size + 1) % pretrain_config.log_interval == 0:
                avg_loss = epoch_loss / num_batches
                avg_remask_rate = batch_metrics["remask_rate"]
                avg_unmask_rate = batch_metrics["unmask_rate"]
                avg_keep_rate = batch_metrics["keep_rate"]
                avg_remask_conf = batch_metrics["remask_conf_mean"]
                avg_unmask_conf = batch_metrics["unmask_conf_mean"]
                avg_keep_conf = batch_metrics["keep_conf_mean"]

                print(
                    f"Epoch {epoch+1}/{num_epochs} | Batch {batch_start // pretrain_config.batch_size + 1} | "
                    f"Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}\n"
                    f"  Action Rates: Remask={avg_remask_rate:.3f} | Unmask={avg_unmask_rate:.3f} | Keep={avg_keep_rate:.3f}\n"
                    f"  Mean Conf: Remask={avg_remask_conf:.4f} | Unmask={avg_unmask_conf:.4f} | Keep={avg_keep_conf:.4f}"
                )
                if pretrain_config.use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/avg_loss": avg_loss,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/action_rate/remask": avg_remask_rate,
                        "train/action_rate/unmask": avg_unmask_rate,
                        "train/action_rate/keep": avg_keep_rate,
                        "train/action_conf/remask": avg_remask_conf,
                        "train/action_conf/unmask": avg_unmask_conf,
                        "train/action_conf/keep": avg_keep_conf,
                        "global_step": global_step,
                    })

        # compute epoch-level averages
        epoch_avg_loss = epoch_loss / num_batches
        epoch_avg_remask_rate = epoch_metrics["remask_rate"] / num_batches
        epoch_avg_unmask_rate = epoch_metrics["unmask_rate"] / num_batches
        epoch_avg_keep_rate = epoch_metrics["keep_rate"] / num_batches
        epoch_avg_remask_conf = epoch_metrics["remask_conf_mean"] / num_batches if num_batches > 0 else 0.0
        epoch_avg_unmask_conf = epoch_metrics["unmask_conf_mean"] / num_batches if num_batches > 0 else 0.0
        epoch_avg_keep_conf = epoch_metrics["keep_conf_mean"] / num_batches if num_batches > 0 else 0.0

        print(
            f"Epoch {epoch+1}/{num_epochs} completed | Avg Loss: {epoch_avg_loss:.4f}\n"
            f"  Action Rates: Remask={epoch_avg_remask_rate:.3f} | Unmask={epoch_avg_unmask_rate:.3f} | Keep={epoch_avg_keep_rate:.3f}\n"
            f"  Mean Conf: Remask={epoch_avg_remask_conf:.4f} | Unmask={epoch_avg_unmask_conf:.4f} | Keep={epoch_avg_keep_conf:.4f}"
        )
        if pretrain_config.use_wandb:
            wandb.log({
                "epoch/loss": epoch_avg_loss,
                "epoch/action_rate/remask": epoch_avg_remask_rate,
                "epoch/action_rate/unmask": epoch_avg_unmask_rate,
                "epoch/action_rate/keep": epoch_avg_keep_rate,
                "epoch/action_conf/remask": epoch_avg_remask_conf,
                "epoch/action_conf/unmask": epoch_avg_unmask_conf,
                "epoch/action_conf/keep": epoch_avg_keep_conf,
                "epoch": epoch + 1,
            })
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(pretrain_config.output_dir, f"checkpoint-epoch-{epoch+1}")
            os.makedirs(checkpoint_path, exist_ok=True)
            policy.save_pretrained(checkpoint_path)
            print(f"✓ Saved checkpoint to {checkpoint_path}")

    # save final checkpoint
    output_dir = pretrain_config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    policy.save_pretrained(output_dir)
    print(f"Pretraining complete. Policy saved to {output_dir}")

    if pretrain_config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    import argparse
    from argparse import Namespace

    parser = argparse.ArgumentParser(description="Pretrain a remasking policy")

    # config file
    parser.add_argument("--config", type=str, default="configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_2way_setstate.yaml",
                       help="Path to experiment config YAML file")

    # pretraining-specific overrides (can override config file)
    parser.add_argument("--dataset", type=str, default=None, help="Dataset: gsm8k or math")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--checkpoint-interval", type=int, default=1, help="Checkpoint interval")
    parser.add_argument("--output-dir", type=str, default="./pretrained_policy", help="Output directory")

    # corruption/masking parameters
    parser.add_argument("--mask-prob", type=float, default=0.4, help="Probability to mask")
    parser.add_argument("--corrupt-prob", type=float, default=0.2, help="Probability to corrupt")
    parser.add_argument("--exclusion-k", type=int, default=5, help="Top-k tokens to exclude")
    parser.add_argument("--top-p", type=float, default=0.70, help="Nucleus sampling p")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for corruption sampling")

    # reward parameters
    parser.add_argument("--confidence-threshold", type=float, default=0.9, help="Threshold for unmask reward")
    parser.add_argument("--unmask-reward", type=float, default=1.0, help="Unmask reward value")
    parser.add_argument("--remask-reward", type=float, default=1.0, help="Remask reward value")

    # logging
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false", default=True, help="Disable wandb logging")

    args = parser.parse_args()

    # Load policy config from YAML file using TrlParser
    from trl import TrlParser
    trl_parser = TrlParser((Config,))
    (config,) = trl_parser.parse_args_and_config(
        args=["--config", args.config], fail_with_unknown_args=False
    )

    # Override dataset if provided
    if args.dataset:
        config.dataset = args.dataset

    model_config = Namespace()

    pretrain_config = Namespace(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        mask_prob=args.mask_prob,
        corrupt_prob=args.corrupt_prob,
        exclusion_k=args.exclusion_k,
        top_p=args.top_p,
        temperature=args.temperature,
        confidence_threshold=args.confidence_threshold,
        unmask_reward=args.unmask_reward,
        remask_reward=args.remask_reward,
        checkpoint_interval=args.checkpoint_interval,
        output_dir=args.output_dir,
        log_interval=10,
        use_wandb=args.use_wandb,
    )

    main(config, model_config, pretrain_config)
