import utils.helper as helper
import torch
from pathlib import Path
import utils.vllm_utils as vllm
import utils.grader as grader
import json
import wandb
import random
import time
import os
import shutil
vllm_device = 3
training_device = "cuda:2"
seed = 42
n_train_examples = 6400 
n_val_examples = 1024 
num_rollout_steps = 200 
learning_rate = 1e-5 
rollout_batch_size = train_batch_size = 256 
group_size = 8 
batch_size = rollout_batch_size//group_size
gradient_accumulation_steps = 32 
sampling_temperature = 1.0 
sampling_max_tokens = 512 
max_grad_norm = 1.0 
model_name = "allenai/OLMo-2-0425-1B"
advantage_algorithm = "GRPO"

helper.seed_everything(seed=seed)

if advantage_algorithm == "GRPO":
    baseline = "mean"
    advantage_normalizer = "std"
    loss_normalization = "sequence"
    normalization_constant = None
elif advantage_algorithm == "GRPO_constant":
    baseline = "mean"
    advantage_normalizer = "std"
    loss_normalization = "constant"
    normalization_constant = rollout_batch_size*sampling_max_tokens
elif advantage_algorithm == "Dr_GRPO":
    baseline = "mean"
    advantage_normalizer = "none"
    loss_normalization = "constant"
    normalization_constant = rollout_batch_size*sampling_max_tokens
elif advantage_algorithm == "RFT":
    baseline = "none"
    advantage_normalizer = "none"
    loss_normalization = "constant"
    normalization_constant = rollout_batch_size*sampling_max_tokens
elif advantage_algorithm == "MaxRL":
    baseline = "mean"
    advantage_normalizer = "mean"
    loss_normalization = "constant"
    normalization_constant = rollout_batch_size*sampling_max_tokens

else:
    raise NotImplementedError
    

wandb_project = "RL"
wandb_run_name = f"{advantage_algorithm}-{model_name.replace('/', '-')}-seed{seed}"
# 关键：添加 group 参数，将 4 个不同的 seed 运行归为同一组
# 这样 WandB 网页端就能自动对它们进行聚合和对比展示
wandb_group = "experiment-1"
wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        group=wandb_group,  
        config={
            "model_name": model_name,
            "learning_rate": learning_rate,
            "num_rollout_steps": num_rollout_steps,
            "rollout_batch_size": rollout_batch_size,
            "group_size": group_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "sampling_temperature": sampling_temperature,
            "sampling_max_tokens": sampling_max_tokens,
            "max_grad_norm": max_grad_norm,
            "seed": seed,
            "n_train_examples": n_train_examples,
            "n_val_examples": n_val_examples,
        }
    )
server = vllm.VLLMServer(model_id=model_name,
                         gpu=vllm_device,
                         seed = seed,
                         gpu_memory_utilization=0.75,
                         )
server.start()
sampling_params = {}
sampling_params['stop'] = ["</answer>"] 
sampling_params['include_stop_str_in_output'] = True
sampling_params["temperature"] = sampling_temperature
sampling_params["max_tokens"] = sampling_max_tokens
sampling_params["n"] = group_size
sampling_params["seed"] = seed

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "r1_zero.prompt"
TRAIN_SET_PATH = Path(__file__).resolve().parent.parent / "data" / "gsm8k" / "train.jsonl"
VALID_SET_PATH = Path(__file__).resolve().parent.parent / "data" / "gsm8k" / "test.jsonl"
CKPT_PATH = Path(__file__).resolve().parent.parent / "results" / "checkpoint" / f"{wandb_run_name}"




prompts = []
groundtruths = []


model,tokenizer = helper.get_model_and_tokenizer(model_id_or_dir=model_name,
                               device=training_device)
tokenizer.save_pretrained(CKPT_PATH)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = learning_rate,
    betas = (0.9, 0.95),
    weight_decay=0.0
)

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    prompt_template = f.read()
    

with open(TRAIN_SET_PATH, "r", encoding="utf-8") as f:
    train_examples = [json.loads(line) for line in f if line.strip()]

if not train_examples:
    raise ValueError(f"No training examples found in {TRAIN_SET_PATH}")

# The local dataset can be smaller than n_train_examples. Sample complete,
# independently shuffled passes over it until the requested size is reached.
sampling_rng = random.Random(seed)
while len(prompts) < n_train_examples:
    epoch_examples = train_examples.copy()
    sampling_rng.shuffle(epoch_examples)
    remaining = n_train_examples - len(prompts)
    for data in epoch_examples[:remaining]:
        prompt = prompt_template.format(question=data["question"])
        prompts.append(prompt)
        groundtruths.append(helper.extract_ground_truth(data["answer"]))

val_prompts, val_groundtruths = [], []
with open(VALID_SET_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= n_val_examples:
            break
        data = json.loads(line)
        val_prompts.append(prompt_template.format(question=data["question"]))
        val_groundtruths.append(helper.extract_ground_truth(data["answer"]))

weight_sync_group = vllm.init_weight_sync(
    vllm_base_url=server.base_url,
    policy_device=training_device
)

indices = list(range(len(prompts)))
rng = random.Random(seed)
rng.shuffle(indices)

grad_clip_history: list[float] = []   # C: 近 50 步的裁剪触发率窗口
best_val_reward = 0.0
for step in range(0,num_rollout_steps):
    repeated_prompts = []
    rollout_responses = []
    repeated_ground_truths = []
    
    start = step * batch_size
    end = start + batch_size
    batch_indices = indices[start:end]
    batch_prompts = [prompts[i] for i in batch_indices]
    batch_gts = [groundtruths[i] for i in batch_indices]
    rollout_start_time = time.time()
    batch_completions = server.generate_completions(
            prompts=batch_prompts,
            sampling_params=sampling_params,
            batch_size=batch_size,
        )
    rollout_seconds = time.time() - rollout_start_time

    for j, completion in enumerate(batch_completions):
        prompt_idx = j // group_size
        repeated_prompts.append(batch_prompts[prompt_idx])
        rollout_responses.append(completion.text)
        repeated_ground_truths.append(batch_gts[prompt_idx])
    
    train_start_time = time.time()
    loss,metadata = helper.grpo_train_step(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        reward_fn=grader.r1_zero_reward_fn,
        repeated_prompts=repeated_prompts,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
        group_size=group_size,
        baseline = baseline,
        advantage_normalizer=advantage_normalizer,
        loss_normalization = loss_normalization,
        normalization_constant=normalization_constant
    )
    train_seconds = time.time() - train_start_time
    vllm.sync_policy_weights(
        policy = model,
        vllm_base_url=server.base_url,
        weight_sync_group=weight_sync_group
    )
    train_rewards = metadata["train_rewards"]
    train_rewards_total = train_rewards[0].item() # type: ignore
    train_rewards_format = train_rewards[1].item() # type: ignore
    train_avg_response_length = sum(len(r.split()) for r in rollout_responses) / len(rollout_responses) if rollout_responses else 0.0

    # C: 梯度裁剪触发率 (clip_grad_norm_ 返回裁剪前的 norm)
    grad_norm = metadata.get("gradient_norm", 0.0)
    grad_norm = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
    grad_clip_history.append(1.0 if grad_norm > max_grad_norm else 0.0)
    grad_clip_history = grad_clip_history[-50:]

    # D: token 口径的长度与截断率 (直接用 vllm 返回的 token_ids / finish_reason)
    n_completions = len(batch_completions)
    response_token_lens = [len(c.token_ids) for c in batch_completions]
    truncated_frac = sum(
        1.0 for c in batch_completions if c.finish_reason == "length"
    ) / n_completions if n_completions else 0.0

    wandb.log({
        # A: 核心进展
        "train/loss": loss.item() if torch.is_tensor(loss) else loss,
        "train/reward_mean": metadata.get("mean_reward", 0.0),
        "train/format_reward_mean": metadata.get("mean_format_reward", 0.0),
        "train/rewards_total": train_rewards_total,
        "train/rewards_format": train_rewards_format,
        # B: 裁剪可观测性
        "train/pruned_frac": metadata.get("pruned_frac", 0.0),
        "train/num_pruned_rollout": metadata.get("num_pruned_rollout", 0),
        "train/degenerate_groups": metadata.get("degenerate_groups", 0),
        "train/all_correct_groups": metadata.get("all_correct_groups", 0),
        "train/all_wrong_groups": metadata.get("all_wrong_groups", 0),
        # C: 优化健康度
        "train/gradient_norm": grad_norm,
        "train/grad_clip_frac": sum(grad_clip_history) / len(grad_clip_history),
        "train/token_entropy": metadata.get("token_entropy", 0.0),
        "train/advantage_abs_mean": metadata.get("advantage_abs_mean", 0.0),
        "train/advantage_std": metadata.get("advantage_std", 0.0),
        "train/lr": optimizer.param_groups[0]["lr"],
        # D: 长度与截断
        "train/response_tokens_mean": sum(response_token_lens) / n_completions if n_completions else 0.0,
        "train/truncated_frac": truncated_frac,
        "train/empty_response_frac": metadata.get("empty_response_frac", 0.0),
        "train/avg_response_length": train_avg_response_length,
        # 耗时
        "time/rollout_seconds": rollout_seconds,
        "time/train_seconds": train_seconds,
    }, step=step)
    if step > 0 and (step+1) % 10 == 0:
        val_rewards_total, val_rewards_format, val_response_lengths = 0.0, 0.0, []
        val_token_lens, val_truncated = [], 0
        val_batch_size = 64
        val_sampling_params = sampling_params.copy()
        val_sampling_params["n"] = 1
        val_sampling_params["temperature"] = 0.0  
        
        for i in range(0, n_val_examples, val_batch_size):
            batch_val_prompts = val_prompts[i:i+val_batch_size]
            batch_val_gts = val_groundtruths[i:i+val_batch_size]
            batch_val_completions = server.generate_completions(
                prompts=batch_val_prompts, sampling_params=val_sampling_params, batch_size=len(batch_val_prompts))
            batch_val_responses = [c.text for c in batch_val_completions]
            val_response_lengths.extend([len(r.split()) for r in batch_val_responses])
            val_token_lens.extend([len(c.token_ids) for c in batch_val_completions])
            val_truncated += sum(1 for c in batch_val_completions if c.finish_reason == "length")
            _,rewards = helper.compute_rollout_rewards(
                reward_fn = grader.r1_zero_reward_fn,
                rollout_responses=batch_val_responses,
                repeated_ground_truths=batch_val_gts
            )
            val_rewards_total += rewards.get('total_reward', 0.0)
            val_rewards_format += rewards.get('total_format_reward', 0.0)
            if val_rewards_total > best_val_reward:
                if os.path.exists(CKPT_PATH):
                    shutil.rmtree(CKPT_PATH)
                os.makedirs(CKPT_PATH,exist_ok=True)
                model.save_pretrained(CKPT_PATH,safe_serialization = True)
                best_val_reward = val_rewards_total
        wandb.log({
            # A: val 才是最终指标
            "val/accuracy": val_rewards_total / n_val_examples,
            "val/format_rate": val_rewards_format / n_val_examples,
            "val/rewards_total": val_rewards_total / n_val_examples,
            "val/rewards_format": val_rewards_format / n_val_examples,
            # D
            "val/response_tokens_mean": sum(val_token_lens) / len(val_token_lens) if val_token_lens else 0.0,
            "val/truncated_frac": val_truncated / len(val_token_lens) if val_token_lens else 0.0,
            "val/avg_response_length": sum(val_response_lengths) / len(val_response_lengths) if val_response_lengths else 0.0,
        }, step=step)
        
server.stop()
wandb.finish()
