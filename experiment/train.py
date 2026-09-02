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
import socket
from torch.utils.data import DataLoader
import utils.dataset as dataset
from utils.config_validation import validate_training_config

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', 0))              # 端口传 0 = 由内核自动分配空闲端口
    vllm_port = s.getsockname()[1]
    
vllm_gpu_memory_utilization = 0.75
vllm_device = 3
training_device = "cuda:2"
seed = 42
learning_rate = 1e-5 
rollout_batch_size = 256 
group_size = 8 
num_rollout_steps = 200
off_policy_speedup_factor = 1
training_batch_size = rollout_batch_size // off_policy_speedup_factor
inference_batch_size = rollout_batch_size//group_size
gradient_accumulation_steps = 32 //off_policy_speedup_factor
val_batch_size = 64
sampling_temperature = 1.0 
sampling_max_tokens = 512 
max_grad_norm = 1.0 
model_name = "allenai/OLMo-2-0425-1B"
advantage_algorithm = "GRPO"

helper.seed_everything(seed=seed)

constant_normalization = rollout_batch_size * sampling_max_tokens
ADVANTAGE_ALGORITHM_CONFIG = {
    "GRPO": {
        "baseline": "mean",
        "advantage_normalizer": "std",
        "loss_normalization": "sequence",
        "normalization_constant": None,
        "importance_reweighting_method":'none',
        "cliprange":None
    },
    "GRPO_constant": {
        "baseline": "mean",
        "advantage_normalizer": "std",
        "loss_normalization": "constant",
        "normalization_constant": constant_normalization,
        "importance_reweighting_method":'none',
        "cliprange":None
    },
    "Dr_GRPO": {
        "baseline": "mean",
        "advantage_normalizer": "none",
        "loss_normalization": "constant",
        "normalization_constant": constant_normalization,
        "importance_reweighting_method":'none',
        "cliprange":None
    },
    "RFT": {
        "baseline": "none",
        "advantage_normalizer": "none",
        "loss_normalization": "constant",
        "normalization_constant": constant_normalization,
        "importance_reweighting_method":'none',
        "cliprange":None
    },
    "MaxRL": {
        "baseline": "mean",
        "advantage_normalizer": "mean",
        "loss_normalization": "constant",
        "normalization_constant": constant_normalization,
        "importance_reweighting_method":'none',
        "cliprange":None
    },
    "offpolicy_naive":{
        "baseline": "mean",
        "advantage_normalizer": "std",
        "loss_normalization": "sequence",
        "normalization_constant": None,
        "importance_reweighting_method":"none",
        "cliprange":None
    },
    "offpolicy_noclip":{
        "baseline": "mean",
        "advantage_normalizer": "std",
        "loss_normalization": "sequence",
        "normalization_constant": None,
        "importance_reweighting_method":"noclip",
        "cliprange":None
    },
    "offpolicy_grpo" : {
        "baseline": "mean",
        "advantage_normalizer": "std",
        "loss_normalization": "sequence",
        "normalization_constant": None,
        "importance_reweighting_method":"grpo",
        "cliprange":0.2
    },
    "offpolicy_gspo":{
        "baseline": "mean",
        "advantage_normalizer": "std",
        "loss_normalization": "sequence",
        "normalization_constant": None,
        "importance_reweighting_method":"gspo",
        "cliprange":3e-4
    },
}

try:
    config = ADVANTAGE_ALGORITHM_CONFIG[advantage_algorithm]
except KeyError:
    supported_algorithms = ", ".join(ADVANTAGE_ALGORITHM_CONFIG)
    raise NotImplementedError(
        f"Unsupported advantage algorithm: {advantage_algorithm!r}. "
        f"Supported algorithms: {supported_algorithms}."
    ) from None

validate_training_config(
    learning_rate=learning_rate,
    rollout_batch_size=rollout_batch_size,
    group_size=group_size,
    inference_batch_size=inference_batch_size,
    off_policy_speedup_factor=off_policy_speedup_factor,
    training_batch_size=training_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_training_steps=num_rollout_steps,
    val_batch_size=val_batch_size,
    sampling_temperature=sampling_temperature,
    sampling_max_tokens=sampling_max_tokens,
    max_grad_norm=max_grad_norm,
    vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
    advantage_algorithm=advantage_algorithm,
    advantage_config=config,
)
    
(
    baseline,
    advantage_normalizer,
    loss_normalization,
    normalization_constant,
    importance_reweighting_method,
    cliprange,
) = (
    config["baseline"],
    config["advantage_normalizer"],
    config["loss_normalization"],
    config["normalization_constant"],
    config["importance_reweighting_method"],
    config["cliprange"],
)

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
        }
    )
server = vllm.VLLMServer(model_id=model_name,
                         port=vllm_port,
                         gpu=vllm_device,
                         seed = seed,
                         gpu_memory_utilization=vllm_gpu_memory_utilization,
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

model,tokenizer = helper.get_model_and_tokenizer(model_id_or_dir=model_name,
                               device=training_device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = learning_rate,
    betas = (0.9, 0.95),
    weight_decay=0.0
)

prompts = []
groundtruths = []

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    prompt_template = f.read()

with open(TRAIN_SET_PATH, "r", encoding="utf-8") as f:
    for line in f :
        data = json.loads(line)
        prompt = prompt_template.format(question=data["question"])
        prompts.append(prompt)
        groundtruths.append(helper.extract_ground_truth(data["answer"]))

val_prompts, val_groundtruths = [], []
with open(VALID_SET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        val_prompts.append(prompt_template.format(question=data["question"]))
        val_groundtruths.append(helper.extract_ground_truth(data["answer"]))

valdata = dataset.inferencedataset(
    prompts = val_prompts,
    gts = val_groundtruths
)
valdataloader = DataLoader(
    valdata,
    batch_size = val_batch_size,
)

weight_sync_group = vllm.init_weight_sync(
    vllm_base_url=server.base_url,
    policy_device=training_device
)

grad_clip_history: list[float] = []   # C: 近 50 步的裁剪触发率窗口
best_val_reward = 0.0
data = dataset.inferencedataset(
    prompts = prompts,
    gts = groundtruths
)
inferencedatalodaer = DataLoader(
    data,
    batch_size=inference_batch_size,
    shuffle=True
)


inference_data_iterator = iter(inferencedatalodaer)
for step in range(num_rollout_steps):
    try:
        batch_prompts,batch_gts = next(inference_data_iterator)
    except StopIteration:
        inference_data_iterator = iter(inferencedatalodaer)
        batch_prompts,batch_gts = next(inference_data_iterator)
    #inference
    repeated_prompts = []
    rollout_responses = []
    repeated_ground_truths = []
    
    rollout_start_time = time.time()
    batch_completions = server.generate_completions(
            prompts=batch_prompts,
            sampling_params=sampling_params,
            batch_size=inference_batch_size,
        )
    rollout_seconds = time.time() - rollout_start_time

    for j, completion in enumerate(batch_completions):
        prompt_idx = j // group_size
        repeated_prompts.append(batch_prompts[prompt_idx])
        rollout_responses.append(completion.text)
        repeated_ground_truths.append(batch_gts[prompt_idx])
    #inference finish
    
    #train
    rolloutdata = dataset.trainingdataset(
        repeated_prompts=repeated_prompts,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths)
    trainingdataloader = DataLoader(
        rolloutdata,
        batch_size = training_batch_size,
        )
    old_log_probs_batches = []
    if importance_reweighting_method != "none":
        for training_prompts,training_responses,_ in trainingdataloader:
            ids = helper.tokenize_prompt_and_output(
                prompt_strs=training_prompts,
                output_strs=training_responses,
                tokenizer = tokenizer
            )
            input_ids = ids["input_ids"]
            labels = ids["labels"]
            with torch.no_grad():
                old_log_probs = helper.get_response_log_probs(
                    model=model,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=False,
                )["log_probs"]
                old_log_probs_batches.append(
                    helper.PackedResponseLogProbs.from_padded(
                        log_probs=old_log_probs,
                        response_mask=ids["response_mask"],
                    )
                )
                del old_log_probs
    else:
        old_log_probs_batches = [None for i in range(0,len(trainingdataloader))]
    
    for ((training_prompts,training_responses,training_gt),old_log_probs) in zip(trainingdataloader, old_log_probs_batches):
        train_start_time = time.time()
        loss,metadata = helper.grpo_train_step(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            reward_fn=grader.r1_zero_reward_fn,
            repeated_prompts=training_prompts,
            rollout_responses=training_responses,
            repeated_ground_truths=training_gt,
            group_size=group_size,
            baseline = baseline,
            advantage_normalizer=advantage_normalizer,
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=old_log_probs,
            cliprange = cliprange,
            loss_normalization = loss_normalization,
            normalization_constant=normalization_constant
        )
        
        train_seconds = time.time() - train_start_time
        #train finish
        #log metrics
        train_rewards_total = metadata.get("total_reward", 0.0)
        train_rewards_format = metadata.get("total_format_reward", 0.0)
        train_rewards_answer = metadata.get("total_answer_reward", 0.0)
        train_avg_response_length = sum(len(r.split()) for r in rollout_responses) / len(rollout_responses) if rollout_responses else 0.0

        # C: 梯度裁剪触发率 (clip_grad_norm_ 返回裁剪前的 norm)
        grad_norm = metadata.get("gradient_norm", 0.0)
        grad_norm = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm) # type: ignore
        grad_clip_history.append(1.0 if grad_norm > max_grad_norm else 0.0)
        grad_clip_history = grad_clip_history[-50:]

        # D: token 口径的长度与截断率 (直接用 vllm 返回的 token_ids / finish_reason)
        n_completions = len(batch_completions)
        response_token_lens = [len(c.token_ids) for c in batch_completions]
        truncated_frac = sum(
            1.0 for c in batch_completions if c.finish_reason == "length"
        ) / n_completions if n_completions else 0.0

        offpolicy_metrics = {"offpolicy/enabled": importance_reweighting_method != "none"}
        if importance_reweighting_method != "none":
            available_offpolicy_metrics = {
                "offpolicy/ratio_mean": metadata["importance_ratio_mean"],
                "offpolicy/ratio_std": metadata["importance_ratio_std"],
                "offpolicy/ratio_min": metadata["importance_ratio_min"],
                "offpolicy/ratio_max": metadata["importance_ratio_max"],
                "offpolicy/approx_kl": metadata["importance_approx_kl"],
                "offpolicy/clip_fraction": metadata["importance_clip_fraction"],
                "offpolicy/ess": metadata["importance_ess"],
                "offpolicy/ess_fraction": metadata["importance_ess_fraction"],
                "offpolicy/sample_count": metadata["importance_sample_count"],
            }
            offpolicy_metrics.update(
                {
                    key: value
                    for key, value in available_offpolicy_metrics.items()
                    if value is not None
                }
            )

        wandb.log({
            # A: 核心进展
            "train/loss": loss.item() if torch.is_tensor(loss) else loss,
            "train/reward_mean": metadata.get("mean_reward", 0.0),
            "train/answer_reward_mean": metadata.get("mean_answer_reward", 0.0),
            "train/format_reward_mean": metadata.get("mean_format_reward", 0.0),
            "train/pass@1": metadata.get("pass@1", 0.0),
            "train/pass@group_size": metadata.get("pass@group_size", 0.0),
            "train/rewards_total": train_rewards_total,
            "train/rewards_answer": train_rewards_answer,
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
            **offpolicy_metrics,
        }, step=step)
        if (step+1) % 10 == 0:
            vllm.sync_policy_weights(
                policy = model,
                vllm_base_url=server.base_url,
                weight_sync_group=weight_sync_group
            )
            val_rewards_total, val_rewards_format, val_response_lengths = 0.0, 0.0, []
            val_token_lens, val_truncated = [], 0
            val_batch_size = 64
            val_sampling_params = sampling_params.copy()
            val_sampling_params["n"] = 1
            val_sampling_params["temperature"] = 0.0  
            
            n_val = len(val_prompts)
            for batch_val_prompts,batch_val_gts in valdataloader:
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
                "val/accuracy": val_rewards_total / n_val if n_val else 0.0,
                "val/format_rate": val_rewards_format / n_val if n_val else 0.0,
                "val/rewards_total": val_rewards_total / n_val if n_val else 0.0,
                "val/rewards_format": val_rewards_format / n_val if n_val else 0.0,
                # D
                "val/response_tokens_mean": sum(val_token_lens) / len(val_token_lens) if val_token_lens else 0.0,
                "val/truncated_frac": val_truncated / len(val_token_lens) if val_token_lens else 0.0,
                "val/avg_response_length": sum(val_response_lengths) / len(val_response_lengths) if val_response_lengths else 0.0,
            }, step=step)
        step+=1    
    vllm.sync_policy_weights(
        policy = model,
        vllm_base_url=server.base_url,
        weight_sync_group=weight_sync_group
    )
tokenizer.save_pretrained(CKPT_PATH)

server.stop()
wandb.finish()
