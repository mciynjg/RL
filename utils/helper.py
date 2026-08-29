import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer,PreTrainedTokenizer
from transformers import PreTrainedTokenizerBase,PreTrainedModel
from typing import Callable,Literal,List,Tuple
import random
import re
import os
import numpy as np
def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # cuDNN 设置
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # 使用确定性算法；遇到不支持的算子时直接报错
    torch.use_deterministic_algorithms(True)
      
def extract_ground_truth(answer: str) -> str|None:
    match = re.search(r"####\s*(.*)", answer)
    if match is None:
        return None
    return match.group(1).strip()

def get_model_and_tokenizer(model_id_or_dir: str, device: str):  
    model = AutoModelForCausalLM.from_pretrained(  
        model_id_or_dir, 
        device_map=device, 
        dtype=torch.bfloat16, 
        attn_implementation="eager" if device=='cpu' else "flash_attention_2", 
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
    
    return model, tokenizer
    
def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:

    prompt_ids = tokenizer(
        prompt_strs,
        add_special_tokens=False,
        padding=False,
    )["input_ids"]

    output_ids = tokenizer(
        output_strs,
        add_special_tokens=False,
        padding=False,
    )["input_ids"]

    ids = [
        torch.tensor(p+o)
        for p, o in zip(prompt_ids, output_ids)
    ]

    prompt_and_output_lens = [len(x) for x in ids]
    max_len = max(prompt_and_output_lens)
    padded_ids = torch.full(
        (len(ids), max_len),
        0,
        dtype=torch.long,
    )

    for i, x in enumerate(ids):
        padded_ids[i, :len(x)] = x

    input_ids = padded_ids[:, :-1]
    labels = padded_ids[:, 1:]
    response_mask = torch.zeros(
        (len(ids),max_len-1),
        dtype=torch.bool
    )
    
    for i, (p, o, x) in enumerate(zip(prompt_ids, output_ids, padded_ids)):
        response_start = len(p)-1
        response_end = len(p)+len(o)-1
        response_mask[i, response_start:response_end] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }
    
def get_response_log_probs(
    model: PreTrainedModel|torch.nn.Module, 
    input_ids: torch.Tensor, 
    labels: torch.Tensor, 
    return_token_entropy: bool = False, 
    ) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    all_log_probs = torch.log_softmax(logits,dim=-1)
    log_probs = all_log_probs.gather(dim = -1,index=labels.unsqueeze(-1)).squeeze(-1)
    result = {
        "log_probs":log_probs,
    }
    if return_token_entropy:
        with torch.no_grad():
            token_entropy = -(all_log_probs.exp()*all_log_probs).sum(dim = -1)
            result["token_entropy"] = token_entropy
    return result


def compute_rollout_rewards(  
    reward_fn: Callable[[str, str], dict[str, float]], 
    rollout_responses: list[str], 
    repeated_ground_truths: list[str], 
    ) -> tuple[torch.Tensor, dict[str, float]]:
    raw_rewards = []
    total_reward = 0.0
    mean_reward = 0.0
    total_format_reward = 0.0
    mean_format_reward = 0.0
    for response,ground_truth in zip(rollout_responses,repeated_ground_truths):
        reward = reward_fn(response,ground_truth)
        raw_rewards.append(reward["reward"])
        total_format_reward +=reward["format_reward"]
    total_reward = sum(raw_rewards)
    mean_reward = total_reward/len(rollout_responses)
    mean_format_reward = total_format_reward/len(rollout_responses)
    metadata = {
        "total_reward":total_reward,
        "mean_reward":mean_reward,
        "total_format_reward":total_format_reward,
        "mean_format_reward":mean_format_reward,
    }
    raw_rewards = torch.tensor(raw_rewards)
    return (raw_rewards,metadata)

def compute_group_normalized_rewards(  
    raw_rewards: torch.Tensor, 
    group_size: int, 
    baseline: Literal["mean", "none"] = "mean", 
    advantage_eps: float = 1e-6, 
    advantage_normalizer: Literal["std", "none", "mean"] = "std", 
    ):
    
    
    if baseline == "mean":
        raw_rewards = raw_rewards.reshape(-1,group_size)
        b = torch.mean(raw_rewards,dim=-1,keepdim=True)
    else:
        b = 0.0
        
    if advantage_normalizer == "std":
        factor = torch.std(raw_rewards,dim=-1,keepdim=True)
    elif advantage_normalizer == "none":
        factor = 1.0
    else:
        factor = torch.mean(raw_rewards,dim=-1,keepdim=True)
    
    advantage = ((raw_rewards-b)/(advantage_eps+factor)).flatten()
    avg_reward = torch.mean(raw_rewards).item()
    std_reward = torch.std(raw_rewards).item()
    max_reward = raw_rewards.max().item()
    min_reward = raw_rewards.min().item()
    metadata = {
        "avg_reward":avg_reward,
        "std_reward":std_reward,
        "max_reward":max_reward,
        "min_reward":min_reward,
    }
    return (advantage,metadata)

def compute_policy_gradient_loss( 
    raw_rewards_or_advantages: torch.Tensor, 
    policy_log_probs: torch.Tensor, 
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none", 
    old_log_probs: torch.Tensor | None = None, 
    cliprange: float | None = None, 
    response_mask: torch.Tensor | None = None, 
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if importance_reweighting_method == "none":
        per_token_policy_gradient_loss = -raw_rewards_or_advantages*policy_log_probs
    else:
        raise NotImplementedError
    metadata = {}
    return (per_token_policy_gradient_loss,metadata)

def aggregate_loss_across_microbatch(  
    per_token_policy_gradient_loss: torch.Tensor, 
    mask: torch.Tensor, 
    loss_normalization: Literal["sequence", "constant"] = "sequence", 
    normalization_constant: int | None = None, 
    ) -> torch.Tensor:
    if loss_normalization == "sequence":
        seq_lens = mask.sum(dim=-1)
        per_token_policy_gradient_loss = per_token_policy_gradient_loss*mask
        loss = per_token_policy_gradient_loss.sum(dim=-1)/seq_lens
        loss = loss.mean()
    else:
        per_token_policy_gradient_loss = per_token_policy_gradient_loss*mask
        if normalization_constant:
            loss = per_token_policy_gradient_loss.sum(dim=-1)/normalization_constant
        else:
            loss = per_token_policy_gradient_loss.sum(dim=-1)
        loss = loss.sum(dim=-1)
    
    return loss

def grpo_train_step(  
    model: PreTrainedModel|torch.nn.Module, 
    tokenizer: PreTrainedTokenizerBase, 
    optimizer: torch.optim.Optimizer, 
    gradient_accumulation_steps: int, 
    max_grad_norm: float|None, 
    reward_fn: Callable[[str, str], dict[str, float]], 
    repeated_prompts: list[str], 
    rollout_responses: list[str], 
    repeated_ground_truths: list[str], 
    group_size: int, # Reward normalization 
    baseline: Literal["mean", "none"] = "mean", 
    advantage_eps: float = 1e-6, 
    advantage_normalizer: Literal["std", "none", "mean"] = "std", 
    # Importance reweighting and clipping 
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none", 
    old_log_probs: torch.Tensor | None = None, cliprange: float | None = None, 
    # Loss normalization 
    loss_normalization: Literal["sequence", "constant"] = "sequence", 
    normalization_constant: int | None = None, 
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    
    optimizer.zero_grad()
    
    device = next(model.parameters()).device
    metadata = {
        "loss":torch.zeros(1).to(device),
        "gradient_norm":0.0,
        "token_entropy":0.0,
        "train_rewards":torch.zeros(2),
        "total_valid_tokens":0.0,
        # A: 核心进展 (按 rollout 数归一化, 跨配置可比)
        "mean_reward":0.0,
        "mean_format_reward":0.0,
        # B: 裁剪可观测性
        "num_rollout":0,
        "num_pruned_rollout":0,
        "pruned_frac":0.0,
        "degenerate_groups":0,
        "all_correct_groups":0,
        "all_wrong_groups":0,
        # C: 优化健康度
        "advantage_abs_mean":0.0,
        "advantage_std":0.0,
        # D: 长度
        "response_tokens_mean":0.0,
        "empty_response_frac":0.0,
    }
    
    tokens = tokenize_prompt_and_output(prompt_strs=repeated_prompts,
                                     output_strs=rollout_responses,
                                     tokenizer=tokenizer)
    
    
    inputs = tokens["input_ids"]#.to(device)
    labels = tokens["labels"]#.to(device)
    response_mask = tokens["response_mask"]#.to(device)
    num_rollout = len(repeated_prompts)
    raw_rewards,raw_rewards_metadata = compute_rollout_rewards(
                reward_fn=reward_fn,
                rollout_responses=rollout_responses,
                repeated_ground_truths=repeated_ground_truths
                )
    
    total_rewards = raw_rewards_metadata["total_reward"]
    total_format_rewards = raw_rewards_metadata["total_format_reward"]
    metadata["train_rewards"][0]+=total_rewards
    metadata["train_rewards"][1]+=total_format_rewards
    metadata["mean_reward"] = raw_rewards_metadata["mean_reward"]
    metadata["mean_format_reward"] = raw_rewards_metadata["mean_format_reward"]

    # B: 组级退化统计 (与 baseline 取值无关, 始终按 group_size 分组统计)
    grouped_for_stats = raw_rewards.reshape(-1,group_size)
    group_is_degenerate = grouped_for_stats.std(dim=-1) == 0.0
    metadata["degenerate_groups"] = int(group_is_degenerate.sum().item())
    metadata["all_correct_groups"] = int((group_is_degenerate & (grouped_for_stats.mean(dim=-1) > 0)).sum().item())
    metadata["all_wrong_groups"] = int((group_is_degenerate & (grouped_for_stats.mean(dim=-1) == 0)).sum().item())

    # D: 响应长度 / 空响应 (裁剪前的全量口径)
    metadata["response_tokens_mean"] = response_mask.sum(dim=-1).float().mean().item()
    metadata["empty_response_frac"] = (response_mask.sum(dim=-1) == 0).float().mean().item()

    if baseline == "mean":
        grouped_raw_rewards = raw_rewards.reshape(-1,group_size)
        std = torch.std(grouped_raw_rewards,dim=-1)
        mask = std!=0.0
        mask = mask.repeat_interleave(group_size)
        pruned_raw_rewards = raw_rewards[mask]

    if baseline == "none":
        mask = raw_rewards!=0.0
        pruned_raw_rewards = raw_rewards[mask]
    num_pruned_rollout = len(pruned_raw_rewards)

    if num_pruned_rollout == 0:
        return (metadata["loss"],metadata)
    inputs = inputs[mask]
    labels = labels[mask]
    response_mask = response_mask[mask]
    group_normalized_rewards,_ = compute_group_normalized_rewards(
                raw_rewards=pruned_raw_rewards,
                group_size=group_size,
                baseline=baseline,
                advantage_eps=advantage_eps,
                advantage_normalizer=advantage_normalizer,
                )
    group_normalized_rewards = group_normalized_rewards.unsqueeze(-1)

    # B/C: 裁剪比例与 advantage 尺度
    metadata["num_rollout"] = num_rollout
    metadata["num_pruned_rollout"] = num_pruned_rollout
    metadata["pruned_frac"] = 1.0 - num_pruned_rollout/num_rollout if num_rollout else 0.0
    if num_pruned_rollout > 0:
        metadata["advantage_abs_mean"] = group_normalized_rewards.abs().mean().item()
        metadata["advantage_std"] = group_normalized_rewards.std().item() if num_pruned_rollout > 1 else 0.0

    microbatch_size = min(num_rollout//gradient_accumulation_steps,num_pruned_rollout)
    for i in range(0,num_pruned_rollout,microbatch_size):
        #划分microbatch
        inputs_microbatch = inputs[i:i+microbatch_size].to(device)
        labels_microbatch = labels[i:i+microbatch_size].to(device)
        response_mask_microbatch = response_mask[i:i+microbatch_size].to(device)
        group_normalized_rewards_microbatch = group_normalized_rewards[i:i+microbatch_size].to(device)
        log_probs_and_token_entropy= get_response_log_probs(
            model=model,
            input_ids=inputs_microbatch,
            labels=labels_microbatch,
            return_token_entropy=True
        )
        log_probs = log_probs_and_token_entropy["log_probs"]
        
        token_entropy = log_probs_and_token_entropy["token_entropy"]
        masked_entropy = token_entropy * response_mask_microbatch  # 将非 response 位置置零
        valid_tokens_in_microbatch = response_mask_microbatch.sum().item()
        if valid_tokens_in_microbatch > 0:
            microbatch_avg_entropy = masked_entropy.sum() / valid_tokens_in_microbatch
        else:
            microbatch_avg_entropy = torch.tensor(0.0, device=token_entropy.device)
        metadata["token_entropy"] += microbatch_avg_entropy.item() * valid_tokens_in_microbatch
        metadata["total_valid_tokens"] += valid_tokens_in_microbatch
        
        per_token_policy_gradient_loss,_ = compute_policy_gradient_loss(
            raw_rewards_or_advantages=group_normalized_rewards_microbatch,
            policy_log_probs=log_probs,
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
            response_mask=response_mask_microbatch
        )
        loss = aggregate_loss_across_microbatch(
            per_token_policy_gradient_loss=per_token_policy_gradient_loss,
            mask=response_mask_microbatch,
            loss_normalization=loss_normalization,
            normalization_constant=normalization_constant
        )
        if loss_normalization == "sequence":
            loss = loss*len(inputs_microbatch)/num_rollout
        else:
            loss = loss
        loss.backward()
        
        metadata["loss"]+=loss.detach()
    if max_grad_norm:
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    else:
        total_norm = 0.0
    metadata["gradient_norm"] = total_norm
    if metadata["total_valid_tokens"] > 0:
        metadata["token_entropy"] = metadata["token_entropy"] / metadata["total_valid_tokens"]
    else:
        metadata["token_entropy"] = 0.0
    
    optimizer.step()
    optimizer.zero_grad()
    
    return (metadata["loss"],metadata)
