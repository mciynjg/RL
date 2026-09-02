import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer,PreTrainedTokenizer
from transformers import PreTrainedTokenizerBase,PreTrainedModel
from typing import Callable,Literal,List,Tuple
from dataclasses import dataclass
import random
import re
import os
import numpy as np


@dataclass(frozen=True)
class PackedResponseLogProbs:
    """CPU-resident response-only log-probabilities for one rollout batch.

    The packed representation deliberately omits prompt and padding positions.
    ``materialize`` reconstructs only the current microbatch, so callers do not
    need to keep a dense ``[batch, sequence]`` cache on the training device.
    """

    values: torch.Tensor
    offsets: torch.Tensor

    @classmethod
    def from_padded(
        cls,
        log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        *,
        storage_dtype: torch.dtype = torch.float32,
    ) -> "PackedResponseLogProbs":
        if log_probs.ndim != 2 or response_mask.ndim != 2:
            raise ValueError("log_probs and response_mask must both be rank-2 tensors.")
        if log_probs.shape != response_mask.shape:
            raise ValueError(
                "log_probs and response_mask must have the same shape "
                f"(got {tuple(log_probs.shape)} and {tuple(response_mask.shape)})."
            )
        if not log_probs.is_floating_point():
            raise TypeError("log_probs must use a floating-point dtype.")
        if not response_mask.dtype == torch.bool:
            response_mask = response_mask.to(dtype=torch.bool)

        mask_on_log_prob_device = response_mask.to(
            device=log_probs.device,
            dtype=torch.bool,
        )
        lengths = response_mask.sum(dim=-1, dtype=torch.long).to(device="cpu")
        offsets = torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(dim=0)))
        return cls(
            values=log_probs.masked_select(mask_on_log_prob_device)
            .detach()
            .to(device="cpu", dtype=storage_dtype)
            .contiguous(),
            offsets=offsets,
        )

    @property
    def num_sequences(self) -> int:
        return self.offsets.numel() - 1

    def select(self, mask: torch.Tensor) -> "PackedResponseLogProbs":
        """Select rollout rows while preserving their response-only packing."""
        mask = mask.detach().to(device="cpu", dtype=torch.bool).flatten()
        if mask.numel() != self.num_sequences:
            raise ValueError(
                "selection mask length must match packed rollout count "
                f"(got {mask.numel()} and {self.num_sequences})."
            )
        selected_indices = mask.nonzero(as_tuple=False).flatten().tolist()
        chunks = [
            self.values[int(self.offsets[index]) : int(self.offsets[index + 1])]
            for index in selected_indices
        ]
        selected_values = torch.cat(chunks) if chunks else self.values.new_empty(0)
        selected_lengths = torch.tensor(
            [chunk.numel() for chunk in chunks],
            dtype=torch.long,
        )
        selected_offsets = torch.cat(
            (torch.zeros(1, dtype=torch.long), selected_lengths.cumsum(dim=0))
        )
        return PackedResponseLogProbs(selected_values, selected_offsets)

    def slice(self, start: int, stop: int) -> "PackedResponseLogProbs":
        """Return a contiguous sequence range without materializing padding."""
        if not 0 <= start <= stop <= self.num_sequences:
            raise ValueError(
                f"invalid packed sequence slice [{start}:{stop}] for {self.num_sequences} sequences."
            )
        value_start = int(self.offsets[start])
        value_stop = int(self.offsets[stop])
        return PackedResponseLogProbs(
            self.values[value_start:value_stop],
            self.offsets[start : stop + 1] - value_start,
        )

    def materialize(
        self,
        response_mask: torch.Tensor,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Materialize packed values into a dense tensor for one microbatch."""
        if response_mask.ndim != 2:
            raise ValueError("response_mask must be a rank-2 tensor.")
        if response_mask.shape[0] != self.num_sequences:
            raise ValueError(
                "response_mask row count must match packed rollout count "
                f"(got {response_mask.shape[0]} and {self.num_sequences})."
            )
        response_mask = response_mask.to(device=device, dtype=torch.bool)
        response_lengths = response_mask.sum(dim=-1, dtype=torch.long).to(device="cpu")
        packed_lengths = self.offsets[1:] - self.offsets[:-1]
        if not torch.equal(response_lengths, packed_lengths):
            raise ValueError(
                "packed response lengths do not match response_mask: "
                f"got {response_lengths.tolist()} and {packed_lengths.tolist()}."
            )
        materialized = torch.zeros(
            response_mask.shape,
            device=device,
            dtype=dtype,
        )
        values = self.values.to(device=device, dtype=dtype)
        if values.numel():
            materialized[response_mask] = values
        return materialized


_IMPORTANCE_STAT_KEYS = (
    "_ratio_sum",
    "_ratio_sq_sum",
    "_ratio_count",
    "_ratio_min",
    "_ratio_max",
    "_kl_sum",
    "_clip_count",
)


@torch.no_grad()
def _importance_statistics(
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor | None,
    response_mask: torch.Tensor | None,
    importance_reweighting_method: str,
    cliprange: float | None,
) -> dict[str, float]:
    """Return detached sufficient statistics for off-policy diagnostics.

    Token-ratio methods contribute one sample per response token. GSPO uses one
    geometric-mean ratio per response sequence, matching its objective.
    """
    if importance_reweighting_method == "none" or old_log_probs is None:
        return {key: 0.0 for key in _IMPORTANCE_STAT_KEYS}
    if response_mask is None:
        mask = torch.ones_like(policy_log_probs, dtype=torch.bool)
    else:
        mask = response_mask.to(device=policy_log_probs.device, dtype=torch.bool)

    log_ratio = policy_log_probs.float() - old_log_probs.float()
    if importance_reweighting_method == "gspo":
        sequence_lengths = mask.sum(dim=-1)
        valid_sequences = sequence_lengths > 0
        if not valid_sequences.any():
            return {key: 0.0 for key in _IMPORTANCE_STAT_KEYS}
        sequence_log_ratio = (log_ratio * mask).sum(dim=-1) / sequence_lengths.clamp_min(1)
        ratios = torch.exp(sequence_log_ratio[valid_sequences])
        sequence_log_ratio = sequence_log_ratio[valid_sequences]
        kl_values = ratios - 1 - sequence_log_ratio
        clipped = torch.zeros_like(ratios, dtype=torch.bool)
        if cliprange is not None:
            clipped = (ratios < 1 - cliprange) | (ratios > 1 + cliprange)
    else:
        valid = mask
        if not valid.any():
            return {key: 0.0 for key in _IMPORTANCE_STAT_KEYS}
        ratios = torch.exp(log_ratio[valid])
        kl_values = ratios - 1 - log_ratio[valid]
        clipped = torch.zeros_like(ratios, dtype=torch.bool)
        if importance_reweighting_method == "grpo" and cliprange is not None:
            clipped = (ratios < 1 - cliprange) | (ratios > 1 + cliprange)

    ratios = ratios.detach()
    kl_values = kl_values.detach()
    finite_values = torch.isfinite(ratios) & torch.isfinite(kl_values)
    ratios = ratios[finite_values]
    kl_values = kl_values[finite_values]
    clipped = clipped[finite_values]
    if ratios.numel() == 0:
        return {key: 0.0 for key in _IMPORTANCE_STAT_KEYS}
    ratio_sum = float(ratios.sum().item())
    ratio_sq_sum = float((ratios * ratios).sum().item())
    return {
        "_ratio_sum": ratio_sum,
        "_ratio_sq_sum": ratio_sq_sum,
        "_ratio_count": float(ratios.numel()),
        "_ratio_min": float(ratios.min().item()),
        "_ratio_max": float(ratios.max().item()),
        "_kl_sum": float(kl_values.sum().item()),
        "_clip_count": float(clipped.sum().item()),
    }


def _finalize_importance_statistics(
    statistics: dict[str, float],
) -> dict[str, float | None]:
    count = statistics["_ratio_count"]
    if count <= 0:
        return {
            "importance_ratio_mean": None,
            "importance_ratio_std": None,
            "importance_ratio_min": None,
            "importance_ratio_max": None,
            "importance_approx_kl": None,
            "importance_clip_fraction": None,
            "importance_ess": None,
            "importance_ess_fraction": None,
            "importance_sample_count": 0.0,
        }
    ratio_sum = statistics["_ratio_sum"]
    ratio_sq_sum = statistics["_ratio_sq_sum"]
    ratio_mean = ratio_sum / count
    ratio_variance = max(ratio_sq_sum / count - ratio_mean * ratio_mean, 0.0)
    ess = (ratio_sum * ratio_sum) / ratio_sq_sum if ratio_sq_sum > 0 else 0.0
    return {
        "importance_ratio_mean": ratio_mean,
        "importance_ratio_std": ratio_variance**0.5,
        "importance_ratio_min": statistics["_ratio_min"],
        "importance_ratio_max": statistics["_ratio_max"],
        "importance_approx_kl": statistics["_kl_sum"] / count,
        "importance_clip_fraction": statistics["_clip_count"] / count,
        "importance_ess": ess,
        "importance_ess_fraction": ess / count,
        "importance_sample_count": count,
    }


def _accumulate_importance_statistics(
    total: dict[str, float],
    current: dict[str, float],
) -> None:
    """Merge one microbatch's sufficient statistics into a train-step total."""
    if current["_ratio_count"] <= 0:
        return
    if total["_ratio_count"] <= 0:
        total.update(current)
        return
    for key in ("_ratio_sum", "_ratio_sq_sum", "_ratio_count", "_kl_sum", "_clip_count"):
        total[key] += current[key]
    total["_ratio_min"] = min(total["_ratio_min"], current["_ratio_min"])
    total["_ratio_max"] = max(total["_ratio_max"], current["_ratio_max"])


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
    
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    labels = labels.to(device)
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
    group_size: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
    raw_rewards = []
    answer_rewards = []
    total_reward = 0.0
    mean_reward = 0.0
    total_answer_reward = 0.0
    mean_answer_reward = 0.0
    total_format_reward = 0.0
    mean_format_reward = 0.0
    for response,ground_truth in zip(rollout_responses,repeated_ground_truths):
        reward = reward_fn(response,ground_truth)
        # The total reward is the optimization signal; answer reward is kept
        # separate for accuracy/pass@k observability.
        raw_rewards.append(reward["reward"])
        # Keep compatibility with custom legacy reward functions while the
        # built-in grader always supplies the full reward schema.
        answer_rewards.append(reward.get("answer_reward", reward["reward"]))
        total_format_reward +=reward["format_reward"]
    total_reward = sum(raw_rewards)
    total_answer_reward = sum(answer_rewards)
    if rollout_responses:
        mean_reward = total_reward/len(rollout_responses)
        mean_answer_reward = total_answer_reward/len(rollout_responses)
        mean_format_reward = total_format_reward/len(rollout_responses)
    pass_at_1 = (
        sum(answer_reward > 0.0 for answer_reward in answer_rewards) / len(answer_rewards)
        if answer_rewards
        else 0.0
    )
    pass_at_group_size = 0.0
    if (
        answer_rewards
        and group_size is not None
        and group_size > 0
        and len(answer_rewards) % group_size == 0
    ):
        grouped_answer_rewards = torch.tensor(answer_rewards).reshape(-1, group_size)
        pass_at_group_size = (grouped_answer_rewards > 0.0).any(dim=-1).float().mean().item()
    metadata = {
        "total_reward":total_reward,
        "mean_reward":mean_reward,
        "total_answer_reward":total_answer_reward,
        "mean_answer_reward":mean_answer_reward,
        "total_format_reward":total_format_reward,
        "mean_format_reward":mean_format_reward,
        "pass@1":pass_at_1,
        "pass@group_size":pass_at_group_size,
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
    elif advantage_normalizer == "mean":
        factor = torch.mean(raw_rewards,dim=-1,keepdim=True)
    else:
        raise NotImplementedError
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
    ) -> tuple[torch.Tensor, dict[str, float]]:
    if importance_reweighting_method == "none":
        per_token_policy_gradient_loss = -raw_rewards_or_advantages*policy_log_probs
    elif importance_reweighting_method == "noclip":
        if old_log_probs is not None :
            importance_reweighting = torch.exp(
                policy_log_probs.float() - old_log_probs.float()
            )
            per_token_policy_gradient_loss = -raw_rewards_or_advantages*importance_reweighting
        else:
            raise ValueError("must pass old_log_probs")
    elif importance_reweighting_method == "grpo":
        if old_log_probs is not None and cliprange is not None:
            per_token_importance_reweighting = torch.exp(
                policy_log_probs.float() - old_log_probs.float()
            )
            clip_term = torch.min(torch.max(per_token_importance_reweighting,torch.tensor(1-cliprange,device=per_token_importance_reweighting.device)),torch.tensor(1+cliprange,device=per_token_importance_reweighting.device))
            per_token_policy_gradient_loss = -torch.min(raw_rewards_or_advantages*per_token_importance_reweighting,raw_rewards_or_advantages*clip_term)
        else:
            raise ValueError("must pass old_log_probs and cliprange")
    elif importance_reweighting_method == "gspo":
        if old_log_probs is not None and cliprange is not None and response_mask is not None:
            sequence_importance_reweighting = (
                (policy_log_probs.float() - old_log_probs.float()) * response_mask
            ).sum(dim=-1,keepdim=True)
            logs = sequence_importance_reweighting/(response_mask.sum(dim=-1,keepdim=True)+1e-6)
            s = torch.exp(logs)
            clip_term = torch.min(torch.max(s,torch.tensor(1-cliprange,device=sequence_importance_reweighting.device)),torch.tensor(1+cliprange,device=sequence_importance_reweighting.device))
            per_token_policy_gradient_loss = -torch.min(raw_rewards_or_advantages*s,raw_rewards_or_advantages*clip_term)
            per_token_policy_gradient_loss = per_token_policy_gradient_loss.expand_as(policy_log_probs)
        else:
            raise ValueError("must pass old_log_probs,cliprange and response_mask")
    else:
        raise NotImplementedError
    raw_statistics = _importance_statistics(
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        response_mask=response_mask,
        importance_reweighting_method=importance_reweighting_method,
        cliprange=cliprange,
    )
    metadata = {}
    if importance_reweighting_method != "none":
        metadata = {
            **raw_statistics,
            **_finalize_importance_statistics(raw_statistics),
        }
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
    old_log_probs: PackedResponseLogProbs | torch.Tensor | None = None,
    cliprange: float | None = None, 
    # Loss normalization 
    loss_normalization: Literal["sequence", "constant"] = "sequence", 
    normalization_constant: int | None = None, 
    ) -> tuple[torch.Tensor, dict[str, object]]:
    
    optimizer.zero_grad()
    
    device = next(model.parameters()).device
    metadata = {
        "loss":torch.zeros(1).to(device),
        "gradient_norm":0.0,
        "token_entropy":0.0,
        "total_valid_tokens":0.0,
        # A: 核心进展 (按 rollout 数归一化, 跨配置可比)
        "total_reward":0.0,
        "total_answer_reward":0.0,
        "total_format_reward":0.0,
        "mean_reward":0.0,
        "mean_answer_reward":0.0,
        "mean_format_reward":0.0,
        "pass@1":0.0,
        "pass@group_size":0.0,
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
    importance_statistics = {key: 0.0 for key in _IMPORTANCE_STAT_KEYS}
    metadata.update(_finalize_importance_statistics(importance_statistics))
    
    tokens = tokenize_prompt_and_output(prompt_strs=repeated_prompts,
                                     output_strs=rollout_responses,
                                     tokenizer=tokenizer)
    
    
    inputs = tokens["input_ids"]
    labels = tokens["labels"]
    response_mask = tokens["response_mask"]
    num_rollout = len(repeated_prompts)
    raw_rewards,raw_rewards_metadata = compute_rollout_rewards(
                reward_fn=reward_fn,
                rollout_responses=rollout_responses,
                repeated_ground_truths=repeated_ground_truths,
                group_size=group_size,
                )
    
    total_rewards = raw_rewards_metadata["total_reward"]
    total_format_rewards = raw_rewards_metadata["total_format_reward"]
    metadata["total_reward"] = total_rewards
    metadata["total_answer_reward"] = raw_rewards_metadata["total_answer_reward"]
    metadata["total_format_reward"] = total_format_rewards
    metadata["mean_reward"] = raw_rewards_metadata["mean_reward"]
    metadata["mean_answer_reward"] = raw_rewards_metadata["mean_answer_reward"]
    metadata["mean_format_reward"] = raw_rewards_metadata["mean_format_reward"]
    metadata["pass@1"] = raw_rewards_metadata["pass@1"]
    metadata["pass@group_size"] = raw_rewards_metadata["pass@group_size"]

    # B: 组级退化统计 (与 baseline 取值无关, 始终按 group_size 分组统计)
    grouped_for_stats = raw_rewards.reshape(-1,group_size)
    group_is_degenerate = grouped_for_stats.std(dim=-1) == 0.0
    metadata["degenerate_groups"] = int(group_is_degenerate.sum().item())
    metadata["all_correct_groups"] = int((group_is_degenerate & (grouped_for_stats.mean(dim=-1) > 0)).sum().item())
    metadata["all_wrong_groups"] = int((group_is_degenerate & (grouped_for_stats.mean(dim=-1) == 0)).sum().item())

    # D: 响应长度 / 空响应 (裁剪前的全量口径)
    metadata["response_tokens_mean"] = response_mask.sum(dim=-1).float().mean().item()
    metadata["empty_response_frac"] = (response_mask.sum(dim=-1) == 0).float().mean().item()
    metadata["num_rollout"] = num_rollout

    if baseline == "mean":
        grouped_raw_rewards = raw_rewards.reshape(-1,group_size)
        std = torch.std(grouped_raw_rewards,dim=-1)
        mask = std!=0.0
        mask = mask.repeat_interleave(group_size)
        pruned_raw_rewards = raw_rewards[mask]
        if old_log_probs is not None:
            if isinstance(old_log_probs, PackedResponseLogProbs):
                old_log_probs = old_log_probs.select(mask)
            else:
                old_log_probs = old_log_probs.to(mask.device)[mask]

    if baseline == "none":
        mask = raw_rewards!=0.0
        pruned_raw_rewards = raw_rewards[mask]
        if old_log_probs is not None:
            if isinstance(old_log_probs, PackedResponseLogProbs):
                old_log_probs = old_log_probs.select(mask)
            else:
                old_log_probs = old_log_probs.to(mask.device)[mask]
    num_pruned_rollout = len(pruned_raw_rewards)
    metadata["num_pruned_rollout"] = num_pruned_rollout
    metadata["pruned_frac"] = 1.0 - num_pruned_rollout/num_rollout if num_rollout else 0.0

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
    if num_pruned_rollout > 0:
        metadata["advantage_abs_mean"] = group_normalized_rewards.abs().mean().item()
        metadata["advantage_std"] = group_normalized_rewards.std().item() if num_pruned_rollout > 1 else 0.0

    microbatch_size = max(
        1,
        min(num_rollout//gradient_accumulation_steps, num_pruned_rollout),
    )
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
        
        if isinstance(old_log_probs, PackedResponseLogProbs):
            old_log_probs_microbatch = old_log_probs.slice(
                i,
                min(i + microbatch_size, num_pruned_rollout),
            ).materialize(
                response_mask_microbatch,
                device=device,
                dtype=torch.float32,
            )
        elif old_log_probs is not None:
            old_log_probs_microbatch = old_log_probs[i:i+microbatch_size].to(device)
        else:
            old_log_probs_microbatch = None

        per_token_policy_gradient_loss,importance_metadata = compute_policy_gradient_loss(
            raw_rewards_or_advantages=group_normalized_rewards_microbatch,
            policy_log_probs=log_probs,
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=old_log_probs_microbatch,
            cliprange=cliprange,
            response_mask=response_mask_microbatch
        )
        if importance_reweighting_method != "none":
            _accumulate_importance_statistics(importance_statistics, importance_metadata)
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
    metadata.update(_finalize_importance_statistics(importance_statistics))
    
    optimizer.step()
    optimizer.zero_grad()
    
    return (metadata["loss"],metadata)
