"""Fast smoke tests for the public, model-independent project seams."""

import importlib
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from utils import config_validation, grader, helper, vllm_utils


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_project_packages_are_importable() -> None:
    """The packages installed by uv expose both project namespaces."""
    assert importlib.import_module("experiment").__name__ == "experiment"
    assert importlib.import_module("utils").__name__ == "utils"


def test_gsm8k_data_and_prompt_template_are_usable() -> None:
    data_path = REPO_ROOT / "data" / "gsm8k" / "train.jsonl"
    prompt_path = REPO_ROOT / "prompts" / "r1_zero.prompt"

    with data_path.open(encoding="utf-8") as data_file:
        first_example = json.loads(next(line for line in data_file if line.strip()))
    prompt_template = prompt_path.read_text(encoding="utf-8")

    assert first_example["question"]
    assert "####" in first_example["answer"]
    prompt = prompt_template.format(question=first_example["question"])
    assert first_example["question"] in prompt
    assert "{question}" not in prompt


def test_extract_ground_truth_handles_gsm8k_answers() -> None:
    assert helper.extract_ground_truth("calculation\n#### 72") == "72"
    assert helper.extract_ground_truth("missing final marker") is None


def test_seed_everything_replays_python_numpy_and_torch_randomness() -> None:
    helper.seed_everything(17)
    first = (random.random(), np.random.random(), torch.rand(3))

    helper.seed_everything(17)
    second = (random.random(), np.random.random(), torch.rand(3))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert os.environ["PYTHONHASHSEED"] == "17"
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True
    assert torch.are_deterministic_algorithms_enabled()


def _valid_training_config() -> dict[str, object]:
    return {
        "learning_rate": 1e-5,
        "rollout_batch_size": 256,
        "group_size": 8,
        "inference_batch_size": 32,
        "off_policy_speedup_factor": 1,
        "training_batch_size": 256,
        "gradient_accumulation_steps": 32,
        "num_training_steps": 200,
        "val_batch_size": 64,
        "sampling_temperature": 1.0,
        "sampling_max_tokens": 512,
        "max_grad_norm": 1.0,
        "vllm_gpu_memory_utilization": 0.75,
        "advantage_algorithm": "GRPO",
        "advantage_config": {
            "baseline": "mean",
            "advantage_normalizer": "std",
            "loss_normalization": "sequence",
            "normalization_constant": None,
            "importance_reweighting_method": "none",
            "cliprange": None,
        },
    }


def test_training_config_validation_accepts_default_layout() -> None:
    config_validation.validate_training_config(**_valid_training_config())


def test_training_config_validation_reports_batch_group_contracts() -> None:
    config = _valid_training_config()
    config["inference_batch_size"] = 31
    config["training_batch_size"] = 60

    with pytest.raises(ValueError) as error:
        config_validation.validate_training_config(**config)

    message = str(error.value)
    assert "inference_batch_size * group_size" in message
    assert "training_batch_size must equal" in message
    assert "training_batch_size must be divisible by group_size" in message


def test_training_config_validation_checks_algorithm_dependencies() -> None:
    config = _valid_training_config()
    config["advantage_algorithm"] = "offpolicy_grpo"
    config["advantage_config"] = {
        **config["advantage_config"],
        "importance_reweighting_method": "grpo",
        "cliprange": None,
    }

    with pytest.raises(ValueError, match="cliprange must be in \\(0, 1\\)"):
        config_validation.validate_training_config(**config)


class _TinyTokenizer:
    """Minimal tokenizer double for the tokenize helper's injected interface."""

    _ids = {
        "prompt": [1, 2],
        "answer": [3, 4, 5],
        "correct": [6],
        "wrong": [7],
    }

    def __call__(self, texts, **kwargs):  # noqa: ANN001 - mirrors tokenizer API
        return {"input_ids": [self._ids[text] for text in texts]}


def test_tokenize_prompt_and_output_builds_response_mask() -> None:
    tokens = helper.tokenize_prompt_and_output(
        prompt_strs=["prompt"],
        output_strs=["answer"],
        tokenizer=_TinyTokenizer(),
    )

    assert tokens["input_ids"].tolist() == [[1, 2, 3, 4]]
    assert tokens["labels"].tolist() == [[2, 3, 4, 5]]
    assert tokens["response_mask"].tolist() == [[False, True, True, True]]


def test_packed_response_log_probs_offloads_only_response_tokens() -> None:
    log_probs = torch.tensor(
        [[-1.0, -2.0, -3.0, -4.0], [-5.0, -6.0, -7.0, -8.0]],
        dtype=torch.float32,
    )
    response_mask = torch.tensor(
        [[False, True, True, False], [False, False, True, True]],
    )

    packed = helper.PackedResponseLogProbs.from_padded(log_probs, response_mask)

    assert packed.values.device.type == "cpu"
    assert packed.values.dtype == torch.float32
    assert packed.values.tolist() == pytest.approx([-2.0, -3.0, -7.0, -8.0])
    assert packed.offsets.tolist() == [0, 2, 4]

    selected = packed.select(torch.tensor([False, True]))
    materialized = selected.materialize(
        response_mask=response_mask[1:],
        device="cpu",
        dtype=torch.float32,
    )
    assert torch.allclose(
        materialized,
        torch.tensor([[0.0, 0.0, -7.0, -8.0]]),
    )


def test_importance_metrics_report_ratio_kl_clip_and_ess() -> None:
    policy_log_probs = torch.log(torch.tensor([[2.0, 0.5]]))
    old_log_probs = torch.zeros_like(policy_log_probs)
    response_mask = torch.ones_like(policy_log_probs, dtype=torch.bool)

    _, metadata = helper.compute_policy_gradient_loss(
        raw_rewards_or_advantages=torch.ones_like(policy_log_probs),
        policy_log_probs=policy_log_probs,
        importance_reweighting_method="grpo",
        old_log_probs=old_log_probs,
        cliprange=0.2,
        response_mask=response_mask,
    )

    assert metadata["importance_sample_count"] == pytest.approx(2.0)
    assert metadata["importance_ratio_mean"] == pytest.approx(1.25)
    assert metadata["importance_ratio_min"] == pytest.approx(0.5)
    assert metadata["importance_ratio_max"] == pytest.approx(2.0)
    assert metadata["importance_clip_fraction"] == pytest.approx(1.0)
    assert metadata["importance_ess"] == pytest.approx(25 / 17)
    assert metadata["importance_ess_fraction"] == pytest.approx(25 / 34)
    assert metadata["importance_approx_kl"] == pytest.approx(
        (
            (2.0 - 1.0 - torch.log(torch.tensor(2.0))).item()
            + (0.5 - 1.0 - torch.log(torch.tensor(0.5))).item()
        )
        / 2
    )


def test_gspo_importance_metrics_aggregate_by_sequence() -> None:
    policy_log_probs = torch.log(torch.tensor([[2.0, 2.0], [0.5, 0.5]]))
    old_log_probs = torch.zeros_like(policy_log_probs)
    response_mask = torch.ones_like(policy_log_probs, dtype=torch.bool)

    _, metadata = helper.compute_policy_gradient_loss(
        raw_rewards_or_advantages=torch.ones_like(policy_log_probs),
        policy_log_probs=policy_log_probs,
        importance_reweighting_method="gspo",
        old_log_probs=old_log_probs,
        cliprange=0.2,
        response_mask=response_mask,
    )

    assert metadata["importance_sample_count"] == pytest.approx(2.0)
    assert metadata["importance_ratio_mean"] == pytest.approx(1.25)
    assert metadata["importance_clip_fraction"] == pytest.approx(1.0)


def test_rollout_rewards_and_group_advantages() -> None:
    def reward_fn(response: str, ground_truth: str) -> dict[str, float]:
        correct = float(response == ground_truth)
        return {
            "reward": correct,
            "answer_reward": correct,
            "format_reward": 1.0,
        }

    raw_rewards, metadata = helper.compute_rollout_rewards(
        reward_fn=reward_fn,
        rollout_responses=["yes", "no"],
        repeated_ground_truths=["yes", "yes"],
        group_size=2,
    )
    advantages, group_metadata = helper.compute_group_normalized_rewards(
        raw_rewards=torch.tensor([0.0, 1.0, 2.0, 4.0]),
        group_size=2,
    )

    assert raw_rewards.tolist() == [1.0, 0.0]
    assert metadata["total_reward"] == pytest.approx(1.0)
    assert metadata["total_answer_reward"] == pytest.approx(1.0)
    assert metadata["pass@1"] == pytest.approx(0.5)
    assert metadata["pass@group_size"] == pytest.approx(1.0)
    assert metadata["mean_format_reward"] == pytest.approx(1.0)
    assert advantages.tolist() == pytest.approx(
        [-0.707106, 0.707106, -0.707106, 0.707106],
        abs=1e-5,
    )
    assert group_metadata["avg_reward"] == pytest.approx(1.75)


def test_rollout_reward_signal_is_distinct_from_answer_reward() -> None:
    def reward_fn(response: str, ground_truth: str) -> dict[str, float]:
        if response == "formatted":
            return {"reward": 0.0, "answer_reward": 1.0, "format_reward": 0.0}
        return {"reward": 1.0, "answer_reward": 0.0, "format_reward": 1.0}

    raw_rewards, metadata = helper.compute_rollout_rewards(
        reward_fn=reward_fn,
        rollout_responses=["formatted", "wrong"],
        repeated_ground_truths=["answer", "answer"],
        group_size=2,
    )

    assert raw_rewards.tolist() == [0.0, 1.0]
    assert metadata["total_reward"] == pytest.approx(1.0)
    assert metadata["total_answer_reward"] == pytest.approx(1.0)
    assert metadata["pass@1"] == pytest.approx(0.5)
    assert metadata["pass@group_size"] == pytest.approx(1.0)


def test_empty_rollout_group_pass_rate_is_zero() -> None:
    def reward_fn(response: str, ground_truth: str) -> dict[str, float]:
        return {"reward": 0.0, "answer_reward": 0.0, "format_reward": 0.0}

    _, metadata = helper.compute_rollout_rewards(
        reward_fn=reward_fn,
        rollout_responses=[],
        repeated_ground_truths=[],
        group_size=2,
    )

    assert metadata["pass@group_size"] == 0.0


def test_loss_aggregation_supports_sequence_and_constant_normalization() -> None:
    losses = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.tensor([[True, True, False], [True, False, False]])

    sequence_loss = helper.aggregate_loss_across_microbatch(losses, mask, "sequence")
    constant_loss = helper.aggregate_loss_across_microbatch(
        losses,
        mask,
        "constant",
        normalization_constant=4,
    )

    assert sequence_loss.item() == pytest.approx(2.75)
    assert constant_loss.item() == pytest.approx(1.75)


class _TinyPolicy(torch.nn.Module):
    """Small CPU policy implementing the model interface used by one train step."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.output = torch.nn.Linear(4, 8, bias=False)

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=self.output(self.embedding(input_ids)))


def test_grpo_train_step_updates_a_tiny_cpu_policy() -> None:
    helper.seed_everything(42)
    model = _TinyPolicy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]

    def reward_fn(response: str, ground_truth: str) -> dict[str, float]:
        correct = float(response == ground_truth)
        return {
            "reward": correct,
            "answer_reward": correct,
            "format_reward": 1.0,
        }

    loss, metadata = helper.grpo_train_step(
        model=model,
        tokenizer=_TinyTokenizer(),
        optimizer=optimizer,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        reward_fn=reward_fn,
        repeated_prompts=["prompt", "prompt"],
        rollout_responses=["correct", "wrong"],
        repeated_ground_truths=["correct", "correct"],
        group_size=2,
    )

    assert torch.isfinite(loss)
    assert metadata["num_rollout"] == 2
    assert metadata["num_pruned_rollout"] == 2
    assert metadata["mean_reward"] == pytest.approx(0.5)
    assert metadata["pass@1"] == pytest.approx(0.5)
    assert metadata["pass@group_size"] == pytest.approx(1.0)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(parameters_before, model.parameters())
    )


def test_grpo_train_step_accepts_packed_off_policy_log_probs() -> None:
    helper.seed_everything(42)
    model = _TinyPolicy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    prompts = ["prompt", "prompt"]
    responses = ["correct", "wrong"]
    tokens = helper.tokenize_prompt_and_output(prompts, responses, _TinyTokenizer())
    with torch.no_grad():
        dense_old_log_probs = helper.get_response_log_probs(
            model,
            tokens["input_ids"],
            tokens["labels"],
        )["log_probs"]
    packed_old_log_probs = helper.PackedResponseLogProbs.from_padded(
        dense_old_log_probs,
        tokens["response_mask"],
    )

    def reward_fn(response: str, ground_truth: str) -> dict[str, float]:
        correct = float(response == ground_truth)
        return {"reward": correct, "format_reward": 1.0}

    loss, metadata = helper.grpo_train_step(
        model=model,
        tokenizer=_TinyTokenizer(),
        optimizer=optimizer,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        reward_fn=reward_fn,
        repeated_prompts=prompts,
        rollout_responses=responses,
        repeated_ground_truths=["correct", "correct"],
        group_size=2,
        importance_reweighting_method="grpo",
        old_log_probs=packed_old_log_probs,
        cliprange=0.2,
    )

    assert torch.isfinite(loss)
    assert metadata["importance_sample_count"] == pytest.approx(2.0)
    assert metadata["importance_approx_kl"] is not None
    assert metadata["importance_ess_fraction"] == pytest.approx(1.0, abs=1e-3)


def test_math_rewards_distinguish_format_and_correctness() -> None:
    correct = grader.r1_zero_reward_fn(
        "<think>2 + 2 = 4</think> <answer>4</answer>",
        "4",
    )
    wrong = grader.r1_zero_reward_fn(
        "<think>2 + 2 = 4</think> <answer>5</answer>",
        "4",
    )
    malformed = grader.r1_zero_reward_fn(
        "<think>2 + 2 = 4</think><answer>4</answer>",
        "4",
    )
    boxed = grader.question_only_reward_fn(r"\boxed{4}", "4")

    assert correct == {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}
    assert wrong == {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}
    assert malformed["format_reward"] == 0.0
    assert boxed == {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}


def test_vllm_completion_adapter_batches_and_sorts_choices(monkeypatch) -> None:  # noqa: ANN001
    requests = []

    def fake_http_json(method, url, payload=None, timeout=60):  # noqa: ANN001
        requests.append((method, url, payload, timeout))
        prompts = payload["prompt"]
        return {
            "choices": [
                {
                    "index": 1,
                    "text": f"{prompts[0]}-second",
                    "token_ids": [2],
                    "finish_reason": "stop",
                },
                {
                    "index": 0,
                    "text": f"{prompts[0]}-first",
                    "token_ids": [1],
                    "finish_reason": "length",
                },
            ]
        }

    monkeypatch.setattr(vllm_utils, "_http_json", fake_http_json)
    completions = vllm_utils.generate_completions(
        vllm_base_url="http://localhost:8000",
        model_id="demo-model",
        prompts=["question-1", "question-2"],
        sampling_params={
            "temperature": 0.7,
            "max_tokens": 16,
            "n": 2,
            "seed": 42,
            "stop": ["</answer>"],
            "include_stop_str_in_output": True,
        },
        batch_size=1,
    )

    assert [completion.text for completion in completions] == [
        "question-1-first",
        "question-1-second",
        "question-2-first",
        "question-2-second",
    ]
    assert completions[0].finish_reason == "length"
    assert len(requests) == 2
    method, url, payload, timeout = requests[0]
    assert method == "POST"
    assert url == "http://localhost:8000/v1/completions"
    assert timeout == 3600
    assert payload["model"] == "demo-model"
    assert payload["return_token_ids"] is True
    assert payload["stop"] == ["</answer>"]
