"""Fail-fast validation for the module-level training configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final


_ALGORITHM_FIELDS: Final = frozenset(
    {
        "baseline",
        "advantage_normalizer",
        "loss_normalization",
        "normalization_constant",
        "importance_reweighting_method",
        "cliprange",
    }
)
_BASELINES: Final = frozenset({"mean", "none"})
_ADVANTAGE_NORMALIZERS: Final = frozenset({"std", "none", "mean"})
_LOSS_NORMALIZATIONS: Final = frozenset({"sequence", "constant"})
_IMPORTANCE_REWEIGHTING_METHODS: Final = frozenset(
    {"none", "noclip", "grpo", "gspo"}
)


def validate_training_config(
    *,
    learning_rate: float,
    rollout_batch_size: int,
    group_size: int,
    inference_batch_size: int,
    off_policy_speedup_factor: int,
    training_batch_size: int,
    gradient_accumulation_steps: int,
    num_training_steps: int,
    val_batch_size: int,
    sampling_temperature: float,
    sampling_max_tokens: int,
    max_grad_norm: float | None,
    vllm_gpu_memory_utilization: float,
    advantage_algorithm: str,
    advantage_config: Mapping[str, object],
) -> None:
    """Validate cross-field invariants before allocating training resources.

    ``training_batch_size`` must contain whole reward groups: GRPO-style
    advantage normalization reshapes every training batch by ``group_size``.
    The function collects all errors so a user can fix a configuration in one
    edit/restart cycle.
    """
    errors: list[str] = []

    _require_positive_int("rollout_batch_size", rollout_batch_size, errors)
    _require_positive_int("group_size", group_size, errors)
    _require_positive_int("inference_batch_size", inference_batch_size, errors)
    _require_positive_int("off_policy_speedup_factor", off_policy_speedup_factor, errors)
    _require_positive_int("training_batch_size", training_batch_size, errors)
    _require_positive_int("gradient_accumulation_steps", gradient_accumulation_steps, errors)
    _require_positive_int("num_training_steps", num_training_steps, errors)
    _require_positive_int("val_batch_size", val_batch_size, errors)
    _require_positive_int("sampling_max_tokens", sampling_max_tokens, errors)

    _require_positive_finite("learning_rate", learning_rate, errors)
    _require_nonnegative_finite("sampling_temperature", sampling_temperature, errors)
    _require_fraction("vllm_gpu_memory_utilization", vllm_gpu_memory_utilization, errors)
    if max_grad_norm is not None:
        _require_positive_finite("max_grad_norm", max_grad_norm, errors)

    if _is_positive_int(rollout_batch_size) and _is_positive_int(group_size):
        if rollout_batch_size % group_size:
            errors.append("rollout_batch_size must be divisible by group_size.")

    if _is_positive_int(inference_batch_size) and _is_positive_int(group_size) and _is_positive_int(rollout_batch_size):
        expected_rollout_batch_size = inference_batch_size * group_size
        if expected_rollout_batch_size != rollout_batch_size:
            errors.append(
                "inference_batch_size * group_size must equal rollout_batch_size "
                f"(got {inference_batch_size} * {group_size} = "
                f"{expected_rollout_batch_size}, expected {rollout_batch_size})."
            )

    if _is_positive_int(rollout_batch_size) and _is_positive_int(off_policy_speedup_factor):
        if rollout_batch_size % off_policy_speedup_factor:
            errors.append("rollout_batch_size must be divisible by off_policy_speedup_factor.")
        elif _is_positive_int(training_batch_size):
            expected_training_batch_size = rollout_batch_size // off_policy_speedup_factor
            if training_batch_size != expected_training_batch_size:
                errors.append(
                    "training_batch_size must equal rollout_batch_size / "
                    "off_policy_speedup_factor "
                    f"(expected {expected_training_batch_size}, got {training_batch_size})."
                )

    if _is_positive_int(training_batch_size) and _is_positive_int(group_size):
        if training_batch_size % group_size:
            errors.append(
                "training_batch_size must be divisible by group_size so reward groups "
                "are not split across updates."
            )

    if _is_positive_int(gradient_accumulation_steps) and _is_positive_int(training_batch_size):
        if gradient_accumulation_steps > training_batch_size:
            errors.append("gradient_accumulation_steps must not exceed training_batch_size.")

    _validate_algorithm_config(advantage_algorithm, advantage_config, errors)

    if errors:
        formatted_errors = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Invalid training configuration:\n{formatted_errors}")


def _validate_algorithm_config(
    advantage_algorithm: str,
    config: Mapping[str, object],
    errors: list[str],
) -> None:
    missing = _ALGORITHM_FIELDS - config.keys()
    unknown = config.keys() - _ALGORITHM_FIELDS
    if missing:
        errors.append(
            f"{advantage_algorithm!r} is missing algorithm fields: {', '.join(sorted(missing))}."
        )
    if unknown:
        errors.append(
            f"{advantage_algorithm!r} has unknown algorithm fields: {', '.join(sorted(unknown))}."
        )
    if missing:
        return

    baseline = config["baseline"]
    if not _is_choice(baseline, _BASELINES):
        errors.append(f"{advantage_algorithm!r}.baseline must be one of {_format_choices(_BASELINES)}.")

    advantage_normalizer = config["advantage_normalizer"]
    if not _is_choice(advantage_normalizer, _ADVANTAGE_NORMALIZERS):
        errors.append(
            f"{advantage_algorithm!r}.advantage_normalizer must be one of "
            f"{_format_choices(_ADVANTAGE_NORMALIZERS)}."
        )

    loss_normalization = config["loss_normalization"]
    if not _is_choice(loss_normalization, _LOSS_NORMALIZATIONS):
        errors.append(
            f"{advantage_algorithm!r}.loss_normalization must be one of "
            f"{_format_choices(_LOSS_NORMALIZATIONS)}."
        )

    normalization_constant = config["normalization_constant"]
    if loss_normalization == "constant":
        if not _is_positive_int(normalization_constant):
            errors.append(
                f"{advantage_algorithm!r}.normalization_constant must be a positive "
                "integer when loss_normalization is 'constant'."
            )
    elif loss_normalization == "sequence" and normalization_constant is not None:
        errors.append(
            f"{advantage_algorithm!r}.normalization_constant must be None when "
            "loss_normalization is 'sequence'."
        )

    method = config["importance_reweighting_method"]
    method_is_valid = _is_choice(method, _IMPORTANCE_REWEIGHTING_METHODS)
    if not method_is_valid:
        errors.append(
            f"{advantage_algorithm!r}.importance_reweighting_method must be one of "
            f"{_format_choices(_IMPORTANCE_REWEIGHTING_METHODS)}."
        )

    cliprange = config["cliprange"]
    if method_is_valid and method in {"grpo", "gspo"}:
        if not _is_finite_number(cliprange) or not 0 < float(cliprange) < 1:
            errors.append(
                f"{advantage_algorithm!r}.cliprange must be in (0, 1) for {method!r}."
            )
    elif method_is_valid and method in {"none", "noclip"} and cliprange is not None:
        errors.append(
            f"{advantage_algorithm!r}.cliprange must be None when "
            f"importance_reweighting_method is {method!r}."
        )


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_choice(value: object, choices: frozenset[str]) -> bool:
    return isinstance(value, str) and value in choices


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_positive_int(name: str, value: object, errors: list[str]) -> None:
    if not _is_positive_int(value):
        errors.append(f"{name} must be a positive integer (got {value!r}).")


def _require_positive_finite(name: str, value: object, errors: list[str]) -> None:
    if not _is_finite_number(value) or float(value) <= 0:
        errors.append(f"{name} must be a positive finite number (got {value!r}).")


def _require_nonnegative_finite(name: str, value: object, errors: list[str]) -> None:
    if not _is_finite_number(value) or float(value) < 0:
        errors.append(f"{name} must be a non-negative finite number (got {value!r}).")


def _require_fraction(name: str, value: object, errors: list[str]) -> None:
    if not _is_finite_number(value) or not 0 < float(value) <= 1:
        errors.append(f"{name} must be in (0, 1] (got {value!r}).")


def _format_choices(choices: frozenset[str]) -> str:
    return ", ".join(repr(choice) for choice in sorted(choices))
