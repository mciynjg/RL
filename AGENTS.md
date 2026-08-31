# Repository Guide

## Environment and Verification

- Use Python 3.12 only (`>=3.12,<3.13`) and manage the installed project with `uv sync`; the lockfile pins the runnable environment.
- Run the CPU/model-independent suite with `uv run pytest -q`; use `uv run pytest -q tests/test_smoke.py` for a focused pass.
- Check syntax without loading a model or starting vLLM: `uv run python -m compileall -q experiment utils`.
- The opt-in GPU integration test downloads a Hugging Face model and launches a real vLLM server: `RUN_E2E=1 E2E_GPU=0 uv run pytest -q tests/test_e2e.py -m e2e`. `E2E_MODEL_ID` can override its default tiny model.

## Layout and Contracts

- `experiment/train.py` is the training entrypoint; its module-level constants are the experiment configuration rather than CLI arguments.
- `utils/helper.py` owns model loading, GSM8K answer extraction, reward normalization, and the GRPO update; `utils/grader.py` owns the reward schema and math-answer validation; `utils/vllm_utils.py` owns server lifecycle, batched completions, and NCCL weight transfer.
- Training reads `data/gsm8k/{train,test}.jsonl` and `prompts/r1_zero.prompt`; GSM8K final answers are extracted from the text after `####`.
- Preserve reward dictionaries with `format_reward`, `answer_reward`, and `reward`. R1-zero completions must contain exactly one `</think>` followed by one literal space and a single `<answer>...</answer>` block; this strict format is part of the reward contract.

## GPU and Experiment Safety

- `uv run python experiment/train.py` starts vLLM on physical GPU 3 and trains on `cuda:2` by default. Adjust both hardcoded device constants before running on a different machine.
- vLLM uses port 8000 and `VLLMServer.start()` terminates an existing matching vLLM process on that port. Do not run training or evaluation concurrently with another workload using that server.
- The training loop synchronizes policy weights to vLLM through NCCL after each rollout; it needs CUDA-capable Linux x86_64, two usable GPUs, a model download, and W&B credentials (or `WANDB_MODE=offline`).
- The evaluation scripts under `utils/eval_*.py` start a real vLLM server and overwrite their fixed JSONL outputs in `results/eval result/`; they are not lightweight checks.
- For stricter reproducibility, set `PYTHONHASHSEED` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` before launching Python; `seed_everything()` cannot change the already-initialized hash seed.
