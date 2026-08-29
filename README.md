# Reasoning RL 实验框架

这是一个面向数学推理任务的强化学习实验项目。当前训练入口以
[`allenai/OLMo-2-0425-1B`](https://huggingface.co/allenai/OLMo-2-0425-1B)
为基础模型，在 GSM8K 上生成分组 rollout，并使用 GRPO、Dr.GRPO、RFT 等策略更新模型。

项目将训练和推理解耦到两张 GPU：Transformers/PyTorch 负责策略模型训练，vLLM
负责批量采样；每轮训练后通过 NCCL 将最新权重同步到 vLLM。训练指标和验证指标记录到
Weights & Biases。

> 当前代码属于实验原型，参数直接写在源码中，并且存在若干已知运行问题。第一次运行前请先阅读
> [已知问题](#已知问题)。

## 功能概览

- 使用 vLLM 为每条 prompt 生成一组候选回答。
- 使用严格的 `<think>...</think> <answer>...</answer>` 格式和数学答案判分器计算奖励。
- 支持 `GRPO`、`GRPO_constant`、`Dr_GRPO`、`RFT` 和 `MaxRL` 配置。
- 过滤奖励完全相同、无法提供组内学习信号的 rollout group。
- 记录奖励、格式正确率、梯度范数、token entropy、输出长度和截断率等指标。
- 定期在 GSM8K 验证集上评估并保存最佳 checkpoint。

## 目录结构

```text
.
├── data/                   # GSM8K、MMLU、HH、AlpacaEval 等数据
├── experiment/
│   ├── __init__.py         # experiment Python 包
│   └── train.py            # 训练主程序与实验参数
├── prompts/                # question-only、R1-zero、three-shot prompt
├── results/
│   └── eval result/        # 已有的基础模型评估结果（JSONL）
├── utils/
│   ├── drgrpo_grader.py    # 格式检查、答案提取与数学判分
│   ├── helper.py           # tokenization、奖励归一化与训练步骤
│   ├── vllm_utils.py       # vLLM 生命周期、生成和 NCCL 权重同步
│   └── eval_*.py           # 三种 prompt 的基础评估脚本
└── pyproject.toml          # Python 版本与依赖配置
```

当前训练只使用 `data/gsm8k/` 和 `prompts/r1_zero.prompt`；其余数据集尚未接入训练入口。

## 环境要求

- Linux x86_64
- Python 3.12（`pyproject.toml` 明确限制为 `>=3.12,<3.13`）
- 支持 CUDA 的 NVIDIA GPU
- 可访问 Hugging Face 以下载模型
- Weights & Biases 账号（也可以通过 `WANDB_MODE=offline` 离线记录）

默认配置将训练模型放在 `cuda:2`，将 vLLM 放在物理 GPU 3，因此机器需要至少有两张可用
GPU，并且默认编号 2、3 必须存在。显存需求取决于模型、batch size 和生成长度；首次尝试建议先
缩小 `rollout_batch_size`、`group_size` 和 `sampling_max_tokens`。

## 安装

推荐使用 [uv](https://docs.astral.sh/uv/) 创建与 `pyproject.toml` 一致的环境：

```bash
uv python install 3.12
uv sync
uv run wandb login
```

Linux 安装会下载 CUDA 12.9 对应的 PyTorch、vLLM 0.19.1，以及项目指定的
FlashAttention wheel，下载量较大。若不希望在线记录 W&B：

```bash
export WANDB_MODE=offline
```

## 训练配置

训练参数目前集中在 `experiment/train.py` 顶部，运行前至少检查以下配置：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `training_device` | `cuda:2` | PyTorch 策略模型所在 GPU |
| `vllm_device` | `3` | vLLM 子进程使用的物理 GPU 编号 |
| `model_name` | `allenai/OLMo-2-0425-1B` | Hugging Face 模型 ID |
| `advantage_algorithm` | `GRPO` | 优势与 loss 归一化方案 |
| `rollout_batch_size` | `256` | 每步生成的回答总数 |
| `group_size` | `8` | 每条问题生成的回答数 |
| `gradient_accumulation_steps` | `32` | 用于切分训练 microbatch |
| `sampling_max_tokens` | `512` | 单条回答的最大生成 token 数 |
| `num_rollout_steps` | `200` | rollout/更新轮数 |
| `n_train_examples` | `6400` | 训练采样数；数据不足时按随机重排后的 epoch 循环取样 |
| `n_val_examples` | `1024` | 使用的验证问题数 |

算法名称对应的归一化设置如下：

| 配置 | baseline | advantage normalizer | loss normalizer |
| --- | --- | --- | --- |
| `GRPO` | group mean | group std | sequence |
| `GRPO_constant` | group mean | group std | constant |
| `Dr_GRPO` | group mean | none | constant |
| `RFT` | none | none | constant |
| `MaxRL` | group mean | group mean | constant |

## 启动训练

完成 `uv sync` 后，从项目根目录直接启动脚本：

```bash
uv run python experiment/train.py
```

`experiment` 和 `utils` 都会作为当前项目的正式 Python 包安装到 uv 环境中，因此直接执行
`experiment/train.py` 时也能导入 `utils`，不需要使用 `python -m`。

训练流程为：

1. 启动本地 vLLM OpenAI-compatible server（默认端口 `8000`）。
2. 加载策略模型、GSM8K 数据和 R1-zero prompt。
3. 每个问题采样 `group_size` 个回答并计算格式/答案奖励。
4. 计算组归一化 advantage，完成一次策略梯度更新。
5. 通过 NCCL 将策略权重同步给 vLLM。
6. 每 10 步验证一次，并尝试保存最佳模型。

checkpoint 目标目录为：

```text
results/checkpoint/<algorithm>-<model>-seed<seed>/
```

## 基础模型评估

仓库提供三种评估脚本：

| 脚本 | Prompt | Reward |
| --- | --- | --- |
| `utils/eval_question_only.py` | 只给问题并要求 `\boxed{}` | `question_only_reward_fn` |
| `utils/eval_r1_zero.py` | R1-zero 格式 | `r1_zero_reward_fn` |
| `utils/eval_rl_three_shot.py` | 三个 GSM8K 示例 + R1-zero 格式 | `r1_zero_reward_fn` |

从项目根目录运行：

```bash
uv run python utils/eval_question_only.py
uv run python utils/eval_r1_zero.py
uv run python utils/eval_rl_three_shot.py
```

已有结果均包含 1,421 条样本，位于 `results/eval result/`：

| 设置 | 答案正确率 | 格式通过率 |
| --- | ---: | ---: |
| Question only | 0.63% | 25.69% |
| R1-zero zero-shot | 0.14% | 59.11% |
| R1-zero three-shot | 21.75% | 96.41% |

每行 JSONL 记录问题、标准答案、模型回答、格式奖励、答案奖励、总奖励和类别。

## 数据说明

`data/gsm8k/train.jsonl` 当前有 1,421 条数据，`data/gsm8k/test.jsonl` 有 1,319 条数据。
每行格式为：

```json
{"question": "...", "answer": "推导过程\n#### 最终答案"}
```

标准答案通过 `####` 后的内容提取。请在使用或分发仓库内各数据集时遵守对应数据集的原始许可。
当 `n_train_examples` 大于训练集行数时，训练入口会以 `seed` 为随机种子重复打乱数据集，按多个
epoch 继续采样，直至达到指定数量；单个 epoch 内不会重复采样。

## 检查与测试

不加载模型时可以先做语法检查：

```bash
uv run python -m compileall -q experiment utils
```

项目配置了 pytest，但当前仓库还没有 `tests/` 下的测试文件，因此 `pytest` 暂时不能提供训练逻辑的回归保障。

## 已知问题

1. 验证期间的最佳模型保存发生在验证 batch 循环内部，可能重复写入大 checkpoint；同时它会
   删除之前保存到同一目录的 tokenizer，最终目录可能只有模型权重。
2. GPU、端口、模型和实验参数均为硬编码，且没有配置校验；未知的算法名会导致变量未初始化。
3. 仓库没有自动化测试。数学判分器还包含多处裸 `except` 和非 raw regex 字符串，可能掩盖异常，
   并会在新版 Python 中产生无效转义警告。

## 结果复现提示

- vLLM sampling、数据顺序都使用 `seed=42`，但训练代码没有显式设置 PyTorch/CUDA 的随机种子，
  因此当前结果并非完全可复现。
- 保留 W&B run 配置、Git commit、GPU 型号和依赖版本，便于比较不同算法或 seed。
- 开始长时间训练前，建议先用很小的数据量和生成长度完成一次 rollout、反向传播、权重同步和验证
  的端到端 smoke test。
