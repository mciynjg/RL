# Reasoning RL 实验框架

这是一个面向 GSM8K 数学推理的强化学习实验项目。训练入口以
[`allenai/OLMo-2-0425-1B`](https://huggingface.co/allenai/OLMo-2-0425-1B)
为默认策略模型：PyTorch/Transformers 在一张 GPU 上训练，vLLM 在另一张 GPU 上批量采样，
并通过 NCCL 将更新后的策略权重同步回推理服务。

项目目前是实验原型：训练配置写在源码中，运行会启动真实 vLLM、占用 GPU，并可能终止同端口的
既有 vLLM 服务。开始训练前请阅读 [训练前须知](#训练前须知) 和
[当前限制](#当前限制)。

## 主要能力

- 使用 vLLM 为每个数学问题生成一组 rollout，并用 GSM8K 标准答案给分。
- 使用严格的 R1-zero 输出契约：`<think>...</think> <answer>...</answer>`。
- 支持 GRPO、`Dr_GRPO`、RFT、MaxRL，以及四个实验性 off-policy 配置。
- 对退化 rollout group 进行过滤，并记录奖励、裁剪比例、梯度范数、entropy、长度、截断率，以及 off-policy 的 ratio、近似 KL 和 ESS。
- 每 10 个外层 rollout 在 GSM8K 测试集上验证，并在指标改善时保存模型权重。
- 提供不加载模型的 CPU smoke 测试，以及一个显式开启的 GPU/vLLM 端到端测试。

## 目录结构

```text
.
├── data/
│   ├── gsm8k/                 # 当前训练和验证实际使用的数据
│   └── mmlu/、hh/、...         # 已收录但尚未接入训练入口的数据
├── experiment/
│   └── train.py               # 训练入口和全部实验常量
├── prompts/
│   ├── r1_zero.prompt         # 训练与 R1-zero 验证模板
│   ├── r1_zero_three_shot_gsm8k.prompt
│   └── question_only.prompt
├── results/
│   ├── checkpoint/            # 训练产生的 checkpoint
│   └── eval result/           # 三种基础模型评估的 JSONL 输出
├── tests/
│   ├── test_smoke.py          # CPU、模型无关的测试
│   └── test_e2e.py            # 可选真实 vLLM 测试
├── utils/
│   ├── dataset.py             # 推理和训练数据集包装
│   ├── grader.py              # 格式检查、答案提取与数学判分
│   ├── helper.py              # tokenization、奖励、loss 与训练步骤
│   ├── config_validation.py   # 训练配置的启动前校验
│   ├── vllm_utils.py          # vLLM 生命周期、批量生成与 NCCL 同步
│   └── eval_*.py              # 三种基础模型评估脚本
├── pyproject.toml             # Python 版本、依赖和 pytest 配置
└── uv.lock                    # 锁定的可运行环境
```

当前训练只读取 `data/gsm8k/{train,test}.jsonl` 与 `prompts/r1_zero.prompt`。

## 训练前须知

完整训练需要以下环境：

- Linux x86_64、Python 3.12（项目限制为 `>=3.12,<3.13`）。
- CUDA-capable NVIDIA GPU；默认训练模型位于 `cuda:2`，vLLM 使用物理 GPU `3`。
- 可访问 Hugging Face 以下载模型，以及可用的 CUDA、vLLM、NCCL 运行环境。
- W&B 凭据，或设置 `WANDB_MODE=offline` 使用离线记录。

默认的 GPU 编号意味着机器至少需要可用的 GPU 2 和 GPU 3。换机器前，请同时修改
`experiment/train.py` 中的 `training_device` 与 `vllm_device`。评估脚本没有覆盖这些设置，
它们使用 `VLLMServer` 的默认 GPU 0。

训练入口会先向操作系统申请一个空闲本地端口；评估脚本则使用默认端口 `8000`。无论端口来自哪里，
`VLLMServer.start()` 都会尝试终止同端口、匹配的已有 vLLM 服务，因此不要与其他使用该端口的
工作负载并行运行。

## 安装

推荐使用 [uv](https://docs.astral.sh/uv/) 依据锁文件安装项目：

```bash
uv python install 3.12
uv sync
uv run wandb login
```

Linux 环境会安装 CUDA 12.9 对应的 PyTorch、vLLM 0.19.1 和项目指定的 FlashAttention wheel，
下载量较大。无需在线记录实验时：

```bash
export WANDB_MODE=offline
```

## 训练配置

所有训练参数都是 [`experiment/train.py`](experiment/train.py) 顶部的模块级常量，而不是 CLI
参数。默认值如下：

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `training_device` | `cuda:2` | PyTorch 策略模型所在 GPU |
| `vllm_device` | `3` | vLLM 子进程可见的物理 GPU |
| `vllm_gpu_memory_utilization` | `0.75` | vLLM 的显存使用上限 |
| `model_name` | `allenai/OLMo-2-0425-1B` | Hugging Face 模型 ID |
| `seed` | `42` | Python、NumPy、PyTorch 与 vLLM sampling 的种子 |
| `rollout_batch_size` | `256` | 每个外层 rollout 中的 completion 总数 |
| `group_size` | `8` | 每条问题采样的 completion 数 |
| `inference_batch_size` | `32` | 每轮送到 vLLM 的问题数，即 `256 / 8` |
| `off_policy_speedup_factor` | `1` | 将一个 rollout 拆成连续训练 batch 的倍数 |
| `training_batch_size` | `256` | `rollout_batch_size / off_policy_speedup_factor` |
| `gradient_accumulation_steps` | `32` | `32 / off_policy_speedup_factor` |
| `sampling_temperature` | `1.0` | 训练 rollout 的采样温度 |
| `sampling_max_tokens` | `512` | 单条 completion 的最大生成 token 数 |
| `num_training_steps` | `200` | 训练更新的名义上限 |
| `val_batch_size` | `64` | 验证阶段每次生成的问题数 |
| `advantage_algorithm` | `GRPO` | 下表中的策略梯度配置名 |

训练在连接 W&B、加载模型或启动 vLLM 前会校验配置。`rollout_batch_size` 必须能被
`group_size` 和 `off_policy_speedup_factor` 整除；`inference_batch_size * group_size` 必须等于
`rollout_batch_size`；`training_batch_size` 必须等于
`rollout_batch_size / off_policy_speedup_factor`，且是 `group_size` 的整数倍。`gradient_accumulation_steps` 必须在
`[1, training_batch_size]` 内。默认超参数下，`1`、`2`、`4`、`8`、`16`、`32` 是满足这些
batch 约束的 `off_policy_speedup_factor` 选择。

### 策略梯度配置

`ADVANTAGE_ALGORITHM_CONFIG` 定义了以下实现配置。`constant` 的归一化常数为
`rollout_batch_size * sampling_max_tokens`，默认值为 `131072`。

| 配置名 | baseline / advantage 归一化 | loss 归一化 | importance reweighting |
| --- | --- | --- | --- |
| `GRPO` | group mean / group std | sequence | 无 |
| `GRPO_constant` | group mean / group std | constant | 无 |
| `Dr_GRPO` | group mean / 无 | constant | 无 |
| `RFT` | 无 / 无 | constant | 无 |
| `MaxRL` | group mean / group mean | constant | 无 |
| `offpolicy_naive` | group mean / group std | sequence | 无 |
| `offpolicy_noclip` | group mean / group std | sequence | token ratio，无裁剪 |
| `offpolicy_grpo` | group mean / group std | sequence | token ratio，裁剪到 `1 ± 0.2` |
| `offpolicy_gspo` | group mean / group std | sequence | sequence 几何均值 ratio，裁剪到 `1 ± 3e-4` |

当 importance reweighting 非 `none` 时，代码会在连续更新前保存旧策略的 token log-probabilities。
这些 off-policy 配置是仓库当前的实验实现；应结合具体实验设定验证其行为，而不应仅依据名称把它们
视为某篇论文的完整复现。

off-policy 训练只在 CPU 上以 float32 缓存 response token 的旧 log-probabilities，训练 microbatch
时才临时搬回策略设备。显存节省来自不保存 prompt/padding token；W&B 会额外记录
`offpolicy/ratio_mean`、`ratio_std`、`ratio_min`、`ratio_max`、
`approx_kl`、`clip_fraction`、`ess`、`ess_fraction` 和 `sample_count`；其中 ESS 是按当前
importance-sample 粒度计算的有效样本数，GSPO 的粒度为 response sequence，其他裁剪方式为
response token。

## 启动训练

先在 `experiment/train.py` 中调整设备、模型、batch size 和算法名，再从项目根目录运行：

```bash
uv run python experiment/train.py
```

一次训练的大致流程为：

1. 分配一个空闲端口，启动本地 OpenAI-compatible vLLM server，并加载策略模型。
2. 读取 GSM8K 训练集与 R1-zero prompt；`DataLoader(shuffle=True)` 产生问题 batch。
3. 为每个问题生成 `group_size` 个回答，计算格式奖励和答案奖励。
4. 按 `off_policy_speedup_factor` 将 rollout 切分为训练 batch，进行策略梯度更新。
5. 每个外层 rollout 结束后通过 NCCL 同步权重；每 10 个外层 rollout 还会在同步后运行验证。
6. 验证总奖励改善时保存模型权重，训练结束时保存 tokenizer。

checkpoint 路径由 W&B run 名组成：

```text
results/checkpoint/<algorithm>-<model>-seed<seed>/
```

训练脚本不是轻量级示例：导入或执行它都会初始化 W&B、加载模型、启动 vLLM 并访问 GPU。

## 奖励契约与数据

训练使用 `data/gsm8k/train.jsonl`（1,421 条）生成 rollout，使用
`data/gsm8k/test.jsonl`（1,319 条）做验证。每行包含问题与带 `####` 标记的解答：

```json
{"question": "...", "answer": "reasoning ... #### final answer"}
```

`helper.extract_ground_truth()` 读取 `####` 后的文本作为标准答案。

R1-zero prompt 已经以 `<think>` 结尾，因此生成的 completion 必须形如：

```text
reasoning</think> <answer>final answer</answer>
```

判分器要求恰好一个 `</think>`、紧随其后的一个字面空格、以及恰好一个 `<answer>...</answer>`
块。它返回始终具有以下键的字典：`format_reward`、`answer_reward` 与 `reward`。格式正确但答案
错误时，前两项分别为 `1.0` 和 `0.0`，总奖励为 `0.0`。

数学答案由 [`utils/grader.py`](utils/grader.py) 验证，涵盖常见数值、分数、LaTeX 与受限的符号
比较。对于使用 group-mean baseline 的配置，奖励在整个 group 中完全相同的 rollout 会被过滤；
`RFT` 则过滤奖励为零的 rollout。

## 基础模型评估

三个评估脚本都在 GSM8K `train.jsonl` 的 1,421 个问题上运行，并将固定 JSONL 文件覆盖写入
`results/eval result/`：

| 脚本 | Prompt | 奖励函数 |
| --- | --- | --- |
| `utils/eval_question_only.py` | 只给问题，要求 `\boxed{}` | `question_only_reward_fn` |
| `utils/eval_r1_zero.py` | R1-zero zero-shot | `r1_zero_reward_fn` |
| `utils/eval_rl_three_shot.py` | 三个 GSM8K 示例 + R1-zero | `r1_zero_reward_fn` |

从根目录运行：

```bash
uv run python utils/eval_question_only.py
uv run python utils/eval_r1_zero.py
uv run python utils/eval_rl_three_shot.py
```

仓库中的现有结果如下：

| 设置 | 答案正确率 | 格式通过率 |
| --- | ---: | ---: |
| Question only | 0.63% | 25.69% |
| R1-zero zero-shot | 0.14% | 59.11% |
| R1-zero three-shot | 21.75% | 96.41% |

每条 JSONL 记录问题、标准答案、completion、三个 reward 字段和类别。评估会启动真实 vLLM，
下载模型或占用 GPU；它不是 CPU smoke 测试。

## 检查与测试

不加载模型、不启动 vLLM 的语法检查：

```bash
uv run python -m compileall -q experiment utils
```

CPU/model-independent 测试覆盖包导入、GSM8K/prompt 读取、随机种子、tokenization、奖励和
advantage、微型策略更新、数学判分及 vLLM HTTP completion 适配：

```bash
uv run pytest -q
uv run pytest -q tests/test_smoke.py
```

`tests/test_e2e.py` 默认跳过。它会下载一个 Hugging Face 小模型、启动真实 vLLM 并生成一条
completion；只有在具备 CUDA、网络和可用显存时才显式开启：

```bash
RUN_E2E=1 E2E_GPU=0 uv run pytest -q tests/test_e2e.py -m e2e
```

默认模型为 `HuggingFaceTB/SmolLM2-135M`，可通过 `E2E_MODEL_ID` 覆盖。

## 当前限制

- 训练数据会在迭代器耗尽后重新创建并继续随机采样，因此外层 rollout 会恰好运行
  `num_training_steps` 次。一个外层 rollout 可按 `off_policy_speedup_factor` 拆为多个连续更新。
- 训练配置、GPU 设备、模型和 W&B 信息均硬编码在模块顶层；启动前会校验 batch、数值范围和算法
  配置之间的依赖关系。
- 只有验证总奖励严格高于初始值时才保存模型权重；tokenizer 会在训练结束时无条件保存。若没有正向
  验证奖励，checkpoint 目录可能只包含 tokenizer 文件。
- 评估脚本会覆盖固定结果文件，且 vLLM 启动逻辑会终止匹配端口的已有服务。请避免与其他实验共享
  目标目录或端口。

## 可复现性提示

`helper.seed_everything(seed)` 设置 Python、NumPy、PyTorch CPU/CUDA 种子，关闭 cuDNN benchmark，
启用 cuDNN deterministic 和 PyTorch 确定性算法。更严格的运行可复现性还需要在启动 Python 前设置
哈希和 cuBLAS 环境变量：

```bash
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run python experiment/train.py
```

`PYTHONHASHSEED` 必须在解释器启动前生效。FlashAttention、vLLM、NCCL、驱动和多 GPU 通信仍可能
引入数值差异，因此该项目不保证多 GPU 训练 bitwise 一致。保留 W&B 配置、Git commit、GPU 型号和
依赖版本，才能可靠比较不同算法或 seed 的实验结果。
