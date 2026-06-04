# GRPO GSM8K Math Reasoning Project

本项目是一个面向 GSM8K 数学推理任务的 **GRPO 强化学习训练系统**。项目以 `Qwen2.5-1.5B-Instruct` 为基座模型，围绕 SFT warmup、GRPO 组内相对优势、多奖励函数、reward hacking 诊断和批量消融实验构建完整实验链路。

当前仓库已经收敛为 GRPO-only 项目：所有训练、消融、评估和结果记录都围绕 GRPO 展开。

## 项目目标

- 使用 SFT warmup 建立 GSM8K 解题格式和基础数学能力。
- 使用 GRPO 对同一 prompt 采样多条 response，并基于组内相对奖励计算 advantage。
- 通过 correctness、format、reasoning 三类奖励分析模型行为。
- 区分 legacy reasoning 与 answer-conditioned reasoning，避免把格式刷分误判为推理能力提升。
- 在结果 JSON 中记录完整实验元数据，保证实验可复查。

## 核心方法

### GRPO

GRPO 使用同一 prompt 的 K 条 completion 组成一个 group，并在组内标准化奖励：

```text
advantage_i = (reward_i - mean(group_rewards)) / std(group_rewards)
```

本项目中的 GRPO trainer 位于：

```text
scripts/train_grpo.py
scripts/grpo_core.py
```

当前实现是单进程/单卡 trainer，适合 Qwen2.5-1.5B 级别模型的研究型训练与消融实验。它不依赖 critic/value model。

### Reward 模式

项目明确区分两套 reasoning reward：

| 模式 | 配置值 | 用途 |
|------|--------|------|
| legacy_reasoning | `reward_mode: legacy_reasoning` | warmup 或对照实验；推理分只看长度和步骤结构，不要求答案正确 |
| answer_conditioned_reasoning | `reward_mode: answer_conditioned_reasoning` | 最终诊断和防 reward hacking；只有答案正确时才给 reasoning reward |

默认训练配置使用 `legacy_reasoning`，原因是 GRPO 早期训练需要更密的结构信号；最终评估和 reward hacking 诊断建议使用 `answer_conditioned_reasoning` 复算 reward breakdown。

### 多奖励函数

奖励由三部分组成：

```text
reward = wc * correctness + wf * format + wr * reasoning
```

- `correctness`：答案抽取与 ground truth 比较，默认使用 flexible extraction，不依赖 `####`。
- `format`：检查步骤结构、计算过程和答案标记，与答案正确性解耦。
- `reasoning`：支持 legacy 或 answer-conditioned 两种模式。

## 项目结构

```text
RL_verl/
├── config/
│   ├── grpo_gsm8k.yaml
│   └── experiments/
│       ├── exp1_correctness_only.yaml
│       ├── exp2_multi_reward_default.yaml
│       ├── exp3_heavy_reasoning.yaml
│       ├── exp4_balanced.yaml
│       ├── exp5_format_emphasis.yaml
│       ├── exp6_answer_conditioned_diagnostic.yaml
│       └── grpo_default.yaml
├── verl_rewards/
│   ├── correctness.py
│   ├── format.py
│   ├── reasoning.py
│   └── composite.py
├── scripts/
│   ├── train_grpo.py
│   ├── train_grpo.sh
│   ├── run_ablation_experiments.sh
│   ├── evaluate.py
│   ├── evaluate_all_models.sh
│   ├── reward_fn.py
│   ├── grpo_core.py
│   ├── sft_warmup.py
│   └── analyze_metrics.py
└── results/
```

## 环境安装

```bash
conda create -n verl python=3.10 -y
conda activate verl

pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

pip install transformers datasets pandas tqdm omegaconf matplotlib
```

当前 GRPO trainer 默认使用 vLLM rollout。为了避免 vLLM 一直从初始权重采样，训练器会周期性将当前 policy 保存到 `_vllm_sync/policy_current`，并从该 checkpoint 重载 vLLM engine。

如果只想调试链路，可以临时使用 `--no-vllm` 回退到 HuggingFace `generate`。

## 数据准备

使用 verl 官方 GSM8K 预处理脚本生成 parquet：

```bash
cd ~/verl
python examples/data_preprocess/gsm8k.py --local_save_dir $HOME/data/gsm8k/
```

默认路径：

```text
$HOME/data/gsm8k/train.parquet
$HOME/data/gsm8k/test.parquet
```

## 训练流程

### 1. SFT Warmup

```bash
python scripts/sft_warmup.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --train-data $HOME/data/gsm8k/train.parquet \
  --val-data $HOME/data/gsm8k/test.parquet \
  --output ./checkpoints/sft_warmup \
  --epochs 3 \
  --lr 2e-5
```

输出：

```text
checkpoints/sft_warmup/final
```

### 2. GRPO 训练

```bash
./scripts/train_grpo.sh \
  --model ./checkpoints/sft_warmup/final \
  --use-vllm \
  --vllm-sync-interval 1 \
  --vllm-gpu-memory-utilization 0.25 \
  --group-size 4 \
  --batch-size 2 \
  --max-response-length 256 \
  --total-steps 1000 \
  --w-correctness 0.7 \
  --w-format 0.1 \
  --w-reasoning 0.2
```

也可以直接运行：

```bash
python scripts/train_grpo.py \
  --config config/grpo_gsm8k.yaml \
  --model ./checkpoints/sft_warmup/final \
  --use-vllm \
  --vllm-sync-interval 1 \
  --group-size 4 \
  --batch-size 2 \
  --total-steps 1000
```

### 3. GRPO 消融实验

```bash
./scripts/run_ablation_experiments.sh \
  --model ./checkpoints/sft_warmup/final \
  --total-steps 1000
```

只查看计划：

```bash
./scripts/run_ablation_experiments.sh --dry-run
```

## 推荐配置

| 项目 | 推荐值 |
|------|--------|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| SFT checkpoint | `./checkpoints/sft_warmup/final` |
| Group size | 4 |
| Train batch size | 2 prompts/step |
| Max response length | 256 |
| Temperature | 0.7 |
| Top-p | 0.95 |
| KL coef | 0.001 |
| Rollout engine | vLLM |
| vLLM sync interval | 1 step |
| vLLM memory utilization | 0.25 |
| Reward mode | `legacy_reasoning` for training, `answer_conditioned_reasoning` for final diagnostics |

如果出现 OOM：

```bash
./scripts/train_grpo.sh \
  --use-vllm \
  --vllm-gpu-memory-utilization 0.15 \
  --group-size 2 \
  --batch-size 1 \
  --max-response-length 128 \
  --model ./checkpoints/sft_warmup/final
```

## 评估

```bash
python scripts/evaluate.py \
  --model-path ./checkpoints/verl_grpo/gsm8k_grpo_multi_reward/final \
  --test-data $HOME/data/gsm8k/test.parquet \
  --output results/eval_w0.7_0.1_0.2.json \
  --temperature 0 \
  --batch-size 8
```

评估结果 JSON 会记录：

- `base_model`
- `sft_checkpoint`
- `grpo_config`
- `reward_mode`
- `group_size`
- `batch_size`
- `temperature`
- `kl_coef`
- `rollout_engine`
- `vllm_sync_interval`
- `vllm_gpu_memory_utilization`
- reward weights
- summary metrics

## 结果文件

新生成的 `results/*.json` 会在顶层记录实验元数据，用于复现实验环境和训练设置。示例：

```json
{
  "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
  "sft_checkpoint": "./checkpoints/sft_warmup/final",
  "grpo_config": "config/grpo_gsm8k.yaml",
  "reward_mode": "legacy_reasoning",
  "group_size": 4,
  "batch_size": 2,
  "temperature": 0.7,
  "kl_coef": 0.001,
  "rollout_engine": "vllm",
  "vllm_sync_interval": 1,
  "vllm_gpu_memory_utilization": 0.25
}
```

仓库中如果保留了历史结果文件，不应在未重新评估的情况下改写其模型路径或指标。新的 Qwen2.5-1.5B GRPO 结果应通过 `scripts/evaluate.py` 重新生成，或至少遵循 [results/qwen2.5-1.5b-grpo_result_schema.json](results/qwen2.5-1.5b-grpo_result_schema.json) 中的字段约定。

## 已知边界

- 当前 GRPO trainer 是单进程/单卡实现，不支持 `torchrun` 多进程。
- vLLM 使用 checkpoint-based sync。`vllm_sync_interval=1` 最接近 on-policy，但会频繁保存/重载 engine；增大该值会提升吞吐，但 rollout 会更 stale。
- 如果 vLLM 与 policy/ref/optimizer 同卡导致 OOM，优先降低 `--vllm-gpu-memory-utilization`，其次降低 batch/group/response length。
- `legacy_reasoning` 可以提供训练信号，但会高估错误答案的 reasoning score；最终报告应使用 answer-conditioned reward breakdown 进行诊断。
- 结果指标应和对应的模型、reward mode、config 一起报告，避免把格式学习误判为数学能力提升。

## 许可证

MIT License
