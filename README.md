# verl Multi-Reward RL Training Project

本项目基于 `verl` 框架，为 GSM8K 数学推理任务实现多奖励 RL 训练，并新增一个独立的 GRPO 原型训练器。当前版本重点修复了早期设计中“格式奖励被重复计数”“推理奖励可被格式刷分”“baseline 过弱导致结论偏乐观”等问题。

## 当前状态

### 奖励系统 v2

旧版奖励存在两个关键问题：

- `correctness(strict)` 和 `format(gsm8k)` 使用同一个 `#### NUMBER` 正则，导致格式缺失时正确性和格式同时归零。
- `reasoning_reward` 实际只衡量长度和步骤标记，模型可以用格式化但错误的推理获得高分。

v2 的默认行为：

- `correctness_method="flexible"`：正确性提取不再依赖 `####` 格式。
- `format_reward`：检查步骤标记、计算过程、答案标记，而不是只检查 `####`。
- `answer_conditioned_reasoning=True`：只有答案正确时才给推理结构奖励，错误答案的推理分为 0。

### GRPO 训练器

新增 `scripts/train_grpo.py` 和 `scripts/grpo_core.py`，实现 Group Relative Policy Optimization：

```text
advantage = (reward - group_mean) / group_std
```

当前 GRPO 训练器是**单进程/单卡可信实现**：

- rollout 使用当前 live policy，避免静态 vLLM 一直从初始权重采样。
- 训练 logprob 基于 `prompt + response`，loss 只作用于 response token。
- 默认不启用 vLLM；如需 vLLM，需要实现权重同步后再打开。
- 不支持 `torchrun` 多进程训练；多卡训练建议继续使用 PPO/verl 脚本。

### SFT 预热

新增 `scripts/sft_warmup.py`，用于在 RL 前建立更合理的数学 baseline：

- 支持常见 GSM8K/verl parquet schema，包括 chat prompt 和 `reward_model.ground_truth`。
- SFT labels 会 mask prompt token，只训练 response 部分。
- 使用 Transformers 新 API：`eval_strategy`、`processing_class`、`warmup_steps`。

### 推荐硬件：4x A100

当前推荐在 **4x A100 40GB/80GB** 上运行本项目：

- PPO/verl：使用 4 卡 FSDP/vLLM，是主要训练路径。
- SFT warmup：单卡或 4 卡机器均可轻松完成。
- GRPO 原型：当前仍是单进程/单卡 trainer，建议在 4 卡机器上指定 1 张空闲 A100 跑验证。

V100-32GB 仍可用于 smoke test 和小规模验证，但吞吐、数值稳定性和 flash-attn/vLLM 支持都弱于 A100。

## 项目结构

```text
RL_verl/
├── config/
│   ├── ppo_gsm8k.yaml
│   ├── grpo_gsm8k.yaml
│   └── experiments/
│       ├── exp1_correctness_only.yaml
│       ├── exp2_multi_reward_default.yaml
│       ├── exp3_heavy_reasoning.yaml
│       ├── exp4_balanced.yaml
│       ├── exp5_xml_format.yaml
│       ├── exp6_legacy_comparison.yaml
│       └── grpo_default.yaml
├── verl_rewards/
│   ├── correctness.py
│   ├── format.py
│   ├── reasoning.py
│   └── composite.py
├── scripts/
│   ├── reward_fn.py
│   ├── multi_reward_manager.py
│   ├── grpo_core.py
│   ├── train_grpo.py
│   ├── train_grpo.sh
│   ├── sft_warmup.py
│   ├── train_ppo.sh
│   ├── train_4gpu.sh
│   ├── train_8gpu.sh
│   ├── train_quick.sh
│   ├── evaluate.py
│   └── analyze_metrics.py
├── results/
├── outputs/
└── checkpoints/
```

## 安装

```bash
conda create -n verl python=3.10 -y
conda activate verl

pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

pip install transformers datasets pandas tqdm omegaconf
```

如需使用 PPO/verl：

```bash
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .
pip install -r requirements.txt
```

`vllm` 是可选依赖。当前 GRPO 训练器默认使用 HF policy rollout，不依赖 vLLM。

## 数据准备

使用 verl 官方 GSM8K 预处理脚本：

```bash
cd ~/verl
python examples/data_preprocess/gsm8k.py --local_save_dir $HOME/data/gsm8k/
```

默认数据路径：

```text
$HOME/data/gsm8k/train.parquet
$HOME/data/gsm8k/test.parquet
```

## 推荐训练流程

### 1. SFT 预热

```bash
cd ~/RL_verl

python scripts/sft_warmup.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --train-data $HOME/data/gsm8k/train.parquet \
  --val-data $HOME/data/gsm8k/test.parquet \
  --output ./checkpoints/sft_warmup \
  --epochs 3 \
  --lr 2e-5
```

输出模型位于：

```text
checkpoints/sft_warmup/final
```

### 2. GRPO 单卡验证

当前 GRPO 是单进程训练器，即使机器有 4 张 A100，也只使用 1 张 GPU。建议先用它验证 GRPO 奖励/训练链路：

```bash
./scripts/train_grpo.sh \
  --n-gpus 1 \
  --model ./checkpoints/sft_warmup/final \
  --group-size 4 \
  --batch-size 4 \
  --max-response-length 384 \
  --total-steps 1000 \
  --w-correctness 0.7 \
  --w-format 0.1 \
  --w-reasoning 0.2
```

也可以直接运行 Python：

```bash
python scripts/train_grpo.py \
  --config config/grpo_gsm8k.yaml \
  --model ./checkpoints/sft_warmup/final \
  --group-size 4 \
  --total-steps 1000
```

注意：`scripts/train_grpo.py` 会解析 OmegaConf 风格配置，例如 `${oc.env:HOME}` 和 `${trainer.project_name}`。

### 3. PPO 多卡训练

4x A100 推荐使用 PPO/verl 作为正式多卡训练路径：

```bash
./scripts/train_4gpu.sh \
  --gpus 4 \
  --batch-size 512 \
  --model ./checkpoints/sft_warmup/final
```

或：

```bash
export PYTHONPATH="${PWD}:${PYTHONPATH}"
python -m verl.trainer.main_ppo \
  --config-path=config \
  --config-name=ppo_gsm8k
```

## GRPO vs PPO

| 项目 | PPO/verl | 当前 GRPO trainer |
|------|----------|-------------------|
| 训练入口 | `python -m verl.trainer.main_ppo` | `python scripts/train_grpo.py` |
| 4x A100 支持 | 推荐 | 仅使用单卡 |
| Critic | 需要 | 不需要 |
| Rollout | vLLM/verl worker | live HF policy |
| 优势估计 | GAE | 组内相对优势 |
| KL | verl KL controller | loss 内 KL penalty |
| 适合场景 | 4卡正式训练 | 单卡验证 GRPO 思路 |

GRPO 默认配置：

```yaml
algorithm:
  type: grpo
  group_size: 4
  advantage:
    method: group_relative
    norm_method: standardize
  kl_penalty:
    kl_coef: 0.001
    estimator: k3
```

`k3` 是非负 KL 近似，比直接 sampled log-ratio 更稳。

## 奖励函数说明

综合奖励：

```python
reward = (
    w_correctness * correctness_reward +
    w_format * format_reward +
    w_reasoning * reasoning_reward
)
```

默认权重：

```text
w_correctness = 0.7
w_format      = 0.1
w_reasoning   = 0.2
```

组件说明：

| 组件 | v1 行为 | v2 行为 |
|------|---------|---------|
| correctness | strict 模式要求 `#### NUMBER` | flexible 模式，提取最后有效数字 |
| format | 只检查 `#### NUMBER` | 检查步骤、计算、答案标记 |
| reasoning | 长度 + 步骤启发式 | 答案正确时才给结构奖励 |

旧版对照实验仍保留：

```bash
./scripts/train_ppo.sh --config experiments/exp6_legacy_comparison
```

## 评估

### 普通评估

```bash
python scripts/evaluate.py \
  --model-path checkpoints/sft_warmup/final \
  --test-data $HOME/data/gsm8k/test.parquet \
  --output results/sft_eval.json \
  --batch-size 16
```

### V100 / fp16 评估

```bash
python scripts/evaluate.py \
  --model-path checkpoints/sft_warmup/final \
  --test-data $HOME/data/gsm8k/test.parquet \
  --dtype fp16 \
  --output results/sft_eval_v100.json
```

`--dtype auto` 的策略：

- Ampere+ GPU：`bf16`
- V100/更老 GPU：`fp16`

### 主要指标

| 指标 | 含义 |
|------|------|
| `accuracy` | 最终答案正确率 |
| `correctness_score` | 正确性奖励 |
| `format_score` | 输出结构奖励 |
| `reasoning_score` | 答案条件推理结构奖励 |
| `score` | 综合奖励 |

## 已知限制

1. **GRPO 多卡未实现**
   当前 `scripts/train_grpo.py` 会拒绝 `torchrun` 多进程启动。需要 DDP/FSDP 后才能可靠多卡。

2. **GRPO vLLM 默认关闭**
   静态 vLLM 不会自动同步 policy 权重。没有权重同步时，rollout 会来自旧模型，实验结论不可信。

3. **推理奖励仍是弱代理**
   v2 用正确性门控降低 reward hacking，但它仍不是严格的推理质量判别器。更强方案可以引入 verifier 或 LLM-as-judge。

4. **历史 results 为 v1 奖励结果**
   `results/` 下旧 JSON 使用 v1 奖励逻辑。用 v2 重新评估时，`reasoning_score` 会明显变化。

## 常见问题

### `TrainingArguments.__init__()` 参数报错

本项目已使用较新的 Transformers API：

- `eval_strategy`
- `processing_class`
- `warmup_steps`

如果仍报错，请检查本地 `transformers` 版本。

### V100 上 bf16 报错或很慢

使用：

```bash
python scripts/evaluate.py ... --dtype fp16
```

训练脚本默认使用 fp16。

### 4x A100 推荐参数

PPO 正式训练：

```bash
./scripts/train_4gpu.sh \
  --gpus 4 \
  --batch-size 512 \
  --model ./checkpoints/sft_warmup/final
```

GRPO 单卡验证：

```bash
./scripts/train_grpo.sh \
  --n-gpus 1 \
  --batch-size 4 \
  --group-size 4 \
  --max-response-length 384 \
  --model ./checkpoints/sft_warmup/final
```

### GRPO OOM

降低 batch 或 group size：

```bash
./scripts/train_grpo.sh --n-gpus 1 --group-size 2
```

### PPO OOM

降低 PPO batch/micro batch：

```bash
./scripts/train_4gpu.sh --batch-size 512
```

## 许可证

MIT License
