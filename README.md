# verl Multi-Reward RL Training Project (v2)

这个项目在 verl 官方框架的基础上，实现了多奖励组合系统和推理结构奖励，用于 GSM8K 数学问题的 RL 训练。v2 新增 **GRPO（Group Relative Policy Optimization）** 算法支持及奖励系统重构。

## V2 更新亮点

### 奖励系统 v2：消除双重计数，引入答案条件推理奖励

**v1 的问题：**
- `correctness.py` (strict 模式) 和 `format.py` (gsm8k 模式) 使用**完全相同的正则** `####\s*([\-]?[0-9\.\,]+)`，导致正确答案若无 `####` 格式则两个奖励同时归零
- `reasoning_reward` 只测量输出长度和步骤标记数（启发式），基线模型在 6% 准确率时就能拿到 **0.87 推理分**——模型可通过生成格式漂亮但内容错误的"伪推理"来获得高奖励

**v2 的修复：**
- **正确性提取**：默认改用 `flexible` 模式，不依赖 `####` 格式即可提取答案
- **格式奖励**：改为评估整体输出结构（步骤标记 + 计算过程 + 答案标记），不再只检查 `####`
- **推理奖励**：引入答案条件门控——`answer_conditioned_reasoning = True`（默认），错误答案的推理分强制归零，防止 reward hacking

### GRPO 算法：无需 Critic 模型

GRPO 是 DeepSeek-R1 的核心对齐算法。相比 PPO 省去 Critic 模型：
- **VRAM 节省**：0.5B 模型约省 17%，7B+ 模型约省 50%
- **优势计算**：组内相对优势 `(R - μ_group) / σ_group`，不依赖 GAE 和价值函数估计
- **KL 惩罚**：直接嵌入损失函数，无需单独 KL 控制器

### SFT 预热基线

新增 `scripts/sft_warmup.py`：在 RL 训练前先用 GSM8K 数据进行监督微调，建立有基本数学能力的基线。v1 的 6%→50% 提升大部分来自学会输出格式，SFT 预热后 RL 阶段的提升可真正归因于推理质量改善。

## 项目特点

### 改进 1：多奖励组合系统 (Multi-Reward)

多维度奖励组合，各组件相互独立：

```python
reward = (
    w_correctness * correctness_reward +  # 答案正确性 (flexible 提取，不依赖格式)
    w_format * format_reward +             # 输出结构 (步骤+计算+答案标记)
    w_reasoning * reasoning_reward         # 推理结构 (答案条件门控)
)
```

### 改进 2：答案条件推理奖励 (Answer-Conditioned Reasoning)

只有正确答案才能获得推理结构分，防止模型生成格式化但内容错误的伪推理：

| Reward 组件 | 作用 | v2 改进 |
|------------|------|---------|
| `correctness_reward` | 答案是否正确 | flexible 提取（不要求 `####`） |
| `format_reward` | 输出结构质量 | 综合评估步骤+计算+答案标记 |
| `reasoning_reward` | 推理过程结构 | 答案条件门控（错误答案=0） |

### 改进 3：GRPO 算法选项

| 特性 | PPO | GRPO |
|------|-----|------|
| 模型数量 | Actor + Critic + Ref + Reward (4个) | Actor + Ref + Reward (3个) |
| 优势估计 | GAE (需要 Critic) | 组内标准化 (无需 Critic) |
| KL 控制 | 独立 KL 控制器 | 直接嵌入损失函数 |
| VRAM 占用 | 较高 | 更低 (7B 约省 50%) |

### 改进 4：可视化与实验对比

- 详细的 reward 分解日志
- KL / Entropy / Advantage 追踪
- 多实验对比分析工具
- GRPO vs PPO 对比实验支持

## 硬件配置

- **推荐配置**: 8x NVIDIA A100 80GB
- **最低配置**: 1x GPU with 24GB+ VRAM
- **GRPO 最低配置**: 1x GPU with 16GB+ VRAM (比 PPO 省显存)

## 示例安装

```bash
conda create -n verl python=3.10 -y
conda activate verl
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121
pip install vllm
pip install flash-attn --no-build-isolation
```

```bash
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .
pip install -r requirements.txt
```

```bash
git clone https://github.com/irroca/RL_verl.git
cd RL_verl
```

## 项目结构

```
RL_verl/
├── config/
│   ├── ppo_gsm8k.yaml                   # PPO 主配置文件 (8-GPU优化)
│   ├── grpo_gsm8k.yaml                  # GRPO 主配置文件 (无需Critic)
│   └── experiments/                      # 实验配置
│       ├── exp1_correctness_only.yaml    # 基线：仅正确性
│       ├── exp2_multi_reward_default.yaml # 默认多奖励 (v2: flexible + 答案条件)
│       ├── exp3_heavy_reasoning.yaml     # 高推理权重
│       ├── exp4_balanced.yaml            # 平衡配置
│       ├── exp5_xml_format.yaml          # XML 结构化输出
│       ├── exp6_legacy_comparison.yaml   # v1 旧版对照 (strict + 非门控)
│       └── grpo_default.yaml             # GRPO 默认实验
├── verl_rewards/                         # 多奖励系统 (v2)
│   ├── __init__.py
│   ├── correctness.py                   # 正确性奖励 (flexible/strict)
│   ├── format.py                        # 格式奖励 (独立于正确性)
│   ├── reasoning.py                     # 推理奖励 (答案条件门控)
│   └── composite.py                     # 组合奖励
├── scripts/
│   ├── reward_fn.py                     # verl 集成入口 (PPO)
│   ├── multi_reward_manager.py          # 增强的 reward manager
│   ├── grpo_core.py                     # GRPO 核心算法 ⭐
│   ├── train_ppo.sh                     # PPO 通用训练脚本
│   ├── train_8gpu.sh                    # PPO 8卡A100训练脚本
│   ├── train_quick.sh                   # PPO 单卡/多卡快速启动
│   ├── train_grpo.py                    # GRPO 训练器 ⭐
│   ├── train_grpo.sh                    # GRPO 启动脚本 ⭐
│   ├── sft_warmup.py                    # SFT 预热基线 ⭐
│   ├── run_all_experiments.sh           # 运行所有实验
│   ├── run_ablation_experiments.sh      # 批量对照实验脚本
│   ├── evaluate.py                      # 单模型评估脚本
│   ├── evaluate_all_models.sh           # 批量并行评估脚本
│   └── analyze_metrics.py               # 指标分析工具
├── results/                              # 评估结果
├── outputs/                             # 训练输出日志
├── checkpoints/                         # 模型检查点
└── README.md
```

## 快速开始

### 1. 环境准备

```bash
cd ~/verl
pip install -e .
```

如果需要 WandB 日志：

```bash
wandb login
```

### 2. 数据准备

```bash
python examples/data_preprocess/gsm8k.py --local_save_dir $HOME/data/gsm8k/
```

### 3. (推荐) SFT 预热 — 建立合理的数学基线

v1 直接对 Qwen2.5-0.5B-Instruct (6% GSM8K 准确率) 做 RL，大部分提升来自学会输出格式。建议先做 SFT：

```bash
python scripts/sft_warmup.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --train-data $HOME/data/gsm8k/train.parquet \
    --val-data $HOME/data/gsm8k/test.parquet \
    --output ./checkpoints/sft_warmup \
    --epochs 3 \
    --lr 2e-5
```

SFT 完成后，用该模型作为 RL 训练的起点：

```bash
# PPO 使用 SFT 模型
./scripts/train_8gpu.sh --model ./checkpoints/sft_warmup/final

# GRPO 使用 SFT 模型
./scripts/train_grpo.sh --model ./checkpoints/sft_warmup/final
```

### 4. 运行训练

#### 方式一：GRPO 训练（推荐，省显存）⭐

```bash
cd ~/RL_verl

# 默认多奖励配置 (单GPU)
./scripts/train_grpo.sh --n-gpus 1

# 8-GPU 全速训练
./scripts/train_grpo.sh --n-gpus 8

# 自定义参数
./scripts/train_grpo.sh \
    --n-gpus 8 \
    --group-size 8 \
    --total-steps 1000 \
    --w-correctness 0.7 \
    --w-format 0.1 \
    --w-reasoning 0.2 \
    --model ./checkpoints/sft_warmup/final
```

#### 方式二：PPO 训练（verl 集成）

```bash
cd ~/RL_verl

# 8x A100 80GB 优化
./scripts/train_8gpu.sh

# 自定义奖励权重
./scripts/train_8gpu.sh --w-correctness 0.6 --w-format 0.1 --w-reasoning 0.3

# 单卡快速启动
./scripts/train_quick.sh

# 使用 verl 直接运行
export PYTHONPATH="${PWD}:${PYTHONPATH}"
python -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_gsm8k
```

#### 批量对照实验

```bash
# PPO 对照实验
./scripts/run_ablation_experiments.sh

# 查看所有预定义配置
./scripts/run_ablation_experiments.sh --help
```

## GRPO vs PPO 对比

| 参数 | PPO | GRPO |
|------|-----|------|
| 配置文件 | `config/ppo_gsm8k.yaml` | `config/grpo_gsm8k.yaml` |
| 启动脚本 | `scripts/train_8gpu.sh` | `scripts/train_grpo.sh` |
| Critic 模型 | 需要 | 不需要 |
| 优势计算 | GAE (γ, λ) | 组内标准化 |
| 组采样 | 不支持 | K 个响应/提示 |
| KL 控制 | 独立控制器 | 嵌入损失 (β 系数) |

**GRPO 关键参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--group-size` | 4 | 每个提示采样的响应数 K |
| `--total-steps` | 1000 | 总训练步数 |
| `--lr` | 1e-6 | 学习率 |

## 实验配置

| 实验 | 配置文件 | 奖励系统版本 | 说明 |
|------|---------|------------|------|
| Exp1 | `exp1_correctness_only.yaml` | v2 | 仅正确性 (flexible 提取) |
| Exp2 | `exp2_multi_reward_default.yaml` | v2 | 默认多奖励 (答案条件推理) |
| Exp3 | `exp3_heavy_reasoning.yaml` | v2 | 强调推理 |
| Exp4 | `exp4_balanced.yaml` | v2 | 平衡配置 |
| Exp5 | `exp5_xml_format.yaml` | v2 | XML 格式输出 |
| Exp6 | `exp6_legacy_comparison.yaml` | v1 (对照) | 旧版 strict + 非门控推理 |
| GRPO | `grpo_default.yaml` | v2 | GRPO 默认多奖励 |

**使用方式：**

```bash
# PPO 运行指定实验
./scripts/train_ppo.sh --config experiments/exp2_multi_reward_default

# GRPO 运行指定实验
python scripts/train_grpo.py --config config/experiments/grpo_default.yaml

# 运行 v1 对照实验（评估旧版奖励函数的影响）
./scripts/train_ppo.sh --config experiments/exp6_legacy_comparison
```

## 奖励函数详解

### V2 奖励函数 (推荐)

```python
# 默认多奖励组合 (v2)
# correctness_method="flexible" — 不依赖 #### 格式
# answer_conditioned_reasoning=True — 推理分受答案正确性门控
compute_score(...)  # w_correctness=0.7, w_format=0.1, w_reasoning=0.2

# 仅正确性 (v2 使用 flexible 提取)
compute_score_correctness_only(...)

# 平衡配置
compute_score_balanced(...)  # w_correctness=0.5, w_format=0.25, w_reasoning=0.25
```

### V1 对照函数 (仅用于对比实验)

```python
# 旧版奖励：strict 提取 + 非门控推理
compute_score_legacy_reasoning(...)
```

### 自定义奖励权重

**PPO (verl Hydra 覆盖)：**
```bash
python -m verl.trainer.main_ppo \
    custom_reward_function.path=$HOME/RL_verl/scripts/reward_fn.py \
    custom_reward_function.name=compute_score \
    '+custom_reward_function.reward_kwargs.w_correctness'=0.8 \
    '+custom_reward_function.reward_kwargs.w_format'=0.05 \
    '+custom_reward_function.reward_kwargs.w_reasoning'=0.15 \
    '+custom_reward_function.reward_kwargs.answer_conditioned_reasoning'=true
```

**GRPO (CLI 参数)：**
```bash
./scripts/train_grpo.sh \
    --w-correctness 0.8 \
    --w-format 0.1 \
    --w-reasoning 0.1
```

### 奖励组件说明

| 组件 | V1 行为 | V2 行为 |
|------|---------|---------|
| `correctness_reward` | strict 模式要求 `####` 格式 | flexible 模式，从文本中提取最后有效数字 |
| `format_reward` | 仅检查 `#### NUMBER` 模式 | 综合评估：步骤标记 + 计算过程 + 答案标记 |
| `reasoning_reward` | 启发式：长度分 + 步骤分（与正确性无关） | 答案条件门控：错误答案的结构分归零 |

## 监控与可视化

### WandB 集成

训练自动上传到 WandB，包含：
- reward 各组件分解
- KL divergence
- Entropy
- Advantage 分布 (GRPO: 组内标准化优势)
- 学习率曲线

### 本地日志

训练日志保存位置：

1. **Hydra 配置日志** (PPO)：`outputs/YYYY-MM-DD/HH-MM-SS/`
2. **训练终端日志** (PPO)：`outputs/logs/`
3. **GRPO 训练历史**：`checkpoints/grpo/gsm8k/training_history.json`

### 指标分析工具 (analyze_metrics.py)

```bash
# 从 metrics 文件生成可视化报告
python scripts/analyze_metrics.py report \
    --experiment-dir outputs/metrics \
    --output-dir reports/

# 对比多个实验
python scripts/analyze_metrics.py compare \
    --files exp1_metrics.jsonl exp2_metrics.jsonl \
    --metric score \
    --output comparison.png
```

## 模型评估

### 预训练模型

我们在 Hugging Face 上发布了使用不同奖励权重训练的模型：

| 模型 | 权重 (C/F/R) | 描述 |
|------|-------------|------|
| [leixinlin/qwen2.5-0.5b-gsm8k-rl-w1.0_0.0_0.0](https://huggingface.co/leixinlin/qwen2.5-0.5b-gsm8k-rl-w1.0_0.0_0.0) | 1.0/0.0/0.0 | 基线：仅正确性奖励 |
| [leixinlin/qwen2.5-0.5b-gsm8k-rl-w0.7_0.1_0.2](https://huggingface.co/leixinlin/qwen2.5-0.5b-gsm8k-rl-w0.7_0.1_0.2) | 0.7/0.1/0.2 | 默认多奖励配置 |
| [leixinlin/qwen2.5-0.5b-gsm8k-rl-w0.8_0.1_0.1](https://huggingface.co/leixinlin/qwen2.5-0.5b-gsm8k-rl-w0.8_0.1_0.1) | 0.8/0.1/0.1 | 高正确性权重 |
| [leixinlin/qwen2.5-0.5b-gsm8k-rl-w0.5_0.1_0.4](https://huggingface.co/leixinlin/qwen2.5-0.5b-gsm8k-rl-w0.5_0.1_0.4) | 0.5/0.1/0.4 | 高推理权重 |
| [leixinlin/qwen2.5-0.5b-gsm8k-rl-w0.6_0.2_0.2](https://huggingface.co/leixinlin/qwen2.5-0.5b-gsm8k-rl-w0.6_0.2_0.2) | 0.6/0.2/0.2 | 平衡配置 |
| [leixinlin/qwen2.5-0.5b-gsm8k-rl-w0.5_0.3_0.2](https://huggingface.co/leixinlin/qwen2.5-0.5b-gsm8k-rl-w0.5_0.3_0.2) | 0.5/0.3/0.2 | 强调格式 |

> **注意**：以上模型使用 v1 奖励系统训练。使用 v2 奖励函数评估时，`reasoning_score` 可能因答案条件门控机制而不同。

### Step 1: 合并 FSDP Checkpoint (仅 PPO)

```bash
cd ~/verl
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /path/to/checkpoints/global_step_XX/actor \
    --target_dir /path/to/merged_model \
    --trust-remote-code
```

> GRPO 训练器直接保存 HuggingFace 格式 checkpoint，无需合并。

### Step 2: 评估模型

```bash
cd ~/RL_verl

# 评估合并后的 PPO 模型
python scripts/evaluate.py \
    --model-path checkpoints/merged_model \
    --test-data $HOME/data/gsm8k/test.parquet \
    --output results/evaluation_results.json \
    --batch-size 16

# 评估 GRPO 模型
python scripts/evaluate.py \
    --model-path checkpoints/grpo/gsm8k/final \
    --test-data $HOME/data/gsm8k/test.parquet \
    --output results/grpo_eval.json \
    --w-correctness 0.7 \
    --w-format 0.1 \
    --w-reasoning 0.2
```

**评估参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch-size` | 16 | 批量推理大小 |
| `--temperature` | 0.7 | 采样温度（0 = 贪婪解码） |
| `--num-samples` | -1 | 评估样本数（-1 = 全部） |
| `--w-correctness` | 0.7 | 正确性权重 |
| `--w-format` | 0.1 | 格式权重 |
| `--w-reasoning` | 0.2 | 推理权重 |

### 评估输出说明

| 指标 | 说明 |
|------|------|
| `accuracy` | 准确率（答案完全正确的比例） |
| `correctness_score` | 正确性得分 |
| `format_score` | 格式得分（v2: 步骤+计算+答案标记综合评估） |
| `reasoning_score` | 推理结构得分（v2: 答案条件门控，错误答案=0） |
| `score` | 综合奖励分数 |

### 实验结果

以下是在 GSM8K 测试集（1319 样本）上使用 **v1 奖励函数**评估的结果：

| 模型 | 准确率 | 综合得分 | 正确性 | 格式 | 推理 |
|------|--------|----------|--------|------|------|
| Qwen2.5-0.5B-Instruct (基线) | 0.0599 | 0.2438 | 0.0599 | 0.2699 | 0.8742 |
| w1.0_0.0_0.0 (仅正确性) | 0.5201 | 0.5201 | 0.5201 | 0.9780 | 0.9153 |
| **w0.7_0.1_0.2** (默认配置) | 0.5292 | 0.6574 | 0.5292 | 0.9742 | 0.9476 |
| w0.8_0.1_0.1 (高正确性) | 0.4898 | 0.5826 | 0.4898 | 0.9780 | 0.9301 |
| w0.5_0.1_0.4 (高推理) | 0.5087 | 0.7403 | 0.5087 | 0.9727 | 0.9718 |
| w0.6_0.2_0.2 (平衡) | 0.5042 | 0.6905 | 0.5042 | 0.9856 | 0.9544 |
| w0.5_0.3_0.2 (强调格式) | 0.5110 | 0.7428 | 0.5110 | 0.9848 | 0.9594 |

**关键发现与注意事项：**

1. **RL 训练显著提升准确率**：~6% → ~50%
2. **基线推理分虚高**：Qwen2.5-0.5B 在 6% 准确率下得到 0.8742 推理分——说明 v1 推理奖励衡量的是输出结构而非数学推理质量
3. **V2 改进**：答案条件门控推理奖励可解决上述问题，错误答案的推理分归零
4. **建议 SFT 预热**：大部分格式提升来自 RL 训练，建议先用 SFT 建立基线再用 RL 提升推理质量

> **结果文件位置**：`results/` 目录下。v1 旧版结果带有原有权重文件名。

### 批量并行评估

```bash
cd ~/RL_verl

# 预览将要执行的命令
./scripts/evaluate_all_models.sh --dry-run

# 运行完整评估
./scripts/evaluate_all_models.sh

# 快速测试
./scripts/evaluate_all_models.sh --num-samples 100
```

## 保存与分享模型

### 方法 1：上传到 Hugging Face Hub（推荐）

```bash
huggingface-cli login

python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = '/path/to/your/model'
repo_name = '你的用户名/qwen2.5-0.5b-gsm8k-rl'

model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model.push_to_hub(repo_name, private=True)
tokenizer.push_to_hub(repo_name, private=True)
"
```

### 方法 2：打包压缩保存

```bash
cd checkpoints
tar -czvf model.tar.gz merged_model/
```

### 方法 3：保存为 safetensors 格式

```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model.save_pretrained(save_path, safe_serialization=True)
tokenizer.save_pretrained(save_path)
"
```

## 性能参考

基于 8x A100 80GB 的典型训练时间：

| 模型 | 算法 | Batch Size | 训练时间 |
|------|------|-----------|---------|
| Qwen2.5-0.5B | PPO | 1024 | ~2.5 hours (15 epochs) |
| Qwen2.5-0.5B | GRPO | 256 prompts × 4 | ~2 hours (1000 steps) |
| Qwen2.5-1.5B | PPO | 512 | ~5 hours |
| Qwen2.5-7B | PPO | 256 | ~11 hours |
| Qwen2.5-7B | GRPO | 128 prompts × 4 | ~7 hours |

> GRPO 在 7B+ 模型上 VRAM 节省约 50%，可用更大 batch 或更少 GPU。

## 故障排除

### 显存不足 (OOM)

```bash
# PPO: 减小 batch size
./scripts/train_8gpu.sh --batch-size 512

# GRPO: 减小 group size 或 batch
./scripts/train_grpo.sh --group-size 2 --n-gpus 4

# 或直接调整 micro batch
python -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    critic.ppo_micro_batch_size_per_gpu=4
```

### 奖励函数版本选择

```bash
# V2 默认 (推荐): flexible 提取 + 答案条件推理
python scripts/evaluate.py --w-correctness 0.7 --w-format 0.1 --w-reasoning 0.2

# V1 对照: strict 提取 + 非门控推理 (仅用于与旧结果对比)
# 需要修改 reward_fn 调用为 compute_score_legacy_reasoning
```

### Hydra 配置错误 (PPO)

```bash
# 新增字段使用 + 前缀
'+custom_reward_function.reward_kwargs.answer_conditioned_reasoning'=true

# 修改已有字段不需要 +
trainer.n_gpus_per_node=8
```

### 数据路径问题

```bash
ls -la $HOME/data/gsm8k/
# 应该有: train.parquet, test.parquet
```

## License

MIT License
