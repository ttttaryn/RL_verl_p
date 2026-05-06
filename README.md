# verl Multi-Reward PPO Training Project

这个项目在 verl 官方框架的基础上，实现了多奖励组合系统和推理结构奖励，用于 GSM8K 数学问题的 PPO 训练。

## 项目特点

### ✅ 改进 1：多奖励组合系统 (Multi-Reward)

从原始的单一正确性奖励扩展为多维度奖励组合：

```python
reward = (
    w_correctness * correctness_reward +  # 答案正确性
    w_format * format_reward +             # 输出格式
    w_reasoning * reasoning_reward         # 推理质量
)
```

### ✅ 改进 2：推理结构奖励 (Reasoning-aware PPO)

鼓励模型产生结构化的推理输出：

| Reward 组件 | 作用 |
|------------|------|
| `reasoning_length_reward` | 鼓励适当长度的推理 |
| `step_count_reward` | 控制推理步数 |
| `xml_format_reward` | 支持 XML 结构化输出 |

### ✅ 改进 3：可视化与实验对比

- 详细的 reward 分解日志
- KL / Entropy / Advantage 追踪
- 多实验对比分析工具

## 硬件配置

本项目针对以下配置进行了优化：
- **推荐配置**: 8x NVIDIA A100 80GB
- **最低配置**: 1x GPU with 24GB+ VRAM

## 示例安装

```bash
conda create -n verl python=3.10 -y
conda activate verl
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121
pip install vllm
# 安装高性能算子
pip install flash-attn --no-build-isolation
```

```bash
# 克隆 verl 仓库
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .
pip install -r requirements.txt
```

```bash
# 克隆本项目仓库
git clone https://github.com/irroca/RL_verl.git
cd RL_verl
```


## 项目结构

```
RL_verl/
├── config/
│   ├── ppo_gsm8k.yaml              # 主配置文件 (8-GPU优化)
│   └── experiments/                 # 实验配置
│       ├── exp1_correctness_only.yaml
│       ├── exp2_multi_reward_default.yaml
│       ├── exp3_heavy_reasoning.yaml
│       ├── exp4_balanced.yaml
│       └── exp5_xml_format.yaml
├── verl_rewards/                    # 多奖励系统
│   ├── __init__.py
│   ├── correctness.py              # 正确性奖励
│   ├── format.py                   # 格式奖励
│   ├── reasoning.py                # 推理奖励
│   └── composite.py                # 组合奖励
├── scripts/
│   ├── reward_fn.py                # verl 集成入口
│   ├── train_8gpu.sh               # 8卡A100训练脚本 ⭐
│   ├── train_quick.sh              # 单卡/多卡快速启动
│   ├── train_ppo.sh                # 通用训练脚本
│   ├── run_all_experiments.sh      # 运行所有实验
│   ├── run_ablation_experiments.sh # 批量对照实验脚本 ⭐
│   ├── evaluate.py                 # 单模型评估脚本（支持批量推理）
│   ├── evaluate_all_models.sh      # 批量并行评估脚本 ⭐
│   ├── analyze_metrics.py          # 指标分析工具
│   └── multi_reward_manager.py     # 增强的 reward manager
├── results/                         # 评估结果 ⭐
│   ├── baseline_results.json       # 基线模型评估结果
│   ├── eval_w1.0_0.0_0.0.json      # 仅正确性奖励模型
│   ├── eval_w0.7_0.1_0.2.json      # 默认多奖励配置
│   ├── eval_w0.8_0.1_0.1.json      # 高正确性权重
│   ├── eval_w0.5_0.1_0.4.json      # 高推理权重
│   ├── eval_w0.6_0.2_0.2.json      # 平衡配置
│   └── eval_w0.5_0.3_0.2.json      # 强调格式
├── outputs/                        # 训练输出日志
├── checkpoints/                    # 模型检查点
└── README.md
```

## 快速开始

### 1. 环境准备

确保已安装 verl 及其依赖（参考前面的安装步骤）：

```bash
cd ~/verl
pip install -e .
```

如果需要将训练过程上传到 WandB，请登录wandb：

```bash
wandb login
```


### 2. 数据准备

如果还没有下载 GSM8K 数据集：

```bash
python examples/data_preprocess/gsm8k.py --local_save_dir $HOME/data/gsm8k/
```

### 3. 运行训练

#### 方式一：8x A100 80GB 优化训练（推荐）⭐

```bash
cd ~/RL_verl

# 使用默认多奖励配置（8卡全速）
./scripts/train_8gpu.sh

# 自定义奖励权重
./scripts/train_8gpu.sh --w-correctness 0.6 --w-format 0.1 --w-reasoning 0.3

# 更大batch size（充分利用80GB显存）
./scripts/train_8gpu.sh --batch-size 2048

# 调整学习率
./scripts/train_8gpu.sh --lr 5e-7 --epochs 20
```

##### 批量运行对照实验 (Ablation Study) ⭐

使用 `run_ablation_experiments.sh` 可以自动运行多组奖励权重对照实验：

```bash
cd ~/RL_verl

# 查看所有预定义实验配置
./scripts/run_ablation_experiments.sh --help

# 运行所有预定义对照实验
./scripts/run_ablation_experiments.sh

# 只打印配置，不实际运行（dry run）
./scripts/run_ablation_experiments.sh --dry-run

# 只运行指定编号的实验（如实验 1、3、4）
./scripts/run_ablation_experiments.sh --exp 1 3 4

# 运行对照实验并指定额外参数
./scripts/run_ablation_experiments.sh --epochs 10 --batch-size 512
```

**预定义对照实验配置：**

| No. | 实验名称 | 权重 (C/F/R) | 描述 |
|-----|---------|-------------|------|
| 1 | `baseline_correctness` | 1.0/0.0/0.0 | 基线：仅正确性奖励 |
| 2 | `default_multi` | 0.7/0.1/0.2 | 默认多奖励配置 |
| 3 | `high_correctness` | 0.8/0.1/0.1 | 高正确性权重 |
| 4 | `high_reasoning` | 0.5/0.1/0.4 | 高推理权重 |
| 5 | `balanced` | 0.6/0.2/0.2 | 平衡配置 |
| 6 | `format_emphasis` | 0.5/0.3/0.2 | 强调格式 |

> **提示**：实验结果汇总日志保存在 `outputs/logs/ablation_summary_*.log`

**8卡优化参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `train_batch_size` | 1024 | 全局batch size（8卡 × 128/卡）|
| `ppo_micro_batch_size_per_gpu` | 16 | 每GPU micro batch |
| `log_prob_micro_batch_size_per_gpu` | 32 | rollout log prob batch |
| `gpu_memory_utilization` | 0.6 | vLLM 显存利用率 |

#### 方式二：快速启动脚本（单卡/少量GPU）

```bash
cd ~/RL_verl

# 默认多奖励配置
./scripts/train_quick.sh

# 只使用正确性奖励（基线）
./scripts/train_quick.sh --correctness-only

# 自定义奖励权重
./scripts/train_quick.sh --w-correctness 0.6 --w-format 0.1 --w-reasoning 0.3

# 指定GPU数量
./scripts/train_quick.sh --n-gpus 4
```

#### 方式三：使用 verl 直接运行

```bash
cd ~/RL_verl
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# 使用配置文件
python -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_gsm8k

# 命令行覆盖参数
python -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_gsm8k \
    trainer.n_gpus_per_node=8 \
    data.train_batch_size=1024 \
    '+custom_reward_function.reward_kwargs.w_correctness'=0.8
```

## 实验配置

项目在 `config/experiments/` 目录下包含 5 个预定义实验配置：

| 实验 | 配置文件 | 说明 |
|------|---------|------|
| Exp1 | `exp1_correctness_only.yaml` | 只使用正确性奖励（基线）|
| Exp2 | `exp2_multi_reward_default.yaml` | 默认多奖励 (0.7/0.1/0.2) |
| Exp3 | `exp3_heavy_reasoning.yaml` | 强调推理 (0.5/0.1/0.4) |
| Exp4 | `exp4_balanced.yaml` | 平衡配置 (0.6/0.15/0.25) |
| Exp5 | `exp5_xml_format.yaml` | XML格式输出 (0.5/0.3/0.2) |

**使用方式**：通过 `train_ppo.sh` 脚本运行预定义实验：

```bash
# 运行单个实验（使用 --exp 参数）
./scripts/train_ppo.sh --exp 1    # 运行 Exp1: correctness only
./scripts/train_ppo.sh --exp 3    # 运行 Exp3: heavy reasoning

# 或指定配置文件名
./scripts/train_ppo.sh --config experiments/exp3_heavy_reasoning
```

运行所有实验：

```bash
./scripts/run_all_experiments.sh
```

> **注意**：`train_8gpu.sh` 脚本**不使用**这些预定义配置文件，而是直接通过命令行参数设置奖励权重。如需使用预定义实验配置，请使用 `train_ppo.sh`。

## 奖励函数详解

### 可用的奖励函数

```python
# 1. 默认多奖励组合
compute_score(...)  # w_correctness=0.7, w_format=0.1, w_reasoning=0.2

# 2. 只使用正确性
compute_score_correctness_only(...)  

# 3. 带推理权重
compute_score_with_reasoning(...)  # w_correctness=0.5, w_format=0.1, w_reasoning=0.4

# 4. 平衡配置
compute_score_balanced(...)  # w_correctness=0.6, w_format=0.15, w_reasoning=0.25
```

### 自定义奖励权重

通过命令行传递 `reward_kwargs`：

```bash
python -m verl.trainer.main_ppo \
    custom_reward_function.path=$HOME/RL_verl/scripts/reward_fn.py \
    custom_reward_function.name=compute_score \
    '+custom_reward_function.reward_kwargs.w_correctness'=0.8 \
    '+custom_reward_function.reward_kwargs.w_format'=0.05 \
    '+custom_reward_function.reward_kwargs.w_reasoning'=0.15
```

## 监控与可视化

### WandB 集成

训练自动上传到 WandB，包含：
- reward 各组件分解
- KL divergence
- Entropy
- Advantage 分布
- 学习率曲线

### 本地日志

训练日志保存位置：

1. **Hydra 配置日志**：`outputs/YYYY-MM-DD/HH-MM-SS/`
   - `.hydra/config.yaml` - 完整配置
   - `.hydra/overrides.yaml` - 命令行覆盖的参数
   - `main_ppo.log` - Hydra 主进程日志（通常为空，因为训练在 Ray worker 中运行）

2. **训练终端日志**（使用 `train_8gpu.sh` 时）：`outputs/logs/`
   ```bash
   ls outputs/logs/
   cat outputs/logs/gsm8k_8gpu_w0.7_0.1_0.2_YYYYMMDD_HHMMSS.log
   ```

### 指标分析工具 (analyze_metrics.py)

`scripts/analyze_metrics.py` 提供了训练指标的可视化和对比分析功能，适用于离线分析自定义记录的指标。

#### 数据格式要求

该工具需要 JSONL 格式的 metrics 文件（`*_metrics.jsonl`），每行包含一个 JSON 对象：

```json
{"timestamp": "2025-12-13T10:30:00", "step": 100, "score": 0.65, "correctness_score": 0.52, "format_score": 0.97}
```

#### 记录自定义指标

可以使用内置的 `MetricsLogger` 类记录指标到 JSONL 文件：

```python
from scripts.analyze_metrics import MetricsLogger

logger = MetricsLogger(log_dir="outputs/metrics", experiment_name="exp1")
logger.log(step=100, metrics={"score": 0.65, "correctness_score": 0.52})
logger.save_summary()
```

#### 生成实验报告

```bash
# 从 metrics 文件生成可视化报告
python scripts/analyze_metrics.py report \
    --experiment-dir outputs/metrics \
    --output-dir reports/
```

#### 对比多个实验的指标

```bash
python scripts/analyze_metrics.py compare \
    --files exp1_metrics.jsonl exp2_metrics.jsonl \
    --metric score \
    --output comparison.png \
    --smoothing 0.9
```

> **注意**：verl 默认使用 WandB 记录训练指标，该工具主要用于离线分析自定义记录的数据。

## 模型评估

### 预训练模型 ⭐

我们在 Hugging Face 上发布了使用不同奖励权重训练的模型，可以直接下载使用：

| 模型 | 权重 (C/F/R) | 描述 |
|------|-------------|------|
| [leixinlin/qwen2.5-0.5b-gsm8k-rl-w1.0_0.0_0.0](https://huggingface.co/leixinlin/qwen2.5-0.5b-gsm8k-rl-w1.0_0.0_0.0) | 1.0/0.0/0.0 | 基线：仅正确性奖励 |
| [leixinlin/qwen2.5-0.5b-gsm8k-rl-w0.7_0.1_0.2](https://huggingface.co/leixinlin/qwen2.5-0.5b-gsm8k-rl-w0.7_0.1_0.2) | 0.7/0.1/0.2 | 默认多奖励配置 |
| [leixinlin/qwen2.5-0.5b-gsm8k-rl-w0.8_0.1_0.1](https://huggingface.co/leixinlin/qwen2.5-0.5b-gsm8k-rl-w0.8_0.1_0.1) | 0.8/0.1/0.1 | 高正确性权重 |
| [leixinlin/qwen2.5-0.5b-gsm8k-rl-w0.5_0.1_0.4](https://huggingface.co/leixinlin/qwen2.5-0.5b-gsm8k-rl-w0.5_0.1_0.4) | 0.5/0.1/0.4 | 高推理权重 |
| [leixinlin/qwen2.5-0.5b-gsm8k-rl-w0.6_0.2_0.2](https://huggingface.co/leixinlin/qwen2.5-0.5b-gsm8k-rl-w0.6_0.2_0.2) | 0.6/0.2/0.2 | 平衡配置 |
| [leixinlin/qwen2.5-0.5b-gsm8k-rl-w0.5_0.3_0.2](https://huggingface.co/leixinlin/qwen2.5-0.5b-gsm8k-rl-w0.5_0.3_0.2) | 0.5/0.3/0.2 | 强调格式 |

### Step 1: 合并 FSDP Checkpoint

verl 训练保存的是 FSDP 分片 checkpoint，需要先合并为 HuggingFace 格式：

```bash
cd ~/verl

# 合并 actor 模型
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /path/to/checkpoints/global_step_XX/actor \
    --target_dir /path/to/merged_model \
    --trust-remote-code

# 示例：合并训练好的模型
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir ~/RL_verl/checkpoints/verl_multi_reward_8gpu/gsm8k_8gpu_w0.7_0.1_0.2_20251213_035342/global_step_45/actor \
    --target_dir ~/RL_verl/checkpoints/merged_model \
    --trust-remote-code
```

### Step 2: 评估模型

```bash
cd ~/RL_verl

# 评估合并后的模型
python scripts/evaluate.py \
    --model-path checkpoints/merged_model \
    --test-data $HOME/data/gsm8k/test.parquet \
    --output results/evaluation_results.json

# 完整参数（支持批量推理加速）
python scripts/evaluate.py \
    --model-path checkpoints/merged_model \
    --test-data $HOME/data/gsm8k/test.parquet \
    --output results/evaluation_results.json \
    --num-samples 500 \
    --batch-size 16 \
    --max-tokens 512 \
    --temperature 0.0 \
    --w-correctness 0.7 \
    --w-format 0.1 \
    --w-reasoning 0.2
```

**批量推理参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch-size` | 16 | 批量推理大小，提高 GPU 利用率 |
| `--temperature` | 0.7 | 采样温度（0 为贪婪解码，结果可复现）|
| `--num-samples` | -1 | 评估样本数（-1 为全部 1319 个）|

> **性能提升**：使用 `batch-size=16` 相比单样本推理可获得约 **9 倍加速**（1319 样本从 3 小时缩短到约 20 分钟）

### Step 3: 对比训练前后效果

```bash
# 评估原始模型（训练前基线）
python scripts/evaluate.py \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --test-data $HOME/data/gsm8k/test.parquet \
    --output results/baseline_results.json

# 评估训练后模型
python scripts/evaluate.py \
    --model-path checkpoints/merged_model \
    --test-data $HOME/data/gsm8k/test.parquet \
    --output results/trained_results.json
```

### 评估输出说明

评估脚本输出以下指标：

| 指标 | 说明 |
|------|------|
| `accuracy` | 准确率（答案完全正确的比例）|
| `correctness_score` | 正确性得分 |
| `format_score` | 格式得分（是否使用 #### 格式）|
| `reasoning_score` | 推理质量得分 |
| `score` | 综合奖励分数 |

### 实验结果 ⭐

以下是在 GSM8K 测试集（1319 样本）上的评估结果：

| 模型 | 准确率 | 综合得分 | 正确性 | 格式 | 推理 |
|------|--------|----------|--------|------|------|
| Qwen2.5-0.5B-Instruct (基线) | 0.0599 | 0.2438 | 0.0599 | 0.2699 | 0.8742 |
| **w1.0_0.0_0.0** (仅正确性) | 0.5201 | 0.5201 | 0.5201 | 0.9780 | 0.9153 |
| **w0.7_0.1_0.2** (默认配置) | 0.5292 | 0.6574 | 0.5292 | 0.9742 | 0.9476 |
| **w0.8_0.1_0.1** (高正确性) | 0.4898 | 0.5826 | 0.4898 | 0.9780 | 0.9301 |
| **w0.5_0.1_0.4** (高推理) | 0.5087 | 0.7403 | 0.5087 | 0.9727 | 0.9718 |
| **w0.6_0.2_0.2** (平衡) | 0.5042 | 0.6905 | 0.5042 | 0.9856 | 0.9544 |
| **w0.5_0.3_0.2** (强调格式) | 0.5110 | 0.7428 | 0.5110 | 0.9848 | 0.9594 |

**关键发现：**

1. **RL 训练显著提升性能**：所有 RL 训练模型的准确率（~50%）相比基线（6%）提升了约 **8 倍**
2. **最高准确率**：`w0.7_0.1_0.2`（默认配置）达到 **52.92%** 准确率
3. **最高综合得分**：`w0.5_0.3_0.2`（强调格式）达到 **0.7428** 综合得分
4. **格式奖励效果明显**：所有 RL 模型的格式得分都超过 97%，远高于基线的 27%
5. **推理质量稳定**：所有模型的推理得分都在 91-97% 之间

> **结果文件位置**：详细评估结果保存在 `results/` 目录下

### Step 4: 批量并行评估多个模型 ⭐

使用 `evaluate_all_models.sh` 可以并行评估多个 Hugging Face 上的预训练模型，每个模型使用单独的 GPU：

```bash
cd ~/RL_verl

# 查看帮助和所有预定义模型
./scripts/evaluate_all_models.sh --help

# 预览将要执行的命令（不实际运行）
./scripts/evaluate_all_models.sh --dry-run

# 运行完整评估（所有 GSM8K 测试样本，使用批量推理加速）
./scripts/evaluate_all_models.sh

# 快速测试（只评估 100 个样本）
./scripts/evaluate_all_models.sh --num-samples 100

# 自定义批量大小（更大的 batch 更快，但需要更多显存）
./scripts/evaluate_all_models.sh --batch-size 32

# 使用贪婪解码确保可复现性（默认）
./scripts/evaluate_all_models.sh --temperature 0.0
```

**批量评估结果输出：**

| 输出位置 | 说明 |
|----------|------|
| `results/eval_w{权重}.json` | 每个模型的详细评估结果 |
| `results/evaluation_summary_*.json` | 所有模型的汇总报告 |
| `results/logs/eval_w{权重}.log` | 每个评估任务的运行日志 |

**汇总报告示例：**

```json
{
  "evaluation_time": "2025-12-14T10:30:00",
  "duration_seconds": 3600,
  "results": [
    {
      "model": "leixinlin/qwen2.5-0.5b-gsm8k-rl-w0.7_0.1_0.2",
      "weights": {"correctness": 0.7, "format": 0.1, "reasoning": 0.2},
      "accuracy": 0.65,
      "average_score": 0.72
    }
  ]
}
```

## 保存与分享模型

### 修复 Tokenizer Regex 警告

如果加载模型时出现以下警告：
```
The tokenizer you are loading with an incorrect regex pattern... 
You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
```

**解决方案**：先修复并保存 tokenizer，再上传模型：

```bash
python -c "
from transformers import AutoTokenizer

model_path = '/home/aiscuser/RL_verl/checkpoints/merged_model'

# 加载并修复 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    fix_mistral_regex=True  # 修复 regex 问题
)

# 保存修复后的 tokenizer（覆盖原来的）
tokenizer.save_pretrained(model_path)

print('Tokenizer 已修复并保存')
"
```

### 方法 1：上传到 Hugging Face Hub（推荐）

```bash
# 1. 先登录 Hugging Face
huggingface-cli login

# 2. 上传模型
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = '/home/aiscuser/RL_verl/checkpoints/merged_model'
repo_name = '你的用户名/qwen2.5-0.5b-gsm8k-rl'  # 替换为你的仓库名

model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 上传到 Hub（private=False 设为公开）
model.push_to_hub(repo_name, private=True)
tokenizer.push_to_hub(repo_name, private=True)

print(f'模型已上传到: https://huggingface.co/{repo_name}')
"
```

### 方法 2：打包压缩保存

```bash
# 压缩模型目录
cd /home/aiscuser/RL_verl/checkpoints
tar -czvf merged_model.tar.gz merged_model/

# 查看大小
ls -lh merged_model.tar.gz

# 上传到云存储（可选）
# Azure: azcopy copy merged_model.tar.gz "https://your-storage.blob.core.windows.net/models/"
# AWS:   aws s3 cp merged_model.tar.gz s3://your-bucket/models/
# GCS:   gsutil cp merged_model.tar.gz gs://your-bucket/models/
```

### 方法 3：保存为标准 HuggingFace 格式

```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = '/home/aiscuser/RL_verl/checkpoints/merged_model'
save_path = '/home/aiscuser/RL_verl/checkpoints/merged_model_hf'

model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 保存为 safetensors 格式
model.save_pretrained(save_path, safe_serialization=True)
tokenizer.save_pretrained(save_path)

print(f'模型已保存到: {save_path}')
"
```

## 性能参考

基于 8x A100 80GB 的典型训练时间：

| 模型 | Batch Size | 每Epoch时间 | 15 Epoch总时间 |
|------|-----------|-------------|---------------|
| Qwen2.5-0.5B | 1024 | ~10 min | ~2.5 hours |
| Qwen2.5-1.5B | 512 | ~20 min | ~5 hours |
| Qwen2.5-7B | 256 | ~45 min | ~11 hours |

## 故障排除

### 显存不足 (OOM)

```bash
# 减小batch size
./scripts/train_8gpu.sh --batch-size 512

# 或直接调整micro batch
python -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    critic.ppo_micro_batch_size_per_gpu=4
```

### Hydra 配置错误

```bash
# 新增字段使用 + 前缀
'+custom_reward_function.reward_kwargs.w_correctness'=0.8

# 修改已有字段不需要 +
trainer.n_gpus_per_node=8
```

### 数据路径问题

确保数据文件存在：

```bash
ls -la $HOME/data/gsm8k/
# 应该有: train.parquet, test.parquet
```

## License

MIT License
