#!/bin/bash
# Quick Start Training Script
#
# This script provides a simple way to run training with the multi-reward system
# using command-line arguments similar to the original demo.
#
# Usage: ./train_quick.sh [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_BATCH_SIZE=512
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=512
LR_ACTOR=1e-6
LR_CRITIC=1e-5
KL_COEF=0.001
N_GPUS=4
TOTAL_EPOCHS=15
SAVE_FREQ=5
TEST_FREQ=5

# Reward weights
W_CORRECTNESS=0.7
W_FORMAT=0.1
W_REASONING=0.2

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --batch-size)
            TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --gpus)
            N_GPUS="$2"
            shift 2
            ;;
        --epochs)
            TOTAL_EPOCHS="$2"
            shift 2
            ;;
        --w-correctness)
            W_CORRECTNESS="$2"
            shift 2
            ;;
        --w-format)
            W_FORMAT="$2"
            shift 2
            ;;
        --w-reasoning)
            W_REASONING="$2"
            shift 2
            ;;
        --correctness-only)
            W_CORRECTNESS=1.0
            W_FORMAT=0.0
            W_REASONING=0.0
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model PATH           Model path (default: Qwen/Qwen2.5-0.5B-Instruct)"
            echo "  --batch-size SIZE      Training batch size (default: 512)"
            echo "  --gpus NUM             Number of GPUs (default: 4)"
            echo "  --epochs NUM           Number of epochs (default: 15)"
            echo "  --w-correctness WEIGHT Weight for correctness reward (default: 0.7)"
            echo "  --w-format WEIGHT      Weight for format reward (default: 0.1)"
            echo "  --w-reasoning WEIGHT   Weight for reasoning reward (default: 0.2)"
            echo "  --correctness-only     Use only correctness reward (baseline)"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check data
GSM8K_TRAIN="$HOME/data/gsm8k/train.parquet"
GSM8K_TEST="$HOME/data/gsm8k/test.parquet"

if [[ ! -f "$GSM8K_TRAIN" ]] || [[ ! -f "$GSM8K_TEST" ]]; then
    echo "Error: GSM8K data not found. Please run data preprocessing first."
    exit 1
fi

# Export paths
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Create experiment name based on weights
EXP_NAME="gsm8k_c${W_CORRECTNESS}_f${W_FORMAT}_r${W_REASONING}"

echo "=============================================="
echo "Quick Start PPO Training"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Batch size: $TRAIN_BATCH_SIZE"
echo "GPUs: $N_GPUS"
echo "Epochs: $TOTAL_EPOCHS"
echo "Reward weights:"
echo "  - Correctness: $W_CORRECTNESS"
echo "  - Format: $W_FORMAT"
echo "  - Reasoning: $W_REASONING"
echo "=============================================="

# Run training with inline config overrides
# Note: Use '+' prefix to add new fields that don't exist in the default config
python3 -m verl.trainer.main_ppo \
    data.train_files="$GSM8K_TRAIN" \
    data.val_files="$GSM8K_TEST" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    custom_reward_function.path="$PROJECT_DIR/scripts/reward_fn.py" \
    custom_reward_function.name=compute_score \
    +custom_reward_function.reward_kwargs.w_correctness=$W_CORRECTNESS \
    +custom_reward_function.reward_kwargs.w_format=$W_FORMAT \
    +custom_reward_function.reward_kwargs.w_reasoning=$W_REASONING \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=$LR_ACTOR \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    critic.optim.lr=$LR_CRITIC \
    critic.model.path="$MODEL_PATH" \
    critic.ppo_micro_batch_size_per_gpu=8 \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    trainer.logger='["console"]' \
    trainer.project_name=verl_multi_reward \
    trainer.experiment_name="$EXP_NAME" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    2>&1 | tee "logs/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Training completed!"
