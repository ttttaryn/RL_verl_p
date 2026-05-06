#!/bin/bash
# ==============================================================================
# 8x A100 80GB Optimized PPO Training Script for GSM8K Multi-Reward
# ==============================================================================
#
# This script is optimized for 8x A100 80GB GPUs with larger batch sizes
# and higher memory utilization.
#
# Usage: 
#   ./scripts/train_8gpu.sh                     # Use default multi-reward settings
#   ./scripts/train_8gpu.sh --model <path>      # Use different model
#   ./scripts/train_8gpu.sh --lr 5e-7           # Different learning rate
#
# Environment variables:
#   WANDB_PROJECT - Override wandb project name
#   WANDB_ENTITY  - Set wandb entity/team
#
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Add project to PYTHONPATH
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# 8x A100 80GB optimized defaults
MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_BATCH_SIZE=1024         # 8 GPUs x 128 per GPU
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=512
LR_ACTOR=1e-6
LR_CRITIC=1e-5
KL_COEF=0.001
N_GPUS=8
TOTAL_EPOCHS=15
SAVE_FREQ=5
TEST_FREQ=5

# Reward weights
W_CORRECTNESS=0.7
W_FORMAT=0.1
W_REASONING=0.2

# Parse command line arguments
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
        --lr)
            LR_ACTOR="$2"
            shift 2
            ;;
        --kl-coef)
            KL_COEF="$2"
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
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "8x A100 80GB PPO Training Configuration"
echo "=============================================="
echo "Model:            $MODEL_PATH"
echo "Train Batch Size: $TRAIN_BATCH_SIZE"
echo "Max Prompt Len:   $MAX_PROMPT_LENGTH"
echo "Max Response Len: $MAX_RESPONSE_LENGTH"
echo "Actor LR:         $LR_ACTOR"
echo "Critic LR:        $LR_CRITIC"
echo "KL Coefficient:   $KL_COEF"
echo "GPUs:             $N_GPUS"
echo "Epochs:           $TOTAL_EPOCHS"
echo "----------------------------------------------"
echo "Reward Weights:"
echo "  Correctness:    $W_CORRECTNESS"
echo "  Format:         $W_FORMAT"
echo "  Reasoning:      $W_REASONING"
echo "=============================================="

# Set experiment name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="gsm8k_8gpu_w${W_CORRECTNESS}_${W_FORMAT}_${W_REASONING}_${TIMESTAMP}"

# Create log directory and file
LOG_DIR="${PROJECT_DIR}/outputs/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}.log"

echo "Log file: $LOG_FILE"

# Run training with 8-GPU optimized settings (use tee to save logs)
python -m verl.trainer.main_ppo \
    data.train_files="$HOME/data/gsm8k/train.parquet" \
    data.val_files="$HOME/data/gsm8k/test.parquet" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    custom_reward_function.path="${PROJECT_DIR}/scripts/reward_fn.py" \
    custom_reward_function.name=compute_score \
    '+custom_reward_function.reward_kwargs.w_correctness'=$W_CORRECTNESS \
    '+custom_reward_function.reward_kwargs.w_format'=$W_FORMAT \
    '+custom_reward_function.reward_kwargs.w_reasoning'=$W_REASONING \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=$LR_ACTOR \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    critic.optim.lr=$LR_CRITIC \
    critic.model.path="$MODEL_PATH" \
    critic.ppo_micro_batch_size_per_gpu=16 \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.project_name=verl_multi_reward_8gpu \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.logger='["console","wandb"]' \
    trainer.val_before_train=True \
    2>&1 | tee "$LOG_FILE"

echo "Training completed!"
echo "Log saved to: $LOG_FILE"
echo "Check wandb for results: https://wandb.ai/${WANDB_ENTITY:-your-entity}/verl_multi_reward_8gpu"
