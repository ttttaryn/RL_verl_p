#!/bin/bash
# ==============================================================================
# 4x A100 Optimized PPO Training Script for GSM8K Multi-Reward
# ==============================================================================
#
# Defaults target 4x A100 40GB/80GB. Use --gpus to override for other A100 nodes.
#
# Usage:
#   ./scripts/train_4gpu.sh
#   ./scripts/train_4gpu.sh --model ./checkpoints/sft_warmup/final
#   ./scripts/train_4gpu.sh --gpus 8 --batch-size 1024
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

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# 4x A100 optimized defaults
MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_BATCH_SIZE=512          # 4 GPUs x 128 prompts/GPU effective global batch
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=512
LR_ACTOR=1e-6
LR_CRITIC=1e-5
KL_COEF=0.001
N_GPUS=4
TOTAL_EPOCHS=15
SAVE_FREQ=5
TEST_FREQ=5

W_CORRECTNESS=0.7
W_FORMAT=0.1
W_REASONING=0.2

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
            echo "Usage: $0 [--model PATH] [--batch-size N] [--gpus N] [--lr LR] [--epochs N]"
            echo "          [--w-correctness F] [--w-format F] [--w-reasoning F]"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "4x A100 PPO Training Configuration"
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

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="gsm8k_${N_GPUS}gpu_w${W_CORRECTNESS}_${W_FORMAT}_${W_REASONING}_${TIMESTAMP}"

LOG_DIR="${PROJECT_DIR}/outputs/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}.log"

echo "Log file: $LOG_FILE"

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
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    critic.optim.lr=$LR_CRITIC \
    critic.model.path="$MODEL_PATH" \
    critic.ppo_micro_batch_size_per_gpu=8 \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.project_name=verl_multi_reward_${N_GPUS}gpu \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.logger='["console","wandb"]' \
    trainer.val_before_train=True \
    2>&1 | tee "$LOG_FILE"

echo "Training completed!"
echo "Log saved to: $LOG_FILE"
echo "Check wandb for results: https://wandb.ai/${WANDB_ENTITY:-your-entity}/verl_multi_reward_${N_GPUS}gpu"
