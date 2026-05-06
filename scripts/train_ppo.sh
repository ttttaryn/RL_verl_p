#!/bin/bash
# Training Script for PPO with Multi-Reward System
# 
# Usage:
#   ./train_ppo.sh                    # Run default multi-reward training
#   ./train_ppo.sh --exp 1            # Run experiment 1 (correctness only)
#   ./train_ppo.sh --exp 2            # Run experiment 2 (multi-reward default)
#   ./train_ppo.sh --exp 3            # Run experiment 3 (heavy reasoning)
#   ./train_ppo.sh --exp 4            # Run experiment 4 (balanced)
#   ./train_ppo.sh --exp 5            # Run experiment 5 (XML format)
#   ./train_ppo.sh --config custom    # Run with custom config

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
EXPERIMENT=""
CONFIG_NAME="ppo_gsm8k"
WANDB_PROJECT="verl_multi_reward"
NUM_GPUS=1
NNODES=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp|--experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        --config)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --nodes)
            NNODES="$2"
            shift 2
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --exp, --experiment NUM    Run a predefined experiment (1-5)"
            echo "  --config NAME              Use a specific config file"
            echo "  --gpus NUM                 Number of GPUs per node (default: 1)"
            echo "  --nodes NUM                Number of nodes (default: 1)"
            echo "  --wandb-project NAME       WandB project name"
            echo "  -h, --help                 Show this help message"
            echo ""
            echo "Predefined experiments:"
            echo "  1: Correctness only (baseline)"
            echo "  2: Multi-reward with default weights"
            echo "  3: Heavy reasoning weight"
            echo "  4: Balanced weights"
            echo "  5: XML format training"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set config based on experiment number
if [[ -n "$EXPERIMENT" ]]; then
    case $EXPERIMENT in
        1) CONFIG_NAME="experiments/exp1_correctness_only" ;;
        2) CONFIG_NAME="experiments/exp2_multi_reward_default" ;;
        3) CONFIG_NAME="experiments/exp3_heavy_reasoning" ;;
        4) CONFIG_NAME="experiments/exp4_balanced" ;;
        5) CONFIG_NAME="experiments/exp5_xml_format" ;;
        *)
            echo "Invalid experiment number: $EXPERIMENT"
            echo "Valid experiments: 1-5"
            exit 1
            ;;
    esac
    echo "Running experiment $EXPERIMENT with config: $CONFIG_NAME"
fi

# Check if data exists
GSM8K_TRAIN="$HOME/data/gsm8k/train.parquet"
GSM8K_TEST="$HOME/data/gsm8k/test.parquet"

if [[ ! -f "$GSM8K_TRAIN" ]] || [[ ! -f "$GSM8K_TEST" ]]; then
    echo "Error: GSM8K data files not found!"
    echo "Expected:"
    echo "  $GSM8K_TRAIN"
    echo "  $GSM8K_TEST"
    echo ""
    echo "Please download the GSM8K dataset first using:"
    echo "  python examples/data_preprocess/gsm8k.py --local_save_dir \$HOME/data/gsm8k/"
    exit 1
fi

# Export environment variables
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Create logs directory
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# Generate timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/train_${CONFIG_NAME//\//_}_$TIMESTAMP.log"

echo "=============================================="
echo "PPO Training with Multi-Reward System"
echo "=============================================="
echo "Config: $CONFIG_NAME"
echo "GPUs per node: $NUM_GPUS"
echo "Number of nodes: $NNODES"
echo "Log file: $LOG_FILE"
echo "=============================================="

# Run training
python3 -m verl.trainer.main_ppo \
    --config-path="$PROJECT_DIR/config" \
    --config-name="$CONFIG_NAME" \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=$NNODES \
    trainer.project_name="$WANDB_PROJECT" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Training completed!"
echo "Log saved to: $LOG_FILE"
