#!/bin/bash
# ==============================================================================
# GRPO Reward Ablation Runner for GSM8K
# ==============================================================================
#
# Runs predefined GRPO reward-weight ablations with Qwen2.5-1.5B-Instruct or an
# SFT warmup checkpoint. This script is GRPO-only.
#
# Usage:
#   ./scripts/run_ablation_experiments.sh
#   ./scripts/run_ablation_experiments.sh --dry-run
#   ./scripts/run_ablation_experiments.sh --exp 1 3 6
#   ./scripts/run_ablation_experiments.sh --model ./checkpoints/sft_warmup/final
#
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

declare -a EXPERIMENTS=(
    "correctness_only|1.0|0.0|0.0|legacy_reasoning|Correctness-only GRPO baseline"
    "default_multi|0.7|0.1|0.2|legacy_reasoning|Default multi-reward GRPO"
    "high_correctness|0.8|0.1|0.1|legacy_reasoning|Higher correctness weight"
    "high_reasoning|0.5|0.1|0.4|legacy_reasoning|Higher reasoning-structure weight"
    "format_emphasis|0.5|0.3|0.2|legacy_reasoning|Higher format weight"
    "answer_conditioned_diagnostic|0.7|0.1|0.2|answer_conditioned_reasoning|Reward-hacking diagnostic run"
)

DRY_RUN=false
SELECTED_EXPS=()
MODEL="./checkpoints/sft_warmup/final"
CONFIG="config/grpo_gsm8k.yaml"
GROUP_SIZE=4
BATCH_SIZE=2
MAX_RESPONSE_LENGTH=256
TOTAL_STEPS=1000
LR=""
KL_COEF=""
USE_VLLM=true
VLLM_SYNC_INTERVAL=1
VLLM_GPU_MEMORY_UTILIZATION=0.25

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --exp)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SELECTED_EXPS+=("$1")
                shift
            done
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --group-size)
            GROUP_SIZE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-response-length)
            MAX_RESPONSE_LENGTH="$2"
            shift 2
            ;;
        --total-steps)
            TOTAL_STEPS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --kl-coef)
            KL_COEF="$2"
            shift 2
            ;;
        --use-vllm)
            USE_VLLM=true
            shift
            ;;
        --no-vllm)
            USE_VLLM=false
            shift
            ;;
        --vllm-sync-interval)
            VLLM_SYNC_INTERVAL="$2"
            shift 2
            ;;
        --vllm-gpu-memory-utilization)
            VLLM_GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run                 Print commands without running"
            echo "  --exp N [M ...]           Run selected experiment numbers"
            echo "  --model PATH              Model/SFT checkpoint path"
            echo "  --config PATH             GRPO config path"
            echo "  --group-size N            GRPO group size"
            echo "  --batch-size N            Unique prompts per GRPO step"
            echo "  --max-response-length N   Max response tokens"
            echo "  --total-steps N           Training steps per experiment"
            echo "  --lr VALUE                Actor learning rate"
            echo "  --kl-coef VALUE           KL coefficient"
            echo "  --use-vllm/--no-vllm      Select rollout engine"
            echo "  --vllm-sync-interval N    Reload vLLM from policy every N steps"
            echo "  --vllm-gpu-memory-utilization F"
            echo ""
            echo "Experiments:"
            for i in "${!EXPERIMENTS[@]}"; do
                IFS='|' read -r name w_c w_f w_r reward_mode desc <<< "${EXPERIMENTS[$i]}"
                printf "  %d. %-32s %s/%s/%s %-30s %s\n" $((i+1)) "$name" "$w_c" "$w_f" "$w_r" "$reward_mode" "$desc"
            done
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

if [[ ${#SELECTED_EXPS[@]} -eq 0 ]]; then
    for i in "${!EXPERIMENTS[@]}"; do
        SELECTED_EXPS+=($((i+1)))
    done
fi

echo -e "${BLUE}=============================================="
echo "          GRPO Reward Ablation Study"
echo "==============================================${NC}"
echo "Model:              $MODEL"
echo "Config:             $CONFIG"
echo "Group size:         $GROUP_SIZE"
echo "Batch size:         $BATCH_SIZE"
echo "Max response length:$MAX_RESPONSE_LENGTH"
echo "Total steps:        $TOTAL_STEPS"
echo "Rollout:            $([ "$USE_VLLM" = true ] && echo "vLLM" || echo "HF live policy")"
if [[ "$USE_VLLM" == true ]]; then
    echo "vLLM sync interval: $VLLM_SYNC_INTERVAL"
    echo "vLLM memory util:   $VLLM_GPU_MEMORY_UTILIZATION"
fi
echo ""

build_command() {
    local w_c="$1"
    local w_f="$2"
    local w_r="$3"
    local reward_mode="$4"
    local cmd="./scripts/train_grpo.sh --config $CONFIG --model $MODEL --group-size $GROUP_SIZE --batch-size $BATCH_SIZE --max-response-length $MAX_RESPONSE_LENGTH --total-steps $TOTAL_STEPS --reward-mode $reward_mode --w-correctness $w_c --w-format $w_f --w-reasoning $w_r"
    if [[ -n "$LR" ]]; then
        cmd="$cmd --lr $LR"
    fi
    if [[ -n "$KL_COEF" ]]; then
        cmd="$cmd --kl-coef $KL_COEF"
    fi
    if [[ "$USE_VLLM" == true ]]; then
        cmd="$cmd --use-vllm --vllm-sync-interval $VLLM_SYNC_INTERVAL --vllm-gpu-memory-utilization $VLLM_GPU_MEMORY_UTILIZATION"
    else
        cmd="$cmd --no-vllm"
    fi
    echo "$cmd"
}

if [[ "$DRY_RUN" == true ]]; then
    echo -e "${YELLOW}[DRY RUN] Commands:${NC}"
    for exp_num in "${SELECTED_EXPS[@]}"; do
        idx=$((exp_num-1))
        if [[ $idx -ge 0 && $idx -lt ${#EXPERIMENTS[@]} ]]; then
            IFS='|' read -r name w_c w_f w_r reward_mode desc <<< "${EXPERIMENTS[$idx]}"
            echo ""
            echo "# $exp_num. $name - $desc"
            build_command "$w_c" "$w_f" "$w_r" "$reward_mode"
        fi
    done
    exit 0
fi

SUMMARY_LOG="${PROJECT_DIR}/outputs/logs/grpo_ablation_summary_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$SUMMARY_LOG")"

{
    echo "GRPO ablation study"
    echo "started_at=$(date -Iseconds)"
    echo "model=$MODEL"
    echo "config=$CONFIG"
    echo "group_size=$GROUP_SIZE"
    echo "batch_size=$BATCH_SIZE"
    echo "max_response_length=$MAX_RESPONSE_LENGTH"
    echo "total_steps=$TOTAL_STEPS"
    echo ""
} > "$SUMMARY_LOG"

FAILED=()
SUCCEEDED=()
TOTAL=${#SELECTED_EXPS[@]}
CURRENT=0

for exp_num in "${SELECTED_EXPS[@]}"; do
    idx=$((exp_num-1))
    if [[ $idx -lt 0 || $idx -ge ${#EXPERIMENTS[@]} ]]; then
        echo -e "${RED}Skip unknown experiment: $exp_num${NC}"
        continue
    fi

    CURRENT=$((CURRENT+1))
    IFS='|' read -r name w_c w_f w_r reward_mode desc <<< "${EXPERIMENTS[$idx]}"
    cmd=$(build_command "$w_c" "$w_f" "$w_r" "$reward_mode")

    echo -e "${BLUE}=============================================="
    echo "Experiment $CURRENT/$TOTAL: $name"
    echo "Weights: c=$w_c f=$w_f r=$w_r"
    echo "Reward mode: $reward_mode"
    echo "Description: $desc"
    echo -e "==============================================${NC}"

    {
        echo "experiment=$name"
        echo "weights=$w_c/$w_f/$w_r"
        echo "reward_mode=$reward_mode"
        echo "command=$cmd"
        echo "started_at=$(date -Iseconds)"
    } >> "$SUMMARY_LOG"

    START_TIME=$(date +%s)
    if eval "$cmd"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo -e "${GREEN}Done: $name (${DURATION}s)${NC}"
        SUCCEEDED+=("$name")
        echo "status=success duration_seconds=$DURATION" >> "$SUMMARY_LOG"
    else
        echo -e "${RED}Failed: $name${NC}"
        FAILED+=("$name")
        echo "status=failed" >> "$SUMMARY_LOG"
    fi
    echo "" >> "$SUMMARY_LOG"
done

echo -e "${BLUE}=============================================="
echo "Summary"
echo -e "==============================================${NC}"
echo -e "${GREEN}Succeeded: ${#SUCCEEDED[@]}/${TOTAL}${NC}"
for name in "${SUCCEEDED[@]}"; do
    echo "  $name"
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo -e "${RED}Failed: ${#FAILED[@]}/${TOTAL}${NC}"
    for name in "${FAILED[@]}"; do
        echo "  $name"
    done
fi

echo "Summary log: $SUMMARY_LOG"
