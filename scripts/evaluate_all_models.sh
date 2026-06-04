#!/bin/bash
# ==============================================================================
# 并行评估多个模型脚本
# ==============================================================================
#
# 该脚本并行评估多个上传到 Hugging Face 的模型
# 每个模型使用一个 GPU 进行推理
#
# Usage:
#   ./scripts/evaluate_all_models.sh              # 评估所有预定义模型
#   ./scripts/evaluate_all_models.sh --dry-run    # 只打印配置，不实际运行
#   ./scripts/evaluate_all_models.sh --num-samples 100  # 限制评估样本数
#
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==============================================================================
# 模型配置定义
# 格式: "模型HF路径|正确性权重|格式权重|推理权重|描述"
# ==============================================================================
HF_USER="leixinlin"
BASE_MODEL_NAME="qwen2.5-1.5b-gsm8k-grpo"
BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
SFT_CHECKPOINT="./checkpoints/sft_warmup/final"
GRPO_CONFIG="config/grpo_gsm8k.yaml"
REWARD_MODE="answer_conditioned_reasoning"
GROUP_SIZE=4
TRAIN_BATCH_SIZE=2
KL_COEF=0.001
ROLLOUT_ENGINE="vllm"
VLLM_SYNC_INTERVAL=1
VLLM_GPU_MEMORY_UTILIZATION=0.25

declare -a MODELS=(
    "${HF_USER}/${BASE_MODEL_NAME}-w1.0_0.0_0.0|1.0|0.0|0.0|基线：仅正确性奖励"
    "${HF_USER}/${BASE_MODEL_NAME}-w0.7_0.1_0.2|0.7|0.1|0.2|默认多奖励配置"
    "${HF_USER}/${BASE_MODEL_NAME}-w0.8_0.1_0.1|0.8|0.1|0.1|高正确性权重"
    "${HF_USER}/${BASE_MODEL_NAME}-w0.5_0.1_0.4|0.5|0.1|0.4|高推理权重"
    "${HF_USER}/${BASE_MODEL_NAME}-w0.6_0.2_0.2|0.6|0.2|0.2|平衡配置"
    "${HF_USER}/${BASE_MODEL_NAME}-w0.5_0.3_0.2|0.5|0.3|0.2|强调格式"
)

# ==============================================================================
# 默认参数
# ==============================================================================
RESULTS_DIR="${PROJECT_DIR}/results"
NUM_SAMPLES=-1  # -1 表示评估所有样本
MAX_TOKENS=512
TEMPERATURE=0.0  # 使用贪婪解码以确保结果可复现
BATCH_SIZE=16    # 批量推理大小，提高GPU利用率
DRY_RUN=false
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "1")

# ==============================================================================
# 参数解析
# ==============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --reward-mode)
            REWARD_MODE="$2"
            shift 2
            ;;
        --group-size)
            GROUP_SIZE="$2"
            shift 2
            ;;
        --train-batch-size)
            TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --kl-coef)
            KL_COEF="$2"
            shift 2
            ;;
        --rollout-engine)
            ROLLOUT_ENGINE="$2"
            shift 2
            ;;
        --vllm-sync-interval)
            VLLM_SYNC_INTERVAL="$2"
            shift 2
            ;;
        --vllm-gpu-memory-utilization)
            VLLM_GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run              只打印配置，不实际运行"
            echo "  --num-samples N        评估样本数量 (-1 表示全部)"
            echo "  --batch-size N         批量推理大小 (默认: 16)"
            echo "  --max-tokens N         最大生成 token 数"
            echo "  --temperature VALUE    采样温度 (0 为贪婪解码)"
            echo "  --reward-mode MODE     记录到结果 JSON 的 reward mode"
            echo "  --group-size N         记录到结果 JSON 的 GRPO group size"
            echo "  --train-batch-size N   记录到结果 JSON 的 GRPO train batch size"
            echo "  --kl-coef VALUE        记录到结果 JSON 的 KL 系数"
            echo "  --rollout-engine NAME  记录到结果 JSON 的 rollout engine"
            echo "  --vllm-sync-interval N"
            echo "  --vllm-gpu-memory-utilization F"
            echo "  --results-dir PATH     结果保存目录"
            echo "  -h, --help             显示帮助信息"
            echo ""
            echo "预定义模型:"
            for i in "${!MODELS[@]}"; do
                IFS='|' read -r model w_c w_f w_r desc <<< "${MODELS[$i]}"
                printf "  %d. %s\n     权重: %.1f/%.1f/%.1f - %s\n" $((i+1)) "$model" "$w_c" "$w_f" "$w_r" "$desc"
            done
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# ==============================================================================
# 创建结果目录
# ==============================================================================
mkdir -p "$RESULTS_DIR"

# ==============================================================================
# 打印配置
# ==============================================================================
echo -e "${BLUE}=============================================="
echo "       并行模型评估 (Parallel Evaluation)"
echo "==============================================${NC}"
echo ""
echo -e "${YELLOW}配置:${NC}"
echo "  结果目录:     $RESULTS_DIR"
echo "  评估样本数:   $NUM_SAMPLES (-1 表示全部)"
echo "  批量大小:     $BATCH_SIZE"
echo "  最大 tokens:  $MAX_TOKENS"
echo "  温度:         $TEMPERATURE"
echo "  Reward mode:  $REWARD_MODE"
echo "  Group size:   $GROUP_SIZE"
echo "  Train batch:  $TRAIN_BATCH_SIZE"
echo "  KL coef:      $KL_COEF"
echo "  Rollout:      $ROLLOUT_ENGINE"
echo "  vLLM sync:    $VLLM_SYNC_INTERVAL"
echo "  vLLM mem:     $VLLM_GPU_MEMORY_UTILIZATION"
echo "  可用 GPU 数:  $NUM_GPUS"
echo ""
echo -e "${YELLOW}待评估模型 (${#MODELS[@]} 个):${NC}"
printf "%-4s %-50s %-15s %s\n" "No." "模型" "权重(C/F/R)" "描述"
echo "--------------------------------------------------------------------------------"

for i in "${!MODELS[@]}"; do
    IFS='|' read -r model w_c w_f w_r desc <<< "${MODELS[$i]}"
    printf "%-4d %-50s %-15s %s\n" $((i+1)) "$model" "$w_c/$w_f/$w_r" "$desc"
done
echo ""

if [[ "$DRY_RUN" == true ]]; then
    echo -e "${YELLOW}[DRY RUN] 将要执行的命令:${NC}"
    echo ""
    for i in "${!MODELS[@]}"; do
        IFS='|' read -r model w_c w_f w_r desc <<< "${MODELS[$i]}"
        # 从模型名称提取权重字符串作为输出文件名
        weight_str=$(echo "$model" | grep -oP 'w[\d.]+_[\d.]+_[\d.]+' || echo "model_$i")
        output_file="${RESULTS_DIR}/eval_${weight_str}.json"
        
        echo -e "${GREEN}模型 $((i+1)):${NC} $model"
        echo "  CUDA_VISIBLE_DEVICES=$((i % NUM_GPUS)) python scripts/evaluate.py \\"
        echo "    --model-path \"$model\" \\"
        echo "    --output \"$output_file\" \\"
        echo "    --num-samples $NUM_SAMPLES \\"
        echo "    --batch-size $BATCH_SIZE \\"
        echo "    --max-tokens $MAX_TOKENS \\"
        echo "    --temperature $TEMPERATURE \\"
        echo "    --base-model-name \"$BASE_MODEL\" \\"
        echo "    --sft-checkpoint \"$SFT_CHECKPOINT\" \\"
        echo "    --grpo-config \"$GRPO_CONFIG\" \\"
        echo "    --reward-mode \"$REWARD_MODE\" \\"
        echo "    --group-size $GROUP_SIZE \\"
        echo "    --train-batch-size $TRAIN_BATCH_SIZE \\"
        echo "    --kl-coef $KL_COEF \\"
        echo "    --rollout-engine \"$ROLLOUT_ENGINE\" \\"
        echo "    --vllm-sync-interval $VLLM_SYNC_INTERVAL \\"
        echo "    --vllm-gpu-memory-utilization $VLLM_GPU_MEMORY_UTILIZATION \\"
        echo "    --w-correctness $w_c \\"
        echo "    --w-format $w_f \\"
        echo "    --w-reasoning $w_r"
        echo ""
    done
    exit 0
fi

# ==============================================================================
# 并行运行评估
# ==============================================================================
echo -e "${BLUE}开始并行评估...${NC}"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 创建临时目录存放 PID 文件
PID_DIR=$(mktemp -d)
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "$LOG_DIR"

# 启动所有评估任务
for i in "${!MODELS[@]}"; do
    IFS='|' read -r model w_c w_f w_r desc <<< "${MODELS[$i]}"
    
    # 从模型名称提取权重字符串
    weight_str=$(echo "$model" | grep -oP 'w[\d.]+_[\d.]+_[\d.]+' || echo "model_$i")
    output_file="${RESULTS_DIR}/eval_${weight_str}.json"
    log_file="${LOG_DIR}/eval_${weight_str}.log"
    
    # 分配 GPU (循环使用)
    gpu_id=$((i % NUM_GPUS))
    
    echo -e "${GREEN}启动评估 $((i+1))/${#MODELS[@]}:${NC} $model (GPU: $gpu_id)"
    
    # 后台运行评估
    CUDA_VISIBLE_DEVICES=$gpu_id python scripts/evaluate.py \
        --model-path "$model" \
        --output "$output_file" \
        --num-samples $NUM_SAMPLES \
        --batch-size $BATCH_SIZE \
        --max-tokens $MAX_TOKENS \
        --temperature $TEMPERATURE \
        --base-model-name "$BASE_MODEL" \
        --sft-checkpoint "$SFT_CHECKPOINT" \
        --grpo-config "$GRPO_CONFIG" \
        --reward-mode "$REWARD_MODE" \
        --group-size $GROUP_SIZE \
        --train-batch-size $TRAIN_BATCH_SIZE \
        --kl-coef $KL_COEF \
        --rollout-engine "$ROLLOUT_ENGINE" \
        --vllm-sync-interval $VLLM_SYNC_INTERVAL \
        --vllm-gpu-memory-utilization $VLLM_GPU_MEMORY_UTILIZATION \
        --w-correctness $w_c \
        --w-format $w_f \
        --w-reasoning $w_r \
        > "$log_file" 2>&1 &
    
    # 保存 PID
    echo $! > "${PID_DIR}/eval_${i}.pid"
    
    # 如果 GPU 数量有限，每启动 NUM_GPUS 个任务后等待一批完成
    if (( (i + 1) % NUM_GPUS == 0 && i + 1 < ${#MODELS[@]} )); then
        echo -e "${YELLOW}等待当前批次完成...${NC}"
        wait
    fi
done

# 等待所有任务完成
echo ""
echo -e "${YELLOW}等待所有评估任务完成...${NC}"
wait

# 清理 PID 目录
rm -rf "$PID_DIR"

# 计算总耗时
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# ==============================================================================
# 汇总结果
# ==============================================================================
echo ""
echo -e "${BLUE}=============================================="
echo "               评估结果汇总"
echo "==============================================${NC}"
echo ""

# 创建汇总文件
SUMMARY_FILE="${RESULTS_DIR}/evaluation_summary_$(date +%Y%m%d_%H%M%S).json"

echo "{" > "$SUMMARY_FILE"
echo "  \"evaluation_time\": \"$(date -Iseconds)\"," >> "$SUMMARY_FILE"
echo "  \"duration_seconds\": $DURATION," >> "$SUMMARY_FILE"
echo "  \"num_samples\": $NUM_SAMPLES," >> "$SUMMARY_FILE"
echo "  \"results\": [" >> "$SUMMARY_FILE"

FIRST=true
for i in "${!MODELS[@]}"; do
    IFS='|' read -r model w_c w_f w_r desc <<< "${MODELS[$i]}"
    weight_str=$(echo "$model" | grep -oP 'w[\d.]+_[\d.]+_[\d.]+' || echo "model_$i")
    output_file="${RESULTS_DIR}/eval_${weight_str}.json"
    
    if [[ -f "$output_file" ]]; then
        # 提取关键指标
        accuracy=$(python -c "import json; d=json.load(open('$output_file')); print(round(d.get('summary', {}).get('accuracy', 0), 4))" 2>/dev/null || echo "N/A")
        avg_score=$(python -c "import json; d=json.load(open('$output_file')); print(round(d.get('summary', {}).get('score_mean', 0), 4))" 2>/dev/null || echo "N/A")
        
        echo -e "${GREEN}✓${NC} $model"
        echo "  权重: $w_c/$w_f/$w_r | 准确率: $accuracy | 综合得分: $avg_score"
        
        # 添加到汇总 JSON
        if [[ "$FIRST" == true ]]; then
            FIRST=false
        else
            echo "    ," >> "$SUMMARY_FILE"
        fi
        echo "    {" >> "$SUMMARY_FILE"
        echo "      \"model\": \"$model\"," >> "$SUMMARY_FILE"
        echo "      \"weights\": {\"correctness\": $w_c, \"format\": $w_f, \"reasoning\": $w_r}," >> "$SUMMARY_FILE"
        echo "      \"description\": \"$desc\"," >> "$SUMMARY_FILE"
        echo "      \"accuracy\": \"$accuracy\"," >> "$SUMMARY_FILE"
        echo "      \"average_score\": \"$avg_score\"," >> "$SUMMARY_FILE"
        echo "      \"result_file\": \"$output_file\"" >> "$SUMMARY_FILE"
        echo -n "    }" >> "$SUMMARY_FILE"
    else
        echo -e "${RED}✗${NC} $model - 结果文件不存在"
    fi
done

echo "" >> "$SUMMARY_FILE"
echo "  ]" >> "$SUMMARY_FILE"
echo "}" >> "$SUMMARY_FILE"

echo ""
echo -e "${BLUE}----------------------------------------------${NC}"
echo "总耗时: ${DURATION}s"
echo "结果目录: $RESULTS_DIR"
echo "汇总文件: $SUMMARY_FILE"
echo "日志目录: $LOG_DIR"
echo -e "${BLUE}=============================================="
echo -e "             评估完成!${NC}"
