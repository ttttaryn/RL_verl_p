#!/bin/bash
# ==============================================================================
# 批量运行对照实验脚本 (Ablation Study)
# ==============================================================================
#
# 该脚本用于在 8x A100 80GB 配置下运行多组奖励权重对照实验
# 每组实验使用不同的 (correctness, format, reasoning) 权重组合
#
# Usage:
#   ./scripts/run_ablation_experiments.sh              # 运行所有预定义实验
#   ./scripts/run_ablation_experiments.sh --dry-run    # 只打印配置，不实际运行
#   ./scripts/run_ablation_experiments.sh --exp 1 3    # 只运行实验 1 和 3
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
# 实验配置定义
# 格式: "实验名称|正确性权重|格式权重|推理权重|描述"
# ==============================================================================
declare -a EXPERIMENTS=(
    "baseline_correctness|1.0|0.0|0.0|基线：仅正确性奖励"
    "default_multi|0.7|0.1|0.2|默认多奖励配置"
    "high_correctness|0.8|0.1|0.1|高正确性权重"
    "high_reasoning|0.5|0.1|0.4|高推理权重"
    "balanced|0.6|0.2|0.2|平衡配置"
    "format_emphasis|0.5|0.3|0.2|强调格式"
)

# ==============================================================================
# 参数解析
# ==============================================================================
DRY_RUN=false
SELECTED_EXPS=()
EXTRA_ARGS=""

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
        --model|--batch-size|--lr|--kl-coef|--epochs)
            EXTRA_ARGS="$EXTRA_ARGS $1 $2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run          只打印配置，不实际运行"
            echo "  --exp N [M ...]    只运行指定编号的实验（从1开始）"
            echo "  --model PATH       指定模型路径"
            echo "  --batch-size N     指定 batch size"
            echo "  --lr VALUE         指定学习率"
            echo "  --epochs N         指定训练轮数"
            echo "  -h, --help         显示帮助信息"
            echo ""
            echo "预定义实验:"
            for i in "${!EXPERIMENTS[@]}"; do
                IFS='|' read -r name w_c w_f w_r desc <<< "${EXPERIMENTS[$i]}"
                printf "  %d. %-20s (%.1f/%.1f/%.1f) - %s\n" $((i+1)) "$name" "$w_c" "$w_f" "$w_r" "$desc"
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
# 打印实验计划
# ==============================================================================
echo -e "${BLUE}=============================================="
echo "       批量对照实验 (Ablation Study)"
echo "==============================================${NC}"
echo ""

# 确定要运行的实验
if [[ ${#SELECTED_EXPS[@]} -eq 0 ]]; then
    # 运行所有实验
    for i in "${!EXPERIMENTS[@]}"; do
        SELECTED_EXPS+=($((i+1)))
    done
fi

echo -e "${YELLOW}计划运行 ${#SELECTED_EXPS[@]} 个实验:${NC}"
echo ""
printf "%-4s %-22s %-15s %s\n" "No." "实验名称" "权重(C/F/R)" "描述"
echo "--------------------------------------------------------------"

for exp_num in "${SELECTED_EXPS[@]}"; do
    idx=$((exp_num-1))
    if [[ $idx -ge 0 && $idx -lt ${#EXPERIMENTS[@]} ]]; then
        IFS='|' read -r name w_c w_f w_r desc <<< "${EXPERIMENTS[$idx]}"
        printf "%-4d %-22s %-15s %s\n" "$exp_num" "$name" "$w_c/$w_f/$w_r" "$desc"
    else
        echo -e "${RED}警告: 实验 $exp_num 不存在，跳过${NC}"
    fi
done
echo ""

if [[ -n "$EXTRA_ARGS" ]]; then
    echo -e "${YELLOW}额外参数: ${EXTRA_ARGS}${NC}"
    echo ""
fi

if [[ "$DRY_RUN" == true ]]; then
    echo -e "${YELLOW}[DRY RUN] 仅打印配置，不实际运行${NC}"
    echo ""
    for exp_num in "${SELECTED_EXPS[@]}"; do
        idx=$((exp_num-1))
        if [[ $idx -ge 0 && $idx -lt ${#EXPERIMENTS[@]} ]]; then
            IFS='|' read -r name w_c w_f w_r desc <<< "${EXPERIMENTS[$idx]}"
            echo -e "${GREEN}实验 $exp_num ($name):${NC}"
            echo "  ./scripts/train_8gpu.sh --w-correctness $w_c --w-format $w_f --w-reasoning $w_r $EXTRA_ARGS"
            echo ""
        fi
    done
    exit 0
fi

# ==============================================================================
# 运行实验
# ==============================================================================
TOTAL=${#SELECTED_EXPS[@]}
CURRENT=0
FAILED=()
SUCCEEDED=()

# 创建实验汇总日志
SUMMARY_LOG="${PROJECT_DIR}/outputs/logs/ablation_summary_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$SUMMARY_LOG")"

echo "实验汇总日志: $SUMMARY_LOG"
echo ""

{
    echo "=============================================="
    echo "批量对照实验汇总"
    echo "开始时间: $(date)"
    echo "=============================================="
    echo ""
} > "$SUMMARY_LOG"

for exp_num in "${SELECTED_EXPS[@]}"; do
    idx=$((exp_num-1))
    if [[ $idx -lt 0 || $idx -ge ${#EXPERIMENTS[@]} ]]; then
        continue
    fi
    
    CURRENT=$((CURRENT+1))
    IFS='|' read -r name w_c w_f w_r desc <<< "${EXPERIMENTS[$idx]}"
    
    echo -e "${BLUE}=============================================="
    echo -e "实验 $CURRENT/$TOTAL: $name"
    echo -e "权重: correctness=$w_c, format=$w_f, reasoning=$w_r"
    echo -e "描述: $desc"
    echo -e "==============================================${NC}"
    
    START_TIME=$(date +%s)
    
    {
        echo "----------------------------------------------"
        echo "实验 $exp_num: $name"
        echo "权重: $w_c / $w_f / $w_r"
        echo "描述: $desc"
        echo "开始时间: $(date)"
    } >> "$SUMMARY_LOG"
    
    # 运行实验
    if ./scripts/train_8gpu.sh \
        --w-correctness "$w_c" \
        --w-format "$w_f" \
        --w-reasoning "$w_r" \
        $EXTRA_ARGS; then
        
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        echo -e "${GREEN}✓ 实验 $name 完成 (耗时: ${DURATION}s)${NC}"
        SUCCEEDED+=("$name")
        
        {
            echo "状态: 成功"
            echo "耗时: ${DURATION}s"
            echo ""
        } >> "$SUMMARY_LOG"
    else
        echo -e "${RED}✗ 实验 $name 失败${NC}"
        FAILED+=("$name")
        
        {
            echo "状态: 失败"
            echo ""
        } >> "$SUMMARY_LOG"
    fi
    
    # ==================================================================
    # WandB 同步等待 - 确保数据完整上传后再开始下一个实验
    # ==================================================================
    echo -e "${YELLOW}等待 WandB 完成同步...${NC}"
    sleep 15  # 等待15秒让 WandB 完成异步上传
    
    # 强制同步任何离线或未完成的数据
    if command -v wandb &> /dev/null; then
        echo "正在同步 WandB 数据..."
        wandb sync --sync-all 2>/dev/null || true
        sleep 5  # 额外等待同步命令完成
    fi
    
    echo -e "${GREEN}WandB 同步完成${NC}"
    echo ""
done

# ==============================================================================
# 打印汇总
# ==============================================================================
echo -e "${BLUE}=============================================="
echo "               实验汇总"
echo "==============================================${NC}"
echo -e "${GREEN}成功: ${#SUCCEEDED[@]}/${TOTAL}${NC}"
for name in "${SUCCEEDED[@]}"; do
    echo -e "  ${GREEN}✓${NC} $name"
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo -e "${RED}失败: ${#FAILED[@]}/${TOTAL}${NC}"
    for name in "${FAILED[@]}"; do
        echo -e "  ${RED}✗${NC} $name"
    done
fi

{
    echo "=============================================="
    echo "实验汇总"
    echo "结束时间: $(date)"
    echo "成功: ${#SUCCEEDED[@]}/${TOTAL}"
    echo "失败: ${#FAILED[@]}/${TOTAL}"
    echo "=============================================="
} >> "$SUMMARY_LOG"

echo ""
echo "汇总日志已保存到: $SUMMARY_LOG"
echo -e "${BLUE}完成所有实验!${NC}"
