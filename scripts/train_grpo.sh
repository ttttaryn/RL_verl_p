#!/bin/bash
# GRPO Training Launcher for GSM8K Multi-Reward
#
# GRPO eliminates the critic model, saving ~50% VRAM vs PPO.
# This means you can train larger models or use larger batch sizes
# on the same hardware.
#
# Usage:
#   ./scripts/train_grpo.sh                          # default (single GPU)
#   ./scripts/train_grpo.sh --group-size 8            # larger groups
#   ./scripts/train_grpo.sh --n-gpus 1                # single GPU
#   ./scripts/train_grpo.sh --model ./checkpoints/sft_warmup/final  # from SFT warmup

set -e

# ── Default settings ───────────────────────────────────────────────────
N_GPUS=1
GROUP_SIZE=4
BATCH_SIZE=8
TOTAL_STEPS=1000
CONFIG="config/grpo_gsm8k.yaml"
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
W_CORRECTNESS=0.7
W_FORMAT=0.1
W_REASONING=0.2

# ── Parse CLI args ─────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --n-gpus)       N_GPUS="$2"; shift 2 ;;
    --group-size)   GROUP_SIZE="$2"; shift 2 ;;
    --batch-size)   BATCH_SIZE="$2"; shift 2 ;;
    --total-steps)  TOTAL_STEPS="$2"; shift 2 ;;
    --config)       CONFIG="$2"; shift 2 ;;
    --model)        MODEL="$2"; shift 2 ;;
    --w-correctness) W_CORRECTNESS="$2"; shift 2 ;;
    --w-format)     W_FORMAT="$2"; shift 2 ;;
    --w-reasoning)  W_REASONING="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--n-gpus N] [--group-size K] [--total-steps N] [--model PATH]"
      echo "          [--w-correctness F] [--w-format F] [--w-reasoning F]"
      exit 1
      ;;
  esac
done

# ── Validate ───────────────────────────────────────────────────────────
if [ ! -f "$CONFIG" ]; then
  echo "Error: Config file not found: $CONFIG"
  exit 1
fi

if [ "$N_GPUS" -gt 1 ]; then
  echo "Error: scripts/train_grpo.py is currently a single-process trainer."
  echo "Use --n-gpus 1, or implement DDP/FSDP before launching with torchrun."
  exit 1
fi

# ── Environment summary ────────────────────────────────────────────────
echo "============================================"
echo "  GRPO Training — GSM8K Multi-Reward"
echo "============================================"
echo "  Config:       $CONFIG"
echo "  Model:        $MODEL"
echo "  GPUs:         $N_GPUS"
echo "  Group size:   K=$GROUP_SIZE"
echo "  Batch size:   $BATCH_SIZE prompts/step"
echo "  Total steps:  $TOTAL_STEPS"
echo "  Weights:      c=$W_CORRECTNESS f=$W_FORMAT r=$W_REASONING"
echo "============================================"
echo ""

# ── VRAM estimate ──────────────────────────────────────────────────────
echo "VRAM comparison (Qwen2.5-0.5B, 8x A100 80GB):"
echo "  PPO:   Actor(~2GB) + Critic(~2GB) + Ref(~2GB) + vLLM(~2GB) + Optim(~4GB) = ~12GB per GPU"
echo "  GRPO:  Actor(~2GB) + Ref(~2GB) + Optim(~4GB) = ~8GB per GPU"
echo "  → GRPO saves ~2GB (17%) by removing the critic"
echo "  → Savings increase with model size (50% for 7B models)"
echo ""

# ── Launch ──────────────────────────────────────────────────────────────
GRPO_ARGS=(
  --config "$CONFIG"
  --group-size "$GROUP_SIZE"
  --batch-size "$BATCH_SIZE"
  --total-steps "$TOTAL_STEPS"
  --model "$MODEL"
  --w-correctness "$W_CORRECTNESS"
  --w-format "$W_FORMAT"
  --w-reasoning "$W_REASONING"
)

echo "Launching on single GPU..."
python scripts/train_grpo.py "${GRPO_ARGS[@]}"

echo ""
echo "Training complete!"
