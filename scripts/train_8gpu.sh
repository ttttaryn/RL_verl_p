#!/bin/bash
# Backward-compatible entrypoint. The recommended A100 launcher is train_4gpu.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/train_4gpu.sh" --gpus 8 --batch-size 1024 "$@"
