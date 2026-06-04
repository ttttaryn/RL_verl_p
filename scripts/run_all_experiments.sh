#!/bin/bash
# Run all predefined GRPO reward ablations.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$SCRIPT_DIR/run_ablation_experiments.sh" "$@"
