#!/bin/bash
# Run All Experiments Script
#
# This script runs all predefined experiments sequentially.
# Useful for comparing different reward configurations.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "Running All Multi-Reward Experiments"
echo "=============================================="

# Run each experiment
for exp in 1 2 3 4 5; do
    echo ""
    echo "=============================================="
    echo "Starting Experiment $exp"
    echo "=============================================="
    
    "$SCRIPT_DIR/train_ppo.sh" --exp $exp
    
    echo ""
    echo "Experiment $exp completed!"
    echo ""
    
    # Optional: wait between experiments
    sleep 10
done

echo ""
echo "=============================================="
echo "All Experiments Completed!"
echo "=============================================="
echo ""
echo "Run the following command to generate analysis report:"
echo "  python scripts/analyze_metrics.py report --experiment-dir logs --output-dir reports"
