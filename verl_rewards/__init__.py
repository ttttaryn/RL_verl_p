# Copyright 2024 Custom Implementation
# Multi-Reward System for verl RL Training
#
# Provides composable rewards for LLM math reasoning training:
#   - Correctness: answer extraction + ground truth comparison
#   - Format: output structure quality (independent of correctness)
#   - Reasoning: answer-conditioned structure reward (recommended)
#                or legacy heuristic scoring

from .correctness import compute_correctness_reward, extract_answer
from .format import (
    compute_format_reward,
    compute_combined_format_reward,
    check_gsm8k_structure,
    check_xml_structure,
)
from .reasoning import (
    compute_reasoning_reward,  # legacy — heuristic only
    compute_answer_conditioned_reasoning_reward,  # recommended
    count_reasoning_steps,
)
from .composite import CompositeReward, compute_composite_score, create_weight_scheduler

__all__ = [
    # Correctness
    "compute_correctness_reward",
    "extract_answer",
    # Format
    "compute_format_reward",
    "compute_combined_format_reward",
    "check_gsm8k_structure",
    "check_xml_structure",
    # Reasoning
    "compute_reasoning_reward",
    "compute_answer_conditioned_reasoning_reward",
    "count_reasoning_steps",
    # Composite
    "CompositeReward",
    "compute_composite_score",
    "create_weight_scheduler",
]
