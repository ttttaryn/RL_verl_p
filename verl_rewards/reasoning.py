# Copyright 2024 Custom Implementation
# Structure & Reasoning Rewards
#
# This module provides TWO types of rewards:
#   1. structure_reward   - measures output organization (length + step count heuristics)
#   2. answer_conditioned_reasoning_reward - only rewards structured outputs that are CORRECT
#
# The original "reasoning_reward" name was misleading: it measured surface-level
# output structure (step count + length), not logical reasoning quality.
# A model could output "Step 1: foo. Step 2: bar. Step 3: 2+2=5." and get a
# near-perfect score. Baseline Qwen2.5-0.5B scored 0.87 reasoning while 6% accuracy,
# confirming the heuristic was orthogonal to math correctness.
#
# The answer_conditioned variant fixes this by gating structure rewards on correctness:
# only CORRECT answers earn structure bonuses. This prevents the model from gaming
# the reward by producing well-formatted but wrong reasoning.

import re
from typing import Optional


def count_reasoning_steps(text: str) -> dict:
    """Count reasoning step indicators in text.

    Looks for: "Step N:", numbered lists, and transition words.
    These are SURFACE features — they don't guarantee valid reasoning.
    """
    result = {
        "explicit_steps": 0,
        "numbered_steps": 0,
        "transition_words": 0,
        "total_estimated_steps": 0,
    }

    explicit_pattern = r"[Ss]tep\s*(\d+)"
    explicit_matches = re.findall(explicit_pattern, text)
    result["explicit_steps"] = len(set(explicit_matches))

    numbered_pattern = r"(?:^|\n)\s*(\d+)\."
    numbered_matches = re.findall(numbered_pattern, text)
    result["numbered_steps"] = len(set(numbered_matches))

    transition_words = [
        r"\bfirst\b", r"\bsecond\b", r"\bthird\b", r"\bfourth\b", r"\bfifth\b",
        r"\bthen\b", r"\bnext\b", r"\bfinally\b", r"\btherefore\b", r"\bthus\b",
        r"\bso\b", r"\bhence\b", r"\blet's\b", r"\blet us\b", r"\bnow\b",
        r"\bwe have\b", r"\bwe get\b", r"\bwe can\b", r"\bthis means\b",
    ]
    transition_count = 0
    for pattern in transition_words:
        transition_count += len(re.findall(pattern, text.lower()))
    result["transition_words"] = min(transition_count, 10)

    result["total_estimated_steps"] = max(
        result["explicit_steps"],
        result["numbered_steps"],
        result["transition_words"] // 2,
        1 if len(text) > 50 else 0,
    )

    return result


def compute_reasoning_length_reward(
    solution_str: str,
    min_length: int = 50,
    optimal_length: int = 300,
    max_length: int = 1000,
    min_score: float = 0.0,
    max_score: float = 1.0,
) -> dict:
    """Compute length-based structure reward.

    Encourages responses in the [min_length, optimal_length] range.
    Penalizes excessively short (<50 chars) or long (>1000 chars) outputs.
    """
    length = len(solution_str)

    if length < min_length:
        score = min_score + (length / min_length) * (max_score * 0.5)
    elif length <= optimal_length:
        progress = (length - min_length) / (optimal_length - min_length)
        score = max_score * 0.5 + progress * (max_score * 0.5)
    elif length <= max_length:
        score = max_score
    else:
        excess_ratio = (length - max_length) / max_length
        score = max(min_score, max_score - excess_ratio * 0.5)

    return {
        "score": score,
        "length": length,
        "length_category": (
            "too_short" if length < min_length
            else "optimal" if length <= optimal_length
            else "acceptable" if length <= max_length
            else "too_long"
        ),
    }


def compute_step_count_reward(
    solution_str: str,
    min_steps: int = 1,
    optimal_steps: int = 3,
    max_steps: int = 10,
    min_score: float = 0.0,
    max_score: float = 1.0,
) -> dict:
    """Compute step-count based structure reward.

    Rewards outputs with a reasonable number of explicit reasoning steps.
    """
    step_info = count_reasoning_steps(solution_str)
    num_steps = step_info["total_estimated_steps"]

    if num_steps < min_steps:
        score = min_score
    elif num_steps <= optimal_steps:
        progress = (num_steps - min_steps) / (optimal_steps - min_steps) if optimal_steps > min_steps else 1.0
        score = min_score + progress * (max_score - min_score)
    elif num_steps <= max_steps:
        score = max_score
    else:
        excess_ratio = (num_steps - max_steps) / max_steps
        score = max(min_score, max_score - excess_ratio * 0.3)

    return {
        "score": score,
        "step_info": step_info,
        "num_steps": num_steps,
    }


def compute_reasoning_reward(
    solution_str: str,
    length_weight: float = 0.4,
    step_weight: float = 0.6,
    min_length: int = 50,
    optimal_length: int = 300,
    max_length: int = 1000,
    min_steps: int = 1,
    optimal_steps: int = 3,
    max_steps: int = 10,
) -> dict:
    """[DEPRECATED] Legacy reasoning reward — purely heuristic structure scoring.

    This function is kept for backward compatibility but its name is misleading.
    It measures output STRUCTURE (length + step count heuristics), not reasoning
    quality. New code should use compute_answer_conditioned_reasoning_reward()
    which gates structure rewards on answer correctness.

    To migrate: replace compute_reasoning_reward() with
    compute_answer_conditioned_reasoning_reward() and pass is_correct.
    """
    length_result = compute_reasoning_length_reward(
        solution_str,
        min_length=min_length,
        optimal_length=optimal_length,
        max_length=max_length,
    )

    step_result = compute_step_count_reward(
        solution_str,
        min_steps=min_steps,
        optimal_steps=optimal_steps,
        max_steps=max_steps,
    )

    combined_score = (
        length_weight * length_result["score"] +
        step_weight * step_result["score"]
    )

    return {
        "score": combined_score,
        "length_score": length_result["score"],
        "step_score": step_result["score"],
        "length": length_result["length"],
        "length_category": length_result["length_category"],
        "num_steps": step_result["num_steps"],
        "step_info": step_result["step_info"],
    }


def compute_answer_conditioned_reasoning_reward(
    solution_str: str,
    is_correct: bool,
    length_weight: float = 0.4,
    step_weight: float = 0.6,
    min_length: int = 50,
    optimal_length: int = 300,
    max_length: int = 1000,
    min_steps: int = 1,
    optimal_steps: int = 3,
    max_steps: int = 10,
) -> dict:
    """Compute structure reward GATED on answer correctness.

    Delegates to compute_reasoning_reward() for the raw structure score,
    then gates on correctness. Only correct answers earn structure bonuses.

    This is the RECOMMENDED replacement for compute_reasoning_reward().
    """
    base_result = compute_reasoning_reward(
        solution_str,
        length_weight=length_weight,
        step_weight=step_weight,
        min_length=min_length,
        optimal_length=optimal_length,
        max_length=max_length,
        min_steps=min_steps,
        optimal_steps=optimal_steps,
        max_steps=max_steps,
    )

    gated_score = base_result["score"] if is_correct else 0.0

    return {
        "score": gated_score,
        "raw_structure_score": base_result["score"],
        "is_correct": is_correct,
        "length_score": base_result["length_score"],
        "step_score": base_result["step_score"],
        "length": base_result["length"],
        "length_category": base_result["length_category"],
        "num_steps": base_result["num_steps"],
        "step_info": base_result["step_info"],
    }
