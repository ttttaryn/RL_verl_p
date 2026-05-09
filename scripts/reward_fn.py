# Copyright 2024 Custom Implementation
# Custom Reward Functions for verl Integration
#
# Provides compute_score entry points for verl's reward manager.
# All functions follow the verl interface:
#   compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs) -> dict

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl_rewards import compute_composite_score


def _call_compute_score_with_overrides(overrides: dict, **kwargs) -> dict:
    merged_kwargs = dict(kwargs)
    merged_kwargs.update(overrides)
    return compute_score(**merged_kwargs)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs,
) -> dict:
    """Default multi-reward entry point.

    Uses answer-conditioned reasoning (recommended): structure rewards are
    gated on correctness, preventing the model from gaming format bonuses
    with well-structured but wrong reasoning.

    Defaults:
        w_correctness=0.7, w_format=0.1, w_reasoning=0.2
        correctness_method="flexible" (format-independent extraction)
        answer_conditioned_reasoning=True
    """
    w_correctness = float(kwargs.get("w_correctness", 0.7))
    w_format = float(kwargs.get("w_format", 0.1))
    w_reasoning = float(kwargs.get("w_reasoning", 0.2))
    correctness_method = kwargs.get("correctness_method", "flexible")
    format_type = kwargs.get("format_type", "gsm8k")
    answer_conditioned = kwargs.get("answer_conditioned_reasoning", True)

    min_reasoning_length = int(kwargs.get("min_reasoning_length", 50))
    optimal_reasoning_length = int(kwargs.get("optimal_reasoning_length", 300))
    max_reasoning_length = int(kwargs.get("max_reasoning_length", 1000))
    min_reasoning_steps = int(kwargs.get("min_reasoning_steps", 1))
    optimal_reasoning_steps = int(kwargs.get("optimal_reasoning_steps", 3))
    max_reasoning_steps = int(kwargs.get("max_reasoning_steps", 10))

    return compute_composite_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        w_correctness=w_correctness,
        w_format=w_format,
        w_reasoning=w_reasoning,
        correctness_method=correctness_method,
        format_type=format_type,
        answer_conditioned_reasoning=answer_conditioned,
        min_reasoning_length=min_reasoning_length,
        optimal_reasoning_length=optimal_reasoning_length,
        max_reasoning_length=max_reasoning_length,
        min_reasoning_steps=min_reasoning_steps,
        optimal_reasoning_steps=optimal_reasoning_steps,
        max_reasoning_steps=max_reasoning_steps,
    )


# ── Alternative reward functions for experiments ──────────────────────────

def compute_score_correctness_only(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs,
) -> dict:
    """Baseline: correctness reward only (no format, no reasoning)."""
    return _call_compute_score_with_overrides(
        {
            "w_correctness": 1.0,
            "w_format": 0.0,
            "w_reasoning": 0.0,
        },
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )


def compute_score_with_reasoning(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs,
) -> dict:
    """Heavy reasoning weight (0.6/0.1/0.3)."""
    return _call_compute_score_with_overrides(
        {
            "w_correctness": 0.6,
            "w_format": 0.1,
            "w_reasoning": 0.3,
        },
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )


def compute_score_balanced(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs,
) -> dict:
    """Balanced weights across all three components (0.5/0.25/0.25)."""
    return _call_compute_score_with_overrides(
        {
            "w_correctness": 0.5,
            "w_format": 0.25,
            "w_reasoning": 0.25,
        },
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )


def compute_score_legacy_reasoning(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs,
) -> dict:
    """Legacy mode: heuristic reasoning reward NOT gated on correctness.

    This matches the original behavior where reasoning score was purely
    based on output structure (length + step count), independent of
    whether the answer was correct. Only use for comparison experiments.
    """
    return _call_compute_score_with_overrides(
        {
            "answer_conditioned_reasoning": False,
            "correctness_method": "strict",
        },
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
