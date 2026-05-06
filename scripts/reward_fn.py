# Copyright 2024 Custom Implementation
# Custom Reward Function for verl Integration
#
# This file provides the compute_score function that integrates with verl's
# reward manager system. Place this file path in the config:
# custom_reward_function.path = "path/to/this/reward_fn.py"
# custom_reward_function.name = "compute_score"

import os
import sys

# Add the parent directory to path so we can import verl_rewards
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl_rewards import compute_composite_score


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs,
) -> dict:
    """Compute the reward score for a solution.
    
    This function is the entry point for verl's reward manager.
    It uses the composite reward system to compute a weighted combination
    of correctness, format, and reasoning rewards.
    
    Args:
        data_source: The dataset source (e.g., "openai/gsm8k")
        solution_str: The model's generated solution
        ground_truth: The correct answer
        extra_info: Optional extra information from the dataset
        **kwargs: Additional keyword arguments from config
        
    Returns:
        Dictionary containing:
            - score: The final reward score
            - correctness_score: Score for answer correctness
            - format_score: Score for output format
            - reasoning_score: Score for reasoning quality
            - is_correct: Boolean indicating if answer is correct
            - Additional detailed metrics
    """
    # Get reward weights from kwargs (can be set in config)
    # Default: emphasize correctness (0.7), with format (0.1) and reasoning (0.2)
    w_correctness = float(kwargs.get("w_correctness", 0.7))
    w_format = float(kwargs.get("w_format", 0.1))
    w_reasoning = float(kwargs.get("w_reasoning", 0.2))
    
    # Get correctness config
    correctness_method = kwargs.get("correctness_method", "strict")
    
    # Get format config
    format_type = kwargs.get("format_type", "gsm8k")
    
    # Get reasoning config
    min_reasoning_length = int(kwargs.get("min_reasoning_length", 50))
    optimal_reasoning_length = int(kwargs.get("optimal_reasoning_length", 300))
    max_reasoning_length = int(kwargs.get("max_reasoning_length", 1000))
    min_reasoning_steps = int(kwargs.get("min_reasoning_steps", 1))
    optimal_reasoning_steps = int(kwargs.get("optimal_reasoning_steps", 3))
    max_reasoning_steps = int(kwargs.get("max_reasoning_steps", 10))
    
    result = compute_composite_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        w_correctness=w_correctness,
        w_format=w_format,
        w_reasoning=w_reasoning,
        correctness_method=correctness_method,
        format_type=format_type,
        min_reasoning_length=min_reasoning_length,
        optimal_reasoning_length=optimal_reasoning_length,
        max_reasoning_length=max_reasoning_length,
        min_reasoning_steps=min_reasoning_steps,
        optimal_reasoning_steps=optimal_reasoning_steps,
        max_reasoning_steps=max_reasoning_steps,
    )
    
    return result


# Alternative reward functions for different experiments

def compute_score_correctness_only(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs,
) -> dict:
    """Compute reward based on correctness only (baseline).
    
    This mimics the original GSM8K reward function.
    """
    return compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        w_correctness=1.0,
        w_format=0.0,
        w_reasoning=0.0,
        **kwargs,
    )


def compute_score_with_reasoning(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs,
) -> dict:
    """Compute reward with emphasis on reasoning quality.
    
    Good for encouraging step-by-step reasoning.
    """
    return compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        w_correctness=0.6,
        w_format=0.1,
        w_reasoning=0.3,
        **kwargs,
    )


def compute_score_balanced(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs,
) -> dict:
    """Compute reward with balanced weights.
    
    Balances correctness, format, and reasoning equally.
    """
    return compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        w_correctness=0.5,
        w_format=0.25,
        w_reasoning=0.25,
        **kwargs,
    )
