# Copyright 2024 Custom Implementation
# Reasoning Reward for encouraging structured reasoning
#
# This module computes rewards based on reasoning quality and structure.

import re
from typing import Optional


def count_reasoning_steps(text: str) -> dict:
    """Count the number of reasoning steps in the text.
    
    Looks for patterns like:
    - "Step 1:", "Step 2:", etc.
    - "1.", "2.", "3.", etc.
    - "First,", "Second,", "Third,", etc.
    - "Let's", "Then", "So", "Therefore" transitions
    
    Args:
        text: The model's output text
        
    Returns:
        Dictionary containing step count information
    """
    result = {
        "explicit_steps": 0,
        "numbered_steps": 0,
        "transition_words": 0,
        "total_estimated_steps": 0,
    }
    
    # Count explicit "Step N:" patterns
    explicit_pattern = r"[Ss]tep\s*(\d+)"
    explicit_matches = re.findall(explicit_pattern, text)
    result["explicit_steps"] = len(set(explicit_matches))
    
    # Count numbered steps (1., 2., etc. at start of line or after newline)
    numbered_pattern = r"(?:^|\n)\s*(\d+)\."
    numbered_matches = re.findall(numbered_pattern, text)
    result["numbered_steps"] = len(set(numbered_matches))
    
    # Count transition words that indicate reasoning steps
    transition_words = [
        r"\bfirst\b", r"\bsecond\b", r"\bthird\b", r"\bfourth\b", r"\bfifth\b",
        r"\bthen\b", r"\bnext\b", r"\bfinally\b", r"\btherefore\b", r"\bthus\b",
        r"\bso\b", r"\bhence\b", r"\blet's\b", r"\blet us\b", r"\bnow\b",
        r"\bwe have\b", r"\bwe get\b", r"\bwe can\b", r"\bthis means\b",
    ]
    transition_count = 0
    for pattern in transition_words:
        transition_count += len(re.findall(pattern, text.lower()))
    result["transition_words"] = min(transition_count, 10)  # Cap at 10
    
    # Estimate total steps
    result["total_estimated_steps"] = max(
        result["explicit_steps"],
        result["numbered_steps"],
        result["transition_words"] // 2,  # Rough estimate
        1 if len(text) > 50 else 0  # At least 1 if there's substantial text
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
    """Compute reward based on reasoning length.
    
    Encourages responses that are not too short (lacking reasoning)
    or too long (verbose/repetitive).
    
    Args:
        solution_str: The model's solution text
        min_length: Minimum acceptable length
        optimal_length: Optimal response length
        max_length: Maximum acceptable length (penalty beyond this)
        min_score: Minimum score for too short/long responses
        max_score: Maximum score for optimal length
        
    Returns:
        Dictionary containing length-based reward information
    """
    length = len(solution_str)
    
    if length < min_length:
        # Too short - linear scale from 0 to partial score
        score = min_score + (length / min_length) * (max_score * 0.5)
    elif length <= optimal_length:
        # Good range - scale up to max score
        progress = (length - min_length) / (optimal_length - min_length)
        score = max_score * 0.5 + progress * (max_score * 0.5)
    elif length <= max_length:
        # Acceptable but longer than optimal
        score = max_score
    else:
        # Too long - gradual penalty
        excess_ratio = (length - max_length) / max_length
        score = max(min_score, max_score - excess_ratio * 0.5)
    
    return {
        "score": score,
        "length": length,
        "length_category": (
            "too_short" if length < min_length else
            "optimal" if length <= optimal_length else
            "acceptable" if length <= max_length else
            "too_long"
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
    """Compute reward based on number of reasoning steps.
    
    Args:
        solution_str: The model's solution text
        min_steps: Minimum acceptable steps
        optimal_steps: Optimal number of steps
        max_steps: Maximum acceptable steps
        min_score: Minimum score
        max_score: Maximum score for optimal step count
        
    Returns:
        Dictionary containing step-based reward information
    """
    step_info = count_reasoning_steps(solution_str)
    num_steps = step_info["total_estimated_steps"]
    
    if num_steps < min_steps:
        score = min_score
    elif num_steps <= optimal_steps:
        # Scale linearly from min to max
        progress = (num_steps - min_steps) / (optimal_steps - min_steps) if optimal_steps > min_steps else 1.0
        score = min_score + progress * (max_score - min_score)
    elif num_steps <= max_steps:
        # Still good, keep max score
        score = max_score
    else:
        # Too many steps - might be repetitive
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
    """Compute combined reasoning reward.
    
    Combines length-based and step-based rewards.
    
    Args:
        solution_str: The model's solution text
        length_weight: Weight for length-based reward
        step_weight: Weight for step-based reward
        Other args: Parameters for individual reward components
        
    Returns:
        Dictionary containing combined reasoning reward information
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
