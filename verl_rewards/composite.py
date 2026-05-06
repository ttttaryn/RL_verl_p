# Copyright 2024 Custom Implementation
# Composite Reward System for verl PPO Training
#
# This module combines multiple reward components into a single reward signal.

from typing import Any, Callable, Optional

from .correctness import compute_correctness_reward
from .format import compute_format_reward
from .reasoning import compute_reasoning_reward


class CompositeReward:
    """Composite reward manager that combines multiple reward signals.
    
    This class provides a flexible way to combine correctness, format, and
    reasoning rewards with configurable weights that can change during training.
    
    Example:
        >>> composite = CompositeReward(
        ...     weights={"correctness": 0.7, "format": 0.1, "reasoning": 0.2}
        ... )
        >>> result = composite(solution_str="...", ground_truth="42")
    """
    
    def __init__(
        self,
        weights: Optional[dict] = None,
        correctness_config: Optional[dict] = None,
        format_config: Optional[dict] = None,
        reasoning_config: Optional[dict] = None,
        weight_scheduler: Optional[Callable] = None,
    ):
        """Initialize the composite reward manager.
        
        Args:
            weights: Dictionary of weights for each reward component
                     Keys: "correctness", "format", "reasoning"
            correctness_config: Configuration for correctness reward
            format_config: Configuration for format reward
            reasoning_config: Configuration for reasoning reward
            weight_scheduler: Optional function(step) -> weights that adjusts
                            weights based on training step
        """
        self.weights = weights or {
            "correctness": 0.7,
            "format": 0.1,
            "reasoning": 0.2,
        }
        
        self.correctness_config = correctness_config or {
            "method": "strict",
            "correct_score": 1.0,
            "incorrect_score": 0.0,
        }
        
        self.format_config = format_config or {
            "format_type": "gsm8k",  # or "xml" for structured output
            "full_format_score": 1.0,
            "partial_format_score": 0.5,
            "no_format_score": 0.0,
        }
        
        self.reasoning_config = reasoning_config or {
            "length_weight": 0.4,
            "step_weight": 0.6,
            "min_length": 50,
            "optimal_length": 300,
            "max_length": 1000,
            "min_steps": 1,
            "optimal_steps": 3,
            "max_steps": 10,
        }
        
        self.weight_scheduler = weight_scheduler
        self.current_step = 0
    
    def get_current_weights(self, step: Optional[int] = None) -> dict:
        """Get the current weights, optionally adjusted by scheduler.
        
        Args:
            step: Optional training step for weight scheduling
            
        Returns:
            Dictionary of current weights
        """
        if step is not None:
            self.current_step = step
        
        if self.weight_scheduler is not None:
            return self.weight_scheduler(self.current_step)
        
        return self.weights
    
    def __call__(
        self,
        solution_str: str,
        ground_truth: str,
        step: Optional[int] = None,
        extra_info: Optional[dict] = None,
    ) -> dict:
        """Compute the composite reward.
        
        Args:
            solution_str: The model's solution text
            ground_truth: The correct answer
            step: Optional training step for weight scheduling
            extra_info: Optional extra information
            
        Returns:
            Dictionary containing:
                - score: The final composite score
                - correctness: Correctness reward details
                - format: Format reward details
                - reasoning: Reasoning reward details
                - weights: Current weights used
        """
        weights = self.get_current_weights(step)
        
        # Compute individual rewards
        correctness_result = compute_correctness_reward(
            solution_str=solution_str,
            ground_truth=ground_truth,
            **self.correctness_config,
        )
        
        format_result = compute_format_reward(
            solution_str=solution_str,
            **self.format_config,
        )
        
        reasoning_result = compute_reasoning_reward(
            solution_str=solution_str,
            **self.reasoning_config,
        )
        
        # Compute weighted sum
        composite_score = (
            weights["correctness"] * correctness_result["score"] +
            weights["format"] * format_result["score"] +
            weights["reasoning"] * reasoning_result["score"]
        )
        
        return {
            "score": composite_score,
            "correctness": correctness_result,
            "format": format_result,
            "reasoning": reasoning_result,
            "weights": weights,
            "is_correct": correctness_result["is_correct"],
        }


def create_weight_scheduler(
    initial_weights: dict,
    final_weights: dict,
    warmup_steps: int = 100,
    total_steps: int = 1000,
) -> Callable:
    """Create a weight scheduler that linearly interpolates weights.
    
    Args:
        initial_weights: Starting weights
        final_weights: Ending weights
        warmup_steps: Steps before starting interpolation
        total_steps: Total training steps
        
    Returns:
        Callable that returns weights for a given step
    """
    def scheduler(step: int) -> dict:
        if step < warmup_steps:
            return initial_weights
        
        if step >= total_steps:
            return final_weights
        
        # Linear interpolation
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        
        return {
            key: initial_weights[key] + progress * (final_weights[key] - initial_weights[key])
            for key in initial_weights.keys()
        }
    
    return scheduler


# Main entry point for verl integration
def compute_composite_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    # Reward weights
    w_correctness: float = 0.7,
    w_format: float = 0.1,
    w_reasoning: float = 0.2,
    # Correctness config
    correctness_method: str = "strict",
    # Format config
    format_type: str = "gsm8k",
    # Reasoning config
    min_reasoning_length: int = 50,
    optimal_reasoning_length: int = 300,
    max_reasoning_length: int = 1000,
    min_reasoning_steps: int = 1,
    optimal_reasoning_steps: int = 3,
    max_reasoning_steps: int = 10,
    **kwargs,
) -> dict:
    """Compute composite score for verl reward manager integration.
    
    This function follows the verl compute_score interface and can be used
    as a custom reward function.
    
    Args:
        data_source: The dataset source identifier
        solution_str: The model's solution text
        ground_truth: The correct answer
        extra_info: Optional extra information
        w_correctness: Weight for correctness reward
        w_format: Weight for format reward
        w_reasoning: Weight for reasoning reward
        correctness_method: Method for answer extraction ('strict' or 'flexible')
        format_type: Format type to check ('gsm8k' or 'xml')
        Other args: Configuration for reward components
        
    Returns:
        Dictionary containing score and detailed breakdown
    """
    composite = CompositeReward(
        weights={
            "correctness": w_correctness,
            "format": w_format,
            "reasoning": w_reasoning,
        },
        correctness_config={
            "method": correctness_method,
            "correct_score": 1.0,
            "incorrect_score": 0.0,
        },
        format_config={
            "format_type": format_type,
            "full_format_score": 1.0,
            "partial_format_score": 0.5,
            "no_format_score": 0.0,
        },
        reasoning_config={
            "length_weight": 0.4,
            "step_weight": 0.6,
            "min_length": min_reasoning_length,
            "optimal_length": optimal_reasoning_length,
            "max_length": max_reasoning_length,
            "min_steps": min_reasoning_steps,
            "optimal_steps": optimal_reasoning_steps,
            "max_steps": max_reasoning_steps,
        },
    )
    
    result = composite(
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    
    # Return in format compatible with verl reward manager
    # Note: Only return numeric values or strings that won't break metric computation
    # extracted_answer must not be None to avoid metric computation errors
    extracted_answer = result["correctness"]["extracted_answer"]
    if extracted_answer is None:
        extracted_answer = ""
    
    return {
        "score": result["score"],
        "correctness_score": result["correctness"]["score"],
        "format_score": result["format"]["score"],
        "reasoning_score": result["reasoning"]["score"],
        "is_correct": result["is_correct"],
        "extracted_answer": extracted_answer,
        "num_reasoning_steps": result["reasoning"]["num_steps"],
        "response_length": result["reasoning"]["length"],
        # Note: weights dict removed as it causes issues with metric aggregation
    }
