# Copyright 2024 Custom Implementation
# Composite Reward System for verl RL Training
#
# Combines correctness, format, and reasoning rewards into a single signal.
# Supports answer-conditioned reasoning (recommended) and legacy heuristic mode.

from typing import Any, Callable, Optional

from .correctness import compute_correctness_reward
from .format import compute_format_reward
from .reasoning import compute_reasoning_reward, compute_answer_conditioned_reasoning_reward


class CompositeReward:
    """Composite reward manager combining multiple reward signals.

    Two modes:
    1. answer_conditioned=True (RECOMMENDED): Structure reward is gated on
       correctness — only correct answers earn reasoning bonuses. This prevents
       the model from gaming structure rewards with well-formatted but wrong
       outputs.
    2. answer_conditioned=False (legacy): Independent heuristic scoring of
       output structure regardless of correctness. The model can earn reasoning
       reward for formatting alone (baseline: 0.87 score at 6% accuracy).

    Example:
        >>> composite = CompositeReward(
        ...     weights={"correctness": 0.7, "format": 0.1, "reasoning": 0.2},
        ...     answer_conditioned_reasoning=True,
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
        answer_conditioned_reasoning: bool = True,
    ):
        self.weights = weights or {
            "correctness": 0.7,
            "format": 0.1,
            "reasoning": 0.2,
        }

        self.correctness_config = correctness_config or {
            "method": "flexible",  # flexible = format-independent extraction
            "correct_score": 1.0,
            "incorrect_score": 0.0,
        }

        self.format_config = format_config or {
            "format_type": "gsm8k",
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
        self.answer_conditioned_reasoning = answer_conditioned_reasoning

    def get_current_weights(self, step: Optional[int] = None) -> dict:
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

        Returns:
            Dictionary with score, correctness, format, reasoning details.
        """
        weights = self.get_current_weights(step)

        correctness_result = compute_correctness_reward(
            solution_str=solution_str,
            ground_truth=ground_truth,
            **self.correctness_config,
        )

        format_result = compute_format_reward(
            solution_str=solution_str,
            **self.format_config,
        )

        # Use answer-conditioned reasoning when enabled (recommended)
        if self.answer_conditioned_reasoning:
            reasoning_result = compute_answer_conditioned_reasoning_reward(
                solution_str=solution_str,
                is_correct=correctness_result["is_correct"],
                **self.reasoning_config,
            )
        else:
            reasoning_result = compute_reasoning_reward(
                solution_str=solution_str,
                **self.reasoning_config,
            )

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
            "answer_conditioned_reasoning": self.answer_conditioned_reasoning,
        }


def create_weight_scheduler(
    initial_weights: dict,
    final_weights: dict,
    warmup_steps: int = 100,
    total_steps: int = 1000,
) -> Callable:
    """Create a linear weight scheduler.

    Args:
        initial_weights: Starting weights
        final_weights: Ending weights
        warmup_steps: Steps before interpolation starts
        total_steps: Total training steps
    """
    def scheduler(step: int) -> dict:
        if step < warmup_steps:
            return initial_weights
        if step >= total_steps:
            return final_weights
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return {
            key: initial_weights[key] + progress * (final_weights[key] - initial_weights[key])
            for key in initial_weights.keys()
        }
    return scheduler


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
    correctness_method: str = "flexible",
    # Format config
    format_type: str = "gsm8k",
    # Reasoning config
    answer_conditioned_reasoning: bool = True,
    min_reasoning_length: int = 50,
    optimal_reasoning_length: int = 300,
    max_reasoning_length: int = 1000,
    min_reasoning_steps: int = 1,
    optimal_reasoning_steps: int = 3,
    max_reasoning_steps: int = 10,
    **kwargs,
) -> dict:
    """Compute composite score for verl reward manager integration.

    Follows the verl compute_score interface.

    Key defaults (changed from original):
    - correctness_method="flexible": no longer requires #### format
    - answer_conditioned_reasoning=True: structure reward gated on correctness
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
        answer_conditioned_reasoning=answer_conditioned_reasoning,
    )

    result = composite(
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )

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
        "answer_conditioned_reasoning": result["answer_conditioned_reasoning"],
    }
