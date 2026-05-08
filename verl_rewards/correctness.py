# Copyright 2024 Custom Implementation
# Correctness Reward for GSM8K dataset
#
# This module extracts the answer from model output and compares it to ground truth.
#
# Key design choice: "flexible" extraction does NOT require #### format, making
# the correctness reward independent from the format reward. This eliminates the
# double-counting bug where strict mode + gsm8k format used the same regex.

import re
from typing import Optional

_SOLUTION_CLIP_CHARS = 300


def extract_answer(solution_str: str, method: str = "flexible") -> Optional[str]:
    """Extract the answer from the solution string.

    Args:
        solution_str: The solution text containing the answer
        method:
            'flexible' (DEFAULT) - finds the last valid number in text.
                Independent of output format. Recommended for use with
                separate format rewards.
            'strict' - requires #### NUMBER pattern (GSM8K convention).
                Note: this couples correctness to format — a correct answer
                without #### scores 0. Only use when format reward is disabled.

    Returns:
        The extracted answer string or None if not found
    """
    assert method in ["strict", "flexible"]

    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        solutions = re.findall(r"####\s*([\-]?[0-9]+\.?[0-9]*)", solution_str)
        if len(solutions) == 0:
            return None
        return solutions[-1].replace(",", "").replace("$", "").strip()

    elif method == "flexible":
        # Find numbers: require at least one digit (fixes bare-period bug)
        numbers = re.findall(r"([\-]?\d+\.?\d*)", solution_str)
        if len(numbers) == 0:
            return None
        invalid_str = {"", "."}
        for answer in reversed(numbers):
            if answer not in invalid_str:
                return answer.replace(",", "").strip()
        return None


def compute_correctness_reward(
    solution_str: str,
    ground_truth: str,
    method: str = "flexible",
    correct_score: float = 1.0,
    incorrect_score: float = 0.0,
) -> dict:
    """Compute the correctness reward for a solution.

    Uses numeric comparison with tolerance. Falls back to string comparison.

    Args:
        solution_str: The model's solution text
        ground_truth: The correct answer
        method: 'flexible' (default, format-independent) or 'strict'
        correct_score: Score for correct answer
        incorrect_score: Score for incorrect answer

    Returns:
        Dictionary containing:
            - score: The correctness score
            - extracted_answer: The extracted answer
            - is_correct: Boolean
            - extraction_method: Which method was used
    """
    extracted_answer = extract_answer(solution_str, method=method)

    if extracted_answer is None:
        return {
            "score": 0.0,
            "extracted_answer": None,
            "is_correct": False,
            "extraction_method": method,
        }

    # Normalize ground truth: strip $, commas, whitespace
    gt_clean = ground_truth.replace(",", "").replace("$", "").strip()

    try:
        extracted_num = float(extracted_answer)
        ground_truth_num = float(gt_clean)
        is_correct = abs(extracted_num - ground_truth_num) < 1e-6
    except (ValueError, TypeError):
        is_correct = extracted_answer.strip() == gt_clean

    return {
        "score": correct_score if is_correct else incorrect_score,
        "extracted_answer": extracted_answer,
        "is_correct": is_correct,
        "extraction_method": method,
    }
