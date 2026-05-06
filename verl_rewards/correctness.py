# Copyright 2024 Custom Implementation
# Correctness Reward for GSM8K dataset
#
# This module computes the correctness reward based on whether the answer is correct.

import re
from typing import Optional

_SOLUTION_CLIP_CHARS = 300


def extract_answer(solution_str: str, method: str = "strict") -> Optional[str]:
    """Extract the answer from the solution string.
    
    Args:
        solution_str: The solution text containing the answer
        method: Extraction method - 'strict' requires #### format, 'flexible' finds last number
        
    Returns:
        The extracted answer string or None if not found
    """
    assert method in ["strict", "flexible"]
    
    # Optimization: Only look at the last part of the string
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]
    
    if method == "strict":
        # Match the #### format used in GSM8K
        solutions = re.findall(r"####\s*([\-]?[0-9\.\,]+)", solution_str)
        if len(solutions) == 0:
            return None
        # Take the last solution
        return solutions[-1].replace(",", "").replace("$", "").strip()
    
    elif method == "flexible":
        # Find any number in the text
        numbers = re.findall(r"([\-]?[0-9\.\,]+)", solution_str)
        if len(numbers) == 0:
            return None
        # Find the last valid number
        invalid_str = ["", "."]
        for answer in reversed(numbers):
            if answer not in invalid_str:
                return answer.replace(",", "").strip()
        return None


def compute_correctness_reward(
    solution_str: str,
    ground_truth: str,
    method: str = "strict",
    correct_score: float = 1.0,
    incorrect_score: float = 0.0,
) -> dict:
    """Compute the correctness reward for a solution.
    
    Args:
        solution_str: The model's solution text
        ground_truth: The correct answer
        method: Answer extraction method ('strict' or 'flexible')
        correct_score: Score for correct answer
        incorrect_score: Score for incorrect answer
        
    Returns:
        Dictionary containing:
            - score: The correctness score
            - extracted_answer: The extracted answer from the solution
            - is_correct: Boolean indicating if the answer is correct
    """
    extracted_answer = extract_answer(solution_str, method=method)
    
    if extracted_answer is None:
        return {
            "score": 0.0,
            "extracted_answer": None,
            "is_correct": False,
        }
    
    # Normalize both answers for comparison
    try:
        # Try to compare as numbers
        extracted_num = float(extracted_answer)
        ground_truth_num = float(ground_truth.replace(",", "").strip())
        is_correct = abs(extracted_num - ground_truth_num) < 1e-6
    except (ValueError, TypeError):
        # Fall back to string comparison
        is_correct = extracted_answer.strip() == ground_truth.strip()
    
    return {
        "score": correct_score if is_correct else incorrect_score,
        "extracted_answer": extracted_answer,
        "is_correct": is_correct,
    }
