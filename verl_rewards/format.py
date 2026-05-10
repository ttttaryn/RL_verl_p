# Copyright 2024 Custom Implementation
# Format Reward for structured output
#
# Checks output structure INDEPENDENTLY of answer correctness.
# The format reward cares about HOW the model presents its reasoning,
# not whether the final answer is right.
#
# Two format types:
#   'gsm8k'  - rewards step-by-step structure with or without #### marker
#   'xml'     - rewards <reasoning>/<answer> XML tag structure
#
# Both are independent of correctness: a wrong answer in perfect format
# still gets the format reward, and a correct answer in messy format
# gets nothing. This avoids the double-counting bug where correctness
# and format were coupled through the same #### regex.

import re
from typing import Optional

from .reasoning import count_reasoning_steps


def check_xml_structure(text: str) -> dict:
    """Check if the text follows XML-like structure.

    Expected: <reasoning>step-by-step</reasoning><answer>42</answer>
    """
    result = {
        "has_reasoning_tag": False,
        "has_answer_tag": False,
        "reasoning_content": "",
        "answer_content": "",
        "is_valid_structure": False,
    }

    reasoning_pattern = r"<reasoning>(.*?)</reasoning>"
    reasoning_match = re.search(reasoning_pattern, text, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        result["has_reasoning_tag"] = True
        result["reasoning_content"] = reasoning_match.group(1).strip()

    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        result["has_answer_tag"] = True
        result["answer_content"] = answer_match.group(1).strip()

    result["is_valid_structure"] = result["has_reasoning_tag"] and result["has_answer_tag"]

    return result


def check_gsm8k_structure(text: str) -> dict:
    """Check for step-by-step reasoning structure.

    Rewards the PRESENCE of structured reasoning, not just the #### marker.
    This is independent of correctness — the correctness module extracts
    the answer separately.

    Structure indicators:
    - Explicit step markers (Step 1, 1., first/then/finally)
    - Calculation notation (a = ..., let x = ..., equations)
    - Final answer marker (####, answer:, therefore, boxed)
    """
    result = {
        "has_step_markers": False,
        "has_calculations": False,
        "has_answer_marker": False,
        "structure_score": 0.0,
    }

    # Reuse reasoning.py's step detection (single source of truth)
    step_info = count_reasoning_steps(text)
    result["has_step_markers"] = step_info["total_estimated_steps"] >= 1
    result["step_marker_count"] = step_info["total_estimated_steps"]

    # Calculation notation: equations, variables, arithmetic
    calc_patterns = [
        r"[-]?\d+\.?\d*\s*[\+\-\*\/\×\÷]\s*[-]?\d+",  # arithmetic
        r"=\s*[-]?\d+",                                    # equals result
        r"\\\(.*?\\\)",                                    # LaTeX inline
        r"\\\[.*?\\\]",                                    # LaTeX display
    ]
    calc_count = 0
    for pat in calc_patterns:
        calc_count += len(re.findall(pat, text))
    result["has_calculations"] = calc_count >= 1
    result["calculation_count"] = min(calc_count, 20)

    # Answer marker (optional — correctness handles extraction independently)
    answer_patterns = [
        r"####\s*[-]?\d+",         # GSM8K traditional
        r"\\boxed\{",               # LaTeX boxed
        r"[Aa]nswer\s*(?::|is)\s*", # Natural language
        r"[Tt]herefore,?\s*[-]?\d+", # Therefore, X
    ]
    ans_count = 0
    for pat in answer_patterns:
        ans_count += len(re.findall(pat, text))
    result["has_answer_marker"] = ans_count >= 1

    # Composite structure score (0.0 - 1.0)
    score = 0.0
    if result["has_step_markers"]:
        score += 0.4
    if result["has_calculations"]:
        score += 0.4
    if result["has_answer_marker"]:
        score += 0.2

    result["structure_score"] = min(score, 1.0)

    return result


def compute_format_reward(
    solution_str: str,
    format_type: str = "gsm8k",
    full_format_score: float = 1.0,
    partial_format_score: float = 0.5,
    no_format_score: float = 0.0,
    continuous: bool = False,
) -> dict:
    """Compute the format reward for a solution.

    GSM8K mode uses structure analysis (steps + calculations + answer marker).
    XML mode uses tag-based structure checking.
    Both are INDEPENDENT of answer correctness.

    Args:
        solution_str: The model's solution text
        format_type: 'gsm8k' (structure-based) or 'xml' (tag-based)
        full_format_score: Score for excellent structure
        partial_format_score: Score for partial structure
        no_format_score: Score for unstructured output
        continuous: When true, return the raw GSM8K structure score instead of
            mapping it into coarse 0/0.5/1 tiers.
    """
    if format_type == "xml":
        format_check = check_xml_structure(solution_str)

        if format_check["is_valid_structure"]:
            score = full_format_score
        elif format_check["has_reasoning_tag"] or format_check["has_answer_tag"]:
            score = partial_format_score
        else:
            score = no_format_score

        return {
            "score": score,
            "format_details": format_check,
            "format_type": "xml",
        }

    elif format_type == "gsm8k":
        structure = check_gsm8k_structure(solution_str)
        struct_score = structure["structure_score"]

        if continuous:
            score = struct_score
        elif struct_score >= 0.8:
            score = full_format_score
        elif struct_score >= 0.4:
            score = partial_format_score
        else:
            score = no_format_score

        return {
            "score": score,
            "format_details": structure,
            "format_type": "gsm8k",
        }

    else:
        raise ValueError(f"Unknown format type: {format_type}")


def compute_combined_format_reward(
    solution_str: str,
    xml_weight: float = 0.5,
    gsm8k_weight: float = 0.5,
) -> dict:
    """Combine XML and GSM8K format scores.

    Args:
        solution_str: The model's solution text
        xml_weight: Weight for XML format score
        gsm8k_weight: Weight for GSM8K format score
    """
    xml_result = compute_format_reward(solution_str, format_type="xml")
    gsm8k_result = compute_format_reward(solution_str, format_type="gsm8k")

    combined_score = (
        xml_weight * xml_result["score"] +
        gsm8k_weight * gsm8k_result["score"]
    )

    return {
        "score": combined_score,
        "xml_score": xml_result["score"],
        "gsm8k_score": gsm8k_result["score"],
        "xml_details": xml_result["format_details"],
        "gsm8k_details": gsm8k_result["format_details"],
    }
