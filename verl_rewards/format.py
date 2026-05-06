# Copyright 2024 Custom Implementation
# Format Reward for structured output
#
# This module computes rewards based on output format compliance.

import re
from typing import Optional


def check_xml_structure(text: str) -> dict:
    """Check if the text follows the expected XML-like structure.
    
    Expected structure:
    <reasoning>
    step 1 ...
    step 2 ...
    </reasoning>
    <answer>
    42
    </answer>
    
    Args:
        text: The model's output text
        
    Returns:
        Dictionary containing format check results
    """
    result = {
        "has_reasoning_tag": False,
        "has_answer_tag": False,
        "reasoning_content": "",
        "answer_content": "",
        "is_valid_structure": False,
    }
    
    # Check for reasoning tags
    reasoning_pattern = r"<reasoning>(.*?)</reasoning>"
    reasoning_match = re.search(reasoning_pattern, text, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        result["has_reasoning_tag"] = True
        result["reasoning_content"] = reasoning_match.group(1).strip()
    
    # Check for answer tags
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        result["has_answer_tag"] = True
        result["answer_content"] = answer_match.group(1).strip()
    
    # Valid structure requires both tags
    result["is_valid_structure"] = result["has_reasoning_tag"] and result["has_answer_tag"]
    
    return result


def check_gsm8k_format(text: str) -> dict:
    """Check if the text follows the GSM8K format (#### answer).
    
    Args:
        text: The model's output text
        
    Returns:
        Dictionary containing format check results
    """
    result = {
        "has_hash_format": False,
        "answer_after_hash": "",
    }
    
    # Check for #### format
    hash_pattern = r"####\s*([\-]?[0-9\.\,]+)"
    hash_match = re.search(hash_pattern, text)
    if hash_match:
        result["has_hash_format"] = True
        result["answer_after_hash"] = hash_match.group(1).strip()
    
    return result


def compute_format_reward(
    solution_str: str,
    format_type: str = "xml",
    full_format_score: float = 1.0,
    partial_format_score: float = 0.5,
    no_format_score: float = 0.0,
) -> dict:
    """Compute the format reward for a solution.
    
    Args:
        solution_str: The model's solution text
        format_type: Type of format to check ('xml' or 'gsm8k')
        full_format_score: Score for fully correct format
        partial_format_score: Score for partially correct format
        no_format_score: Score for no format compliance
        
    Returns:
        Dictionary containing:
            - score: The format score
            - format_details: Detailed format check results
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
        format_check = check_gsm8k_format(solution_str)
        
        if format_check["has_hash_format"]:
            score = full_format_score
        else:
            score = no_format_score
        
        return {
            "score": score,
            "format_details": format_check,
            "format_type": "gsm8k",
        }
    
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def compute_combined_format_reward(
    solution_str: str,
    xml_weight: float = 0.5,
    gsm8k_weight: float = 0.5,
) -> dict:
    """Compute a combined format reward considering multiple formats.
    
    Args:
        solution_str: The model's solution text
        xml_weight: Weight for XML format score
        gsm8k_weight: Weight for GSM8K format score
        
    Returns:
        Dictionary containing combined format information
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
