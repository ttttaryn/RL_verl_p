#!/usr/bin/env python3
# Copyright 2024 Custom Implementation
# Evaluation Script for Trained Models
#
# This script evaluates trained models on the GSM8K test set
# and computes detailed metrics including multi-reward breakdown.

import argparse
import json
import os
import sys
import glob
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent))

from verl_rewards import compute_composite_score


def resolve_torch_dtype(dtype: str):
    if dtype == "auto":
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            return torch.bfloat16
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype}")


def is_verl_checkpoint(model_path: str) -> bool:
    """Check if the path is a verl FSDP checkpoint directory."""
    path = Path(model_path)
    # Check for verl checkpoint indicators
    if path.is_dir():
        # Check for FSDP sharded files
        fsdp_files = list(path.glob("model_world_size_*_rank_*.pt"))
        if fsdp_files:
            return True
        # Check for actor subdirectory
        actor_path = path / "actor"
        if actor_path.exists():
            fsdp_files = list(actor_path.glob("model_world_size_*_rank_*.pt"))
            if fsdp_files:
                return True
    return False


def load_verl_checkpoint(model_path: str, base_model_path: str = None, dtype: str = "auto") -> tuple:
    """Load model from verl FSDP checkpoint.
    
    Args:
        model_path: Path to verl checkpoint (global_step_X or global_step_X/actor)
        base_model_path: Path to the base model for tokenizer and config
        
    Returns:
        Tuple of (model, tokenizer)
    """
    path = Path(model_path)
    
    # Determine the actual actor path
    if (path / "actor").exists():
        actor_path = path / "actor"
    elif path.name == "actor":
        actor_path = path
    else:
        actor_path = path
    
    print(f"Loading verl checkpoint from {actor_path}...")
    
    # Find the huggingface config directory
    hf_path = actor_path / "huggingface"
    
    # Try to infer base model from config
    if base_model_path is None:
        config_path = hf_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                # Try to get original model name
                base_model_path = config.get("_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")
        else:
            base_model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"Using base model: {base_model_path}")
    
    # Load tokenizer from base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=resolve_torch_dtype(dtype),
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load and merge FSDP sharded weights
    fsdp_files = sorted(glob.glob(str(actor_path / "model_world_size_*_rank_*.pt")))
    
    if fsdp_files:
        print(f"Found {len(fsdp_files)} FSDP shard files, merging weights...")
        
        merged_state_dict = {}
        for shard_file in fsdp_files:
            print(f"  Loading {Path(shard_file).name}...")
            shard_state = torch.load(shard_file, map_location="cpu")
            
            # FSDP shards may have different structures depending on verl version
            if isinstance(shard_state, dict):
                for key, value in shard_state.items():
                    if key not in merged_state_dict:
                        merged_state_dict[key] = value
                    else:
                        # For FSDP, same keys should have same values (full replication)
                        # or need concatenation (tensor parallelism) - verl uses full replication
                        pass
        
        # Load merged weights into model
        if merged_state_dict:
            print("Loading merged weights into model...")
            # Handle potential key mismatches
            model_state = model.state_dict()
            
            # Try to match keys
            matched_keys = 0
            for key in merged_state_dict:
                if key in model_state:
                    if merged_state_dict[key].shape == model_state[key].shape:
                        model_state[key] = merged_state_dict[key]
                        matched_keys += 1
            
            if matched_keys > 0:
                model.load_state_dict(model_state, strict=False)
                print(f"Loaded {matched_keys} weight tensors from checkpoint")
            else:
                print("Warning: Could not match checkpoint weights to model. Using base model weights.")
    
    return model, tokenizer


def load_gsm8k_test_data(data_path: str) -> List[Dict]:
    """Load GSM8K test data from parquet file.
    
    Args:
        data_path: Path to the test parquet file
        
    Returns:
        List of test samples
    """
    import pandas as pd
    import numpy as np
    
    df = pd.read_parquet(data_path)
    
    samples = []
    for _, row in df.iterrows():
        # Handle prompt - it may be a numpy array of chat messages or a string
        prompt_data = row.get("prompt", row.get("question", ""))
        
        if isinstance(prompt_data, np.ndarray):
            # Convert numpy array to list
            prompt_data = prompt_data.tolist()
        
        if isinstance(prompt_data, list):
            # Chat format: list of {"role": ..., "content": ...}
            # Extract the user message content
            prompt_text = ""
            for msg in prompt_data:
                if isinstance(msg, dict):
                    if msg.get("role") == "user":
                        prompt_text = msg.get("content", "")
                        break
                    elif "content" in msg:
                        prompt_text = msg.get("content", "")
            if not prompt_text and prompt_data:
                # Fallback: use the last message's content
                last_msg = prompt_data[-1]
                if isinstance(last_msg, dict):
                    prompt_text = last_msg.get("content", str(last_msg))
                else:
                    prompt_text = str(last_msg)
        elif isinstance(prompt_data, str):
            prompt_text = prompt_data
        else:
            prompt_text = str(prompt_data)
        
        # Handle ground_truth
        reward_model = row.get("reward_model", {})
        if isinstance(reward_model, dict):
            ground_truth = str(reward_model.get("ground_truth", ""))
        else:
            ground_truth = str(row.get("answer", ""))
        
        samples.append({
            "prompt": prompt_text,
            "ground_truth": ground_truth,
        })
    
    return samples


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> str:
    """Generate a response from the model.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 for greedy decoding)
        top_p: Top-p sampling parameter
        
    Returns:
        Generated response string
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 当 temperature <= 0 时使用贪婪解码
    if temperature <= 0:
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
    else:
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_kwargs,
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def generate_responses_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> List[str]:
    """Generate responses for a batch of prompts.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of input prompts
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 for greedy decoding)
        top_p: Top-p sampling parameter
        
    Returns:
        List of generated response strings
    """
    # Tokenize all prompts with padding (left padding for decoder-only models)
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=2048,
    ).to(model.device)
    
    # 记录输入序列的总长度（包括padding）
    input_seq_length = inputs["input_ids"].shape[1]
    
    # 当 temperature <= 0 时使用贪婪解码
    if temperature <= 0:
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
    else:
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_kwargs,
        )
    
    # Decode each response, removing the input prompt
    # 由于 generate 返回的是完整序列（input + generated），我们需要跳过输入部分
    responses = []
    for i, output in enumerate(outputs):
        # 生成的新token从input_seq_length位置开始
        response = tokenizer.decode(output[input_seq_length:], skip_special_tokens=True)
        responses.append(response)
    
    return responses


def evaluate_model(
    model_path: str,
    test_data_path: str,
    output_path: Optional[str] = None,
    num_samples: int = -1,
    batch_size: int = 1,
    reward_weights: Optional[Dict[str, float]] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    base_model_path: Optional[str] = None,
    dtype: str = "auto",
) -> Dict[str, Any]:
    """Evaluate a model on GSM8K test set.
    
    Args:
        model_path: Path to the model checkpoint
        test_data_path: Path to test data
        output_path: Path to save results
        num_samples: Number of samples to evaluate (-1 for all)
        batch_size: Batch size for generation
        reward_weights: Weights for multi-reward scoring
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        base_model_path: Base model path for verl checkpoints
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"Loading model from {model_path}...")
    
    # Check if this is a verl checkpoint
    if is_verl_checkpoint(model_path):
        print("Detected verl FSDP checkpoint format")
        model, tokenizer = load_verl_checkpoint(model_path, base_model_path, dtype=dtype)
    else:
        # Standard HuggingFace model
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=resolve_torch_dtype(dtype),
            device_map="auto",
            trust_remote_code=True,
        )
    
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 设置 padding_side='left' 用于批量生成（decoder-only 模型需要左填充）
    tokenizer.padding_side = 'left'
    
    print(f"Loading test data from {test_data_path}...")
    test_samples = load_gsm8k_test_data(test_data_path)
    
    if num_samples > 0:
        test_samples = test_samples[:num_samples]
    
    print(f"Evaluating on {len(test_samples)} samples with batch_size={batch_size}...")
    
    # Default reward weights
    if reward_weights is None:
        reward_weights = {
            "w_correctness": 0.7,
            "w_format": 0.1,
            "w_reasoning": 0.2,
        }
    
    # Collect results
    results = []
    metrics = defaultdict(list)
    
    # Process in batches
    num_batches = (len(test_samples) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(test_samples))
        batch_samples = test_samples[start_idx:end_idx]
        
        # Extract prompts and ground truths
        prompts = [sample["prompt"] for sample in batch_samples]
        ground_truths = [sample["ground_truth"] for sample in batch_samples]
        
        # Generate responses in batch
        if batch_size > 1:
            responses = generate_responses_batch(
                model,
                tokenizer,
                prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        else:
            # Single sample, use original function
            responses = [generate_response(
                model,
                tokenizer,
                prompts[0],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )]
        
        # Process each response in the batch
        for i, (prompt, ground_truth, response) in enumerate(zip(prompts, ground_truths, responses)):
            # Compute multi-reward score
            score_result = compute_composite_score(
                data_source="openai/gsm8k",
                solution_str=response,
                ground_truth=ground_truth,
                **reward_weights,
            )
            
            # Store result
            result = {
                "prompt": prompt,
                "ground_truth": ground_truth,
                "response": response,
                **score_result,
            }
            results.append(result)
            
            # Aggregate metrics
            for key, value in score_result.items():
                if isinstance(value, (int, float, bool)):
                    metrics[key].append(float(value))
    
    # Compute summary statistics
    summary = {}
    for key, values in metrics.items():
        summary[f"{key}_mean"] = sum(values) / len(values)
        summary[f"{key}_std"] = (sum((v - summary[f"{key}_mean"])**2 for v in values) / len(values)) ** 0.5
    
    # Accuracy
    accuracy = sum(1 for r in results if r.get("is_correct", False)) / len(results)
    summary["accuracy"] = accuracy
    
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Total Score (mean): {summary.get('score_mean', 0):.4f}")
    print(f"Correctness Score (mean): {summary.get('correctness_score_mean', 0):.4f}")
    print(f"Format Score (mean): {summary.get('format_score_mean', 0):.4f}")
    print(f"Reasoning Score (mean): {summary.get('reasoning_score_mean', 0):.4f}")
    print("="*50)
    
    # Save results
    if output_path:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        output = {
            "model_path": model_path,
            "test_data_path": test_data_path,
            "num_samples": len(results),
            "reward_weights": reward_weights,
            "summary": summary,
            "results": results,
        }
        
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on GSM8K")
    
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint")
    parser.add_argument("--test-data", default="$HOME/data/gsm8k/test.parquet",
                        help="Path to test data")
    parser.add_argument("--output", help="Path to save results JSON")
    parser.add_argument("--num-samples", type=int, default=-1,
                        help="Number of samples to evaluate (-1 for all)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for generation (default: 16)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--w-correctness", type=float, default=0.7,
                        help="Weight for correctness reward")
    parser.add_argument("--w-format", type=float, default=0.1,
                        help="Weight for format reward")
    parser.add_argument("--w-reasoning", type=float, default=0.2,
                        help="Weight for reasoning reward")
    parser.add_argument("--base-model", type=str, default=None,
                        help="Base model path (for verl checkpoints)")
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto",
                        help="Model dtype. auto uses bf16 on Ampere+ and fp16 on V100/older GPUs.")
    
    args = parser.parse_args()
    
    # Expand environment variables in paths
    test_data = os.path.expandvars(args.test_data)
    
    reward_weights = {
        "w_correctness": args.w_correctness,
        "w_format": args.w_format,
        "w_reasoning": args.w_reasoning,
    }
    
    evaluate_model(
        model_path=args.model_path,
        test_data_path=test_data,
        output_path=args.output,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        reward_weights=reward_weights,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        base_model_path=args.base_model,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
