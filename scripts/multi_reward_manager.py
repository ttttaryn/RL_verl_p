#!/usr/bin/env python3
# Copyright 2024 Custom Implementation
# Enhanced Reward Manager with Detailed Logging
#
# This module provides a custom reward manager that extends verl's NaiveRewardManager
# to include detailed logging of multi-reward components.

import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

from analyze_metrics import MetricsLogger


@register("multi_reward")
class MultiRewardManager(AbstractRewardManager):
    """Enhanced reward manager with detailed multi-reward logging.
    
    This manager extends the basic functionality to:
    1. Use the multi-reward composite scoring system
    2. Log detailed metrics for each reward component
    3. Track statistics for analysis and visualization
    """
    
    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        log_dir: str = "logs",
        experiment_name: str = "multi_reward_experiment",
        log_detailed_rewards: bool = True,
        **kwargs,
    ):
        """Initialize the multi-reward manager.
        
        Args:
            tokenizer: Tokenizer for decoding responses
            num_examine: Number of samples to print for debugging
            compute_score: Custom scoring function
            reward_fn_key: Key for accessing data source
            log_dir: Directory for saving logs
            experiment_name: Name of the experiment
            log_detailed_rewards: Whether to log detailed reward components
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.log_detailed_rewards = log_detailed_rewards
        
        # Initialize metrics logger
        if log_detailed_rewards:
            self.metrics_logger = MetricsLogger(log_dir, experiment_name)
        else:
            self.metrics_logger = None
        
        # Track aggregate statistics
        self.step_counter = 0
        self.batch_stats = defaultdict(list)
    
    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | Dict[str, Any]:
        """Compute rewards for a batch of data with detailed logging.
        
        Args:
            data: DataProto containing the batch data
            return_dict: Whether to return detailed info dict
            
        Returns:
            Reward tensor or dict with reward tensor and extra info
        """
        # Check if rm_scores already exist
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]
        
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        # Batch statistics for this call
        batch_stats = defaultdict(list)
        already_print_data_sources = {}
        
        for i in range(len(data)):
            data_item = data[i]
            
            # Extract prompt and response
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            # Decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            # Get ground truth and data source
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            
            # Compute score
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            if isinstance(score, dict):
                reward = score["score"]
                
                # Store detailed info
                for key, value in score.items():
                    reward_extra_info[key].append(value)
                    if isinstance(value, (int, float, bool)):
                        batch_stats[key].append(float(value))
            else:
                reward = score
                batch_stats["score"].append(float(score))
            
            reward_tensor[i, valid_response_length - 1] = reward
            
            # Print samples for debugging
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)
        
        # Log batch statistics
        if self.log_detailed_rewards and self.metrics_logger is not None:
            self.step_counter += 1
            
            # Compute aggregate statistics
            log_data = {}
            for key, values in batch_stats.items():
                if values:
                    log_data[f"{key}_mean"] = sum(values) / len(values)
                    log_data[f"{key}_min"] = min(values)
                    log_data[f"{key}_max"] = max(values)
            
            self.metrics_logger.log(self.step_counter, log_data)
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        else:
            return reward_tensor
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics from training.
        
        Returns:
            Dictionary of aggregate statistics
        """
        if self.metrics_logger:
            return dict(self.metrics_logger.metrics_history)
        return {}
    
    def save_logs(self):
        """Save all logged metrics."""
        if self.metrics_logger:
            self.metrics_logger.save_summary()
