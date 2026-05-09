#!/usr/bin/env python3
"""GRPO Trainer — Group Relative Policy Optimization for LLM math reasoning.

Usage:
    # Single GPU
    python scripts/train_grpo.py --config config/grpo_gsm8k.yaml

    # This trainer is single-process/single-GPU. Use PPO/verl for multi-GPU.

Key features:
  - No critic model needed (~50% VRAM savings vs PPO)
  - Group-relative advantage: (reward - group_mean) / group_std
  - KL penalty baked into the loss (no separate KL controller)
  - Compatible with the multi-reward system (correctness + format + reasoning)
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None

# Add project and script paths. This lets the file work both as:
#   python scripts/train_grpo.py
# and:
#   python -m scripts.train_grpo
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from verl_rewards import compute_composite_score
try:
    from scripts.grpo_core import (
        compute_group_relative_advantage,
        compute_global_advantage,
        compute_grpo_loss,
        compute_kl_divergence,
        collect_group_rollout_stats,
    )
except ModuleNotFoundError:
    from grpo_core import (
        compute_group_relative_advantage,
        compute_global_advantage,
        compute_grpo_loss,
        compute_kl_divergence,
        collect_group_rollout_stats,
    )


def load_config(config_path: str) -> dict:
    """Load GRPO config and resolve OmegaConf-style interpolations."""
    if OmegaConf is not None:
        cfg = OmegaConf.load(config_path)
        return OmegaConf.to_container(cfg, resolve=True)

    with open(config_path, "r", encoding="utf-8") as f:
        import yaml
        config = yaml.safe_load(f)

    # Minimal fallback for this repo's config if omegaconf is unavailable.
    for key, value in config.get("data", {}).items():
        if isinstance(value, str):
            value = value.replace("${oc.env:HOME}", os.environ.get("HOME", ""))
            config["data"][key] = os.path.expandvars(value)

    trainer = config.get("trainer", {})
    if isinstance(trainer.get("default_local_dir"), str):
        trainer["default_local_dir"] = (
            trainer["default_local_dir"]
            .replace("${trainer.project_name}", str(trainer.get("project_name", "")))
            .replace("${trainer.experiment_name}", str(trainer.get("experiment_name", "")))
        )

    return config


# ── Dataset ───────────────────────────────────────────────────────────────

def _extract_prompt(prompt_data) -> str:
    """Extract a plain prompt string from common GSM8K parquet schemas."""
    if isinstance(prompt_data, np.ndarray):
        prompt_data = prompt_data.tolist()
    if isinstance(prompt_data, list):
        for msg in prompt_data:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", ""))
        if prompt_data:
            last_msg = prompt_data[-1]
            if isinstance(last_msg, dict):
                return str(last_msg.get("content", last_msg))
            return str(last_msg)
        return ""
    return str(prompt_data)


def _extract_ground_truth(row) -> str:
    reward_model = row.get("reward_model", {})
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"])
    return str(row.get("ground_truth", row.get("answer", "")))


class GSM8KDataset(Dataset):
    """GSM8K dataset for GRPO training.

    Returns prompts with ground truth for reward computation.
    """

    def __init__(self, parquet_path: str, max_prompt_length: int = 512):
        df = pd.read_parquet(parquet_path)
        self.prompts = []
        self.ground_truths = []
        self.data_sources = []

        for _, row in df.iterrows():
            prompt = _extract_prompt(row.get("prompt", row.get("question", "")))
            answer = _extract_ground_truth(row)

            # Ensure prompt includes the GSM8K instruction
            if "####" not in prompt:
                prompt = prompt + " Let's think step by step and output the final answer after \"####\"."

            self.prompts.append(prompt)
            self.ground_truths.append(answer)
            self.data_sources.append(row.get("data_source", "gsm8k"))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {
            "prompt": self.prompts[idx],
            "ground_truth": self.ground_truths[idx],
            "data_source": self.data_sources[idx],
            "prompt_id": idx,  # Used for group assignment
        }


# ── Rollout Engine (vLLM) ─────────────────────────────────────────────────

class RolloutEngine:
    """Generate responses from the current policy.

    The training path intentionally uses the live HF policy model instead of a
    separate vLLM engine. A static vLLM instance would keep sampling from stale
    initial weights unless explicit weight synchronization is implemented.
    """

    def __init__(self, model_path: str, temperature: float = 0.7,
                 top_p: float = 0.95, max_tokens: int = 512,
                 gpu_memory_utilization: float = 0.5,
                 tensor_parallel_size: int = 1,
                 use_vllm: bool = False):
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self._use_vllm = False
        self.llm = None
        self.model = None
        self.tokenizer = None

        if use_vllm:
            from vllm import LLM, SamplingParams
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True,
            )
            self.sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            self._use_vllm = True
            print(f"  vLLM engine initialized (tp={tensor_parallel_size})")

    def set_hf_model(self, model, tokenizer):
        """Set the live HF policy model for rollout generation."""
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompts: List[str], n_samples: int = 1) -> List[str]:
        """Generate n_samples responses for each prompt."""
        if self._use_vllm:
            all_prompts = []
            for p in prompts:
                for _ in range(n_samples):
                    all_prompts.append(p)

            outputs = self.llm.generate(all_prompts, self.sampling_params)
            return [o.outputs[0].text for o in outputs]

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("HF rollout model/tokenizer were not initialized")

        was_training = self.model.training
        self.model.eval()
        responses = []
        for prompt in prompts:
            for _ in range(n_samples):
                inputs = self.tokenizer(
                    prompt, return_tensors="pt",
                    truncation=True, max_length=512,
                ).to(self.model.device)

                with torch.no_grad():
                    output_ids = self.model.generate(
                        inputs.input_ids,
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                response = self.tokenizer.decode(
                    output_ids[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                )
                responses.append(response)

        if was_training:
            self.model.train()
        return responses


# ── GRPO Trainer ──────────────────────────────────────────────────────────

class GRPOTrainer:
    """Self-contained GRPO trainer for LLM math reasoning.

    Architecture (compared to PPO):
      PPO:  Actor + Critic + Reference + Reward (4 models)
      GRPO: Actor + Reference + Reward            (3 models, no Critic)

    The key innovation is replacing GAE with group-relative advantage:
    for each prompt, we sample K responses and standardize rewards within
    the group. A response is "good" if it's better than the average response
    to the same prompt.
    """

    def __init__(self, config: dict):
        self.config = config

        # Extract config sections
        self.data_config = config.get("data", {})
        self.algo_config = config.get("algorithm", {})
        self.trainer_config = config.get("trainer", {})
        self.model_config = config.get("actor_rollout_ref", {})

        # GRPO hyperparameters
        self.group_size = self.algo_config.get("group_size", 4)
        self.clip_ratio = self.algo_config.get("clip_ratio", 0.2)
        self.kl_coef = self.algo_config.get("kl_penalty", {}).get("kl_coef", 0.001)
        self.kl_estimator = self.algo_config.get("kl_penalty", {}).get("estimator", "k1")
        self.adv_method = self.algo_config.get("advantage", {}).get("method", "group_relative")
        self.adv_norm = self.algo_config.get("advantage", {}).get("norm_method", "standardize")

        # Training params
        self.lr = self.model_config.get("actor", {}).get("optim", {}).get("lr", 1e-6)
        self.total_steps = self.trainer_config.get("total_training_steps", 1000)
        self.save_freq = self.trainer_config.get("save_freq", 100)
        self.test_freq = self.trainer_config.get("test_freq", 100)
        self.log_dir = self.trainer_config.get("default_local_dir", "./checkpoints/grpo")
        self.project_name = self.trainer_config.get("project_name", "grpo")
        self.experiment_name = self.trainer_config.get("experiment_name", "gsm8k")

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise NotImplementedError(
                "scripts/train_grpo.py is a single-process trainer. "
                "Use one process/GPU or implement DDP/FSDP before torchrun."
            )
        # Note: full multi-GPU DDP requires torch.distributed.init_process_group()
        # For single-GPU or FSDP-managed multi-GPU, is_main is always True
        self.is_main = True

    def setup(self):
        """Initialize models, tokenizer, optimizer, and data."""
        model_path = self.model_config.get("model", {}).get("path", "Qwen/Qwen2.5-0.5B-Instruct")

        if self.is_main:
            print(f"Loading model: {model_path}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Policy model (actor)
        self.policy = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=None,
        )
        if torch.cuda.is_available():
            self.policy = self.policy.to(self.device)

        # Reference model (frozen, for KL computation)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=None,
        )
        if torch.cuda.is_available():
            self.ref_model = self.ref_model.to(self.device)
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

        # NO critic model — this is the key advantage of GRPO

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.lr,
        )

        # Rollout engine
        rollout_temp = self.model_config.get("rollout", {}).get("temperature", 0.7)
        rollout_top_p = self.model_config.get("rollout", {}).get("top_p", 0.95)
        max_response_len = self.data_config.get("max_response_length", 512)

        rollout_use_vllm = self.model_config.get("rollout", {}).get("use_vllm", False)
        if rollout_use_vllm and self.is_main:
            print("  Warning: vLLM rollout is static unless weight sync is implemented.")

        self.rollout = RolloutEngine(
            model_path=model_path,
            temperature=rollout_temp,
            top_p=rollout_top_p,
            max_tokens=max_response_len,
            gpu_memory_utilization=self.model_config.get("rollout", {}).get("gpu_memory_utilization", 0.5),
            tensor_parallel_size=self.model_config.get("rollout", {}).get("tensor_model_parallel_size", 1),
            use_vllm=rollout_use_vllm,
        )
        if not self.rollout._use_vllm:
            self.rollout.set_hf_model(self.policy, self.tokenizer)

        # Dataset
        train_path = self.data_config.get("train_files", "")
        train_path = os.path.expandvars(train_path)
        self.train_dataset = GSM8KDataset(train_path, self.data_config.get("max_prompt_length", 512))

        val_path = self.data_config.get("val_files", "")
        if val_path:
            val_path = os.path.expandvars(val_path)
            self.val_dataset = GSM8KDataset(val_path, self.data_config.get("max_prompt_length", 512))
        else:
            self.val_dataset = None

        # Reward weights
        self.reward_weights = {
            "correctness": float(self.config.get("reward_weights", {}).get("w_correctness", 0.7)),
            "format": float(self.config.get("reward_weights", {}).get("w_format", 0.1)),
            "reasoning": float(self.config.get("reward_weights", {}).get("w_reasoning", 0.2)),
        }

        if self.is_main:
            print(f"  Train samples: {len(self.train_dataset)}")
            print(f"  Group size: K={self.group_size}")
            print(f"  Reward weights: {self.reward_weights}")
            print(f"  KL coef: β={self.kl_coef}")
            print(f"  Learning rate: {self.lr}")
            print(f"  Total steps: {self.total_steps}")

    def compute_rewards(self, prompts: List[str], responses: List[str],
                        ground_truths: List[str], data_sources: List[str]) -> List[Dict]:
        """Compute multi-dimensional rewards for all responses."""
        results = []
        for prompt, response, gt, ds in zip(prompts, responses, ground_truths, data_sources):
            result = compute_composite_score(
                data_source=ds,
                solution_str=response,
                ground_truth=gt,
                extra_info=None,
                w_correctness=self.reward_weights["correctness"],
                w_format=self.reward_weights["format"],
                w_reasoning=self.reward_weights["reasoning"],
            )
            results.append(result)
        return results

    def compute_log_probs(self, model, input_ids: torch.Tensor,
                          attention_mask: torch.Tensor,
                          requires_grad: bool = False) -> torch.Tensor:
        """Compute token-level log probabilities under a model."""
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        with context:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [batch, seq_len, vocab]

        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log probs of the actual tokens (shifted for next-token prediction)
        shift_log_probs = log_probs[:, :-1, :]  # [batch, seq_len-1, vocab]
        shift_labels = input_ids[:, 1:]           # [batch, seq_len-1]

        token_log_probs = torch.gather(
            shift_log_probs, dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)  # [batch, seq_len-1]

        # Pad to original length
        padded = F.pad(token_log_probs, (1, 0), value=0.0)  # [batch, seq_len]
        return padded

    def build_policy_batch(self, prompts: List[str], responses: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize prompt+response and return a mask covering response tokens only."""
        full_texts = [prompt + response for prompt, response in zip(prompts, responses)]
        encoded = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.data_config.get("max_prompt_length", 512) + self.data_config.get("max_response_length", 512),
            return_tensors="pt",
        )
        prompt_encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.data_config.get("max_prompt_length", 512),
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        response_mask = attention_mask.clone()

        prompt_lengths = prompt_encoded["attention_mask"].sum(dim=1).tolist()
        for i, prompt_len in enumerate(prompt_lengths):
            response_mask[i, :int(prompt_len)] = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
        }

    def training_step(self, prompts: List[str], ground_truths: List[str],
                      data_sources: List[str], prompt_ids: List[int]) -> Dict[str, float]:
        """Execute one GRPO training step.

        For each prompt, sample K responses, compute rewards, compute
        group-relative advantages, and update the policy.
        """
        # 1. Rollout: generate K responses per prompt
        all_prompts = []
        all_gt = []
        all_ds = []
        all_pids = []

        for prompt, gt, ds, pid in zip(prompts, ground_truths, data_sources, prompt_ids):
            for _ in range(self.group_size):
                all_prompts.append(prompt)
                all_gt.append(gt)
                all_ds.append(ds)
                all_pids.append(pid)

        responses = self.rollout.generate(prompts, n_samples=self.group_size)

        # 2. Compute rewards
        reward_results = self.compute_rewards(all_prompts, responses, all_gt, all_ds)
        rewards = [r["score"] for r in reward_results]

        # 3. Compute group-relative advantages
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        pid_tensor = torch.tensor(all_pids, dtype=torch.long, device=self.device)

        if self.adv_method == "group_relative":
            advantages = compute_group_relative_advantage(
                reward_tensor, pid_tensor, norm_method=self.adv_norm,
            )
        else:
            advantages = compute_global_advantage(reward_tensor)

        # 4. Prepare prompt-conditioned training batch.
        batch_tensors = self.build_policy_batch(all_prompts, responses)
        input_ids = batch_tensors["input_ids"]
        attention_mask = batch_tensors["attention_mask"]
        response_mask = batch_tensors["response_mask"]

        # 5. Compute reference model log probs (frozen, for KL penalty)
        with torch.no_grad():
            ref_log_probs = self.compute_log_probs(self.ref_model, input_ids, attention_mask)

        # 6. Single policy forward pass — in on-policy GRPO with 1 epoch,
        #    old_log_probs == new_log_probs since no update has occurred yet.
        #    The ratio r(θ) = π_new/π_old = 1.0 on the first epoch, so
        #    advantage clipping only matters when multi_epoch > 1.
        self.policy.train()
        log_probs = self.compute_log_probs(self.policy, input_ids, attention_mask, requires_grad=True)

        # 7. Compute KL divergence (policy vs reference)
        kl = compute_kl_divergence(log_probs, ref_log_probs, estimator=self.kl_estimator)

        # 8. Compute GRPO loss (old_log_probs = log_probs for on-policy first epoch)
        loss, stats = compute_grpo_loss(
            log_probs=log_probs,
            old_log_probs=log_probs.detach(),
            advantages=advantages,
            kl_divergence=kl,
            clip_ratio=self.clip_ratio,
            kl_coef=self.kl_coef,
            response_mask=response_mask,
        )

        # 9. Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 10. Collect step stats
        group_stats = collect_group_rollout_stats(
            reward_tensor, pid_tensor,
        )

        step_stats = {
            **stats,
            **group_stats,
            "reward_mean": np.mean(rewards),
            "reward_std": np.std(rewards),
            "correctness_mean": np.mean([r["correctness_score"] for r in reward_results]),
            "format_mean": np.mean([r["format_score"] for r in reward_results]),
            "reasoning_mean": np.mean([r["reasoning_score"] for r in reward_results]),
            "accuracy": np.mean([r["is_correct"] for r in reward_results]),
        }

        return step_stats

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation on the test set."""
        if self.val_dataset is None:
            return {}

        self.policy.eval()

        all_rewards = []
        all_correct = []
        all_format = []
        all_reasoning = []

        dataloader = DataLoader(self.val_dataset, batch_size=8, shuffle=False)

        for batch in tqdm(dataloader, desc="Validation", disable=not self.is_main):
            responses = self.rollout.generate(
                batch["prompt"], n_samples=1,
            )

            for prompt, response, gt, ds in zip(
                batch["prompt"], responses,
                batch["ground_truth"], batch["data_source"],
            ):
                result = compute_composite_score(
                    data_source=ds,
                    solution_str=response,
                    ground_truth=gt,
                    extra_info=None,
                    w_correctness=self.reward_weights["correctness"],
                    w_format=self.reward_weights["format"],
                    w_reasoning=self.reward_weights["reasoning"],
                )
                all_rewards.append(result["score"])
                all_correct.append(result["is_correct"])
                all_format.append(result["format_score"])
                all_reasoning.append(result["reasoning_score"])

        return {
            "val_score": np.mean(all_rewards),
            "val_accuracy": np.mean(all_correct),
            "val_format": np.mean(all_format),
            "val_reasoning": np.mean(all_reasoning),
        }

    def train(self):
        """Main GRPO training loop."""
        self.setup()

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.data_config.get("train_batch_size", 256),
            shuffle=True,
        )

        history = defaultdict(list)
        os.makedirs(self.log_dir, exist_ok=True)

        if self.is_main:
            print(f"\nStarting GRPO training ({self.total_steps} steps)...")
            print(f"  {'Step':>6} {'Loss':>8} {'Reward':>8} {'Acc':>7} {'KL':>8} {'Clip%':>8}")

        step = 0
        pbar = tqdm(total=self.total_steps, disable=not self.is_main)

        while step < self.total_steps:
            for batch in dataloader:
                if step >= self.total_steps:
                    break

                prompts = batch["prompt"]
                ground_truths = batch["ground_truth"]
                data_sources = batch["data_source"]
                prompt_ids = batch["prompt_id"].tolist()

                try:
                    stats = self.training_step(
                        prompts, ground_truths, data_sources, prompt_ids,
                    )
                except torch.cuda.OutOfMemoryError:
                    print(f"  OOM at step {step}, skipping...")
                    torch.cuda.empty_cache()
                    continue

                for k, v in stats.items():
                    history[k].append(v)

                if self.is_main:
                    pbar.update(1)
                    pbar.set_postfix({
                        "loss": f"{stats['loss']:.4f}",
                        "rew": f"{stats['reward_mean']:.3f}",
                        "acc": f"{stats['accuracy']:.2%}",
                    })

                step += 1

                # Periodic logging
                if step % 10 == 0 and self.is_main:
                    print(
                        f"  {step:>6d} {stats['loss']:>8.4f} {stats['reward_mean']:>8.3f} "
                        f"{stats['accuracy']:>7.2%} {stats['approx_kl']:>8.4f} "
                        f"{stats['ratio_clip_frac']:>8.3f}"
                    )

                # Save checkpoint
                if step % self.save_freq == 0:
                    ckpt_path = os.path.join(self.log_dir, f"step_{step}")
                    self.policy.save_pretrained(ckpt_path)
                    self.tokenizer.save_pretrained(ckpt_path)
                    if self.is_main:
                        print(f"  Checkpoint saved to {ckpt_path}")

                # Validation
                if step % self.test_freq == 0:
                    val_stats = self.validate()
                    if val_stats and self.is_main:
                        print(
                            f"  [Val {step:>5d}] accuracy={val_stats['val_accuracy']:.2%} "
                            f"format={val_stats['val_format']:.3f} "
                            f"reasoning={val_stats['val_reasoning']:.3f}"
                        )

        # Final save
        final_path = os.path.join(self.log_dir, "final")
        self.policy.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)

        # Save training history
        history_path = os.path.join(self.log_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(dict(history), f, indent=2)

        if self.is_main:
            print(f"\nTraining complete. Model saved to {final_path}")
            print(f"Training history saved to {history_path}")


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GRPO Trainer for GSM8K")
    parser.add_argument("--config", type=str, default="config/grpo_gsm8k.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--w-correctness", type=float, default=None)
    parser.add_argument("--w-format", type=float, default=None)
    parser.add_argument("--w-reasoning", type=float, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    # CLI overrides
    if args.w_correctness is not None or args.w_format is not None or args.w_reasoning is not None:
        if "reward_weights" not in config:
            config["reward_weights"] = {}
        if args.w_correctness is not None:
            config["reward_weights"]["w_correctness"] = args.w_correctness
        if args.w_format is not None:
            config["reward_weights"]["w_format"] = args.w_format
        if args.w_reasoning is not None:
            config["reward_weights"]["w_reasoning"] = args.w_reasoning
    if args.group_size is not None:
        config["algorithm"]["group_size"] = args.group_size
    if args.total_steps is not None:
        config["trainer"]["total_training_steps"] = args.total_steps
    if args.lr is not None:
        config["actor_rollout_ref"]["actor"]["optim"]["lr"] = args.lr
    if args.model is not None:
        config["actor_rollout_ref"]["model"]["path"] = args.model

    trainer = GRPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
