#!/usr/bin/env python3
"""GRPO Trainer — Group Relative Policy Optimization for LLM math reasoning.

Usage:
    # Single GPU
    python scripts/train_grpo.py --config config/grpo_gsm8k.yaml

    # This trainer is single-process/single-GPU.

Key features:
  - No critic/value model needed
  - Group-relative advantage: (reward - group_mean) / group_std
  - KL penalty baked into the loss (no separate KL controller)
  - Compatible with the multi-reward system (correctness + format + reasoning)
"""

import argparse
import gc
import json
import os
import shutil
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

try:
    from scripts.reward_fn import compute_score, compute_score_legacy_reasoning
except ModuleNotFoundError:
    from reward_fn import compute_score, compute_score_legacy_reasoning

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
    config_file = Path(config_path)

    def deep_merge(base: dict, override: dict) -> dict:
        result = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    if OmegaConf is not None:
        cfg = OmegaConf.load(config_path)
        defaults = cfg.get("defaults", [])
        if defaults:
            merged = OmegaConf.create({})
            for item in defaults:
                if isinstance(item, str):
                    name = item
                elif isinstance(item, dict):
                    name = next(iter(item.values()))
                else:
                    continue
                if name == "_self_":
                    continue
                base_path = config_file.parent / f"{name}.yaml"
                if not base_path.exists():
                    base_path = config_file.parent.parent / f"{name}.yaml"
                if base_path.exists():
                    merged = OmegaConf.merge(merged, OmegaConf.load(base_path))
            cfg = OmegaConf.merge(merged, cfg)
            if "defaults" in cfg:
                del cfg["defaults"]
        return OmegaConf.to_container(cfg, resolve=True)

    with open(config_path, "r", encoding="utf-8") as f:
        import yaml
        config = yaml.safe_load(f)

    defaults = config.pop("defaults", [])
    if defaults:
        merged = {}
        for item in defaults:
            name = item if isinstance(item, str) else next(iter(item.values()), None)
            if not name or name == "_self_":
                continue
            base_path = config_file.parent / f"{name}.yaml"
            if not base_path.exists():
                base_path = config_file.parent.parent / f"{name}.yaml"
            if base_path.exists():
                with open(base_path, "r", encoding="utf-8") as bf:
                    base_cfg = yaml.safe_load(bf)
                merged = deep_merge(merged, base_cfg)
        config = deep_merge(merged, config)

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

    vLLM is supported through checkpoint-based synchronization: the trainer
    periodically saves the current HF policy to a local sync directory and
    reloads the vLLM engine from that directory. This keeps rollout reasonably
    close to the trained policy without using HF generate for sampling.
    """

    def __init__(self, model_path: str, temperature: float = 0.7,
                 top_p: float = 0.95, max_tokens: int = 512,
                 gpu_memory_utilization: float = 0.5,
                 tensor_parallel_size: int = 1,
                 use_vllm: bool = False,
                 generation_batch_size: int = 8,
                 vllm_dtype: str = "float16",
                 enforce_eager: bool = False):
        self.model_path = model_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.generation_batch_size = generation_batch_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.vllm_dtype = vllm_dtype
        self.enforce_eager = enforce_eager
        self._use_vllm = False
        self.llm = None
        self.model = None
        self.tokenizer = None
        self.sampling_params = None
        self._LLM = None
        self._SamplingParams = None

        if use_vllm:
            try:
                from vllm import LLM, SamplingParams
            except ImportError as exc:
                raise ImportError(
                    "vLLM rollout requested but vllm is not installed. "
                    "Install vllm or pass --no-vllm to use HF rollout."
                ) from exc
            self._LLM = LLM
            self._SamplingParams = SamplingParams
            self._load_vllm(model_path)
            self._use_vllm = True

    def _load_vllm(self, model_path: str):
        """Load a vLLM engine from a model path."""
        self.model_path = model_path
        self.llm = self._LLM(
            model=model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            dtype=self.vllm_dtype,
            enforce_eager=self.enforce_eager,
        )
        self.sampling_params = self._SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        print(
            "  vLLM engine initialized "
            f"(model={model_path}, tp={self.tensor_parallel_size}, "
            f"mem={self.gpu_memory_utilization}, dtype={self.vllm_dtype})"
        )

    def unload_vllm(self):
        """Best-effort release of the current vLLM engine."""
        if self.llm is not None:
            del self.llm
            self.llm = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def sync_from_hf_model(self, model, tokenizer, sync_dir: str, step: int):
        """Save the current policy and reload vLLM from the saved checkpoint."""
        if not self._use_vllm:
            return

        sync_root = Path(sync_dir)
        sync_path = sync_root / "policy_current"
        sync_root.mkdir(parents=True, exist_ok=True)

        self.unload_vllm()
        if sync_path.exists():
            shutil.rmtree(sync_path)
        model.save_pretrained(sync_path)
        tokenizer.save_pretrained(sync_path)
        self._load_vllm(str(sync_path))
        print(f"  vLLM rollout synced from policy checkpoint at step {step}: {sync_path}")

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

            if self.llm is None:
                raise RuntimeError("vLLM rollout engine was not initialized")
            outputs = self.llm.generate(all_prompts, self.sampling_params)
            return [o.outputs[0].text for o in outputs]

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("HF rollout model/tokenizer were not initialized")

        all_prompts = []
        for prompt in prompts:
            for _ in range(n_samples):
                all_prompts.append(prompt)

        was_training = self.model.training
        old_padding_side = self.tokenizer.padding_side
        self.model.eval()
        self.tokenizer.padding_side = "left"
        responses = []
        batch_size = max(1, self.generation_batch_size)
        for start in range(0, len(all_prompts), batch_size):
            batch_prompts = all_prompts[start:start + batch_size]
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            input_width = inputs.input_ids.shape[1]
            for output in output_ids:
                response = self.tokenizer.decode(
                    output[input_width:],
                    skip_special_tokens=True,
                )
                responses.append(response)

        self.tokenizer.padding_side = old_padding_side
        if was_training:
            self.model.train()
        return responses


# ── GRPO Trainer ──────────────────────────────────────────────────────────

class GRPOTrainer:
    """Self-contained GRPO trainer for LLM math reasoning.

    Architecture:
      GRPO: Actor + Reference + Reward, no Critic/Value model

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
        self.train_batch_size = self.data_config.get("train_batch_size", 2)
        self.total_steps = self.trainer_config.get("total_training_steps", 1000)
        self.save_freq = self.trainer_config.get("save_freq", 100)
        self.test_freq = self.trainer_config.get("test_freq", 100)
        self.log_dir = self.trainer_config.get("default_local_dir", "./checkpoints/grpo")
        self.project_name = self.trainer_config.get("project_name", "grpo")
        self.experiment_name = self.trainer_config.get("experiment_name", "gsm8k")
        rollout_cfg = self.model_config.get("rollout", {})
        self.use_vllm_rollout = bool(rollout_cfg.get("use_vllm", True))
        self.vllm_sync_interval = int(rollout_cfg.get("vllm_sync_interval", 1))
        self.vllm_sync_dir = rollout_cfg.get(
            "vllm_sync_dir",
            os.path.join(self.log_dir, "_vllm_sync"),
        )
        if isinstance(self.vllm_sync_dir, str):
            self.vllm_sync_dir = (
                self.vllm_sync_dir
                .replace("${trainer.project_name}", str(self.project_name))
                .replace("${trainer.experiment_name}", str(self.experiment_name))
            )

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
        model_path = self.model_config.get("model", {}).get("path", "Qwen/Qwen2.5-1.5B-Instruct")

        if self.is_main:
            print(f"Loading model: {model_path}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Policy model (actor)
        self.policy = AutoModelForCausalLM.from_pretrained(
            model_path,
            # Keep trainable policy weights in fp32 for optimizer stability.
            # fp16 trainable weights can produce invalid sampling probabilities
            # after a single GRPO update on small batches.
            dtype=torch.float32,
            device_map=None,
        )
        if torch.cuda.is_available():
            self.policy = self.policy.to(self.device)

        # Reference model (frozen, for KL computation)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
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

        self.rollout = RolloutEngine(
            model_path=model_path,
            temperature=rollout_temp,
            top_p=rollout_top_p,
            max_tokens=max_response_len,
            gpu_memory_utilization=self.model_config.get("rollout", {}).get("gpu_memory_utilization", 0.5),
            tensor_parallel_size=self.model_config.get("rollout", {}).get("tensor_model_parallel_size", 1),
            use_vllm=self.use_vllm_rollout,
            generation_batch_size=self.model_config.get("rollout", {}).get("generation_batch_size", 2),
            vllm_dtype=self.model_config.get("rollout", {}).get("vllm_dtype", "float16"),
            enforce_eager=bool(self.model_config.get("rollout", {}).get("enforce_eager", False)),
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
        reward_cfg = self.config.get("custom_reward_function", {}).get("reward_kwargs", {})
        self.reward_kwargs = dict(reward_cfg)
        self.reward_kwargs.setdefault("w_correctness", self.reward_weights["correctness"])
        self.reward_kwargs.setdefault("w_format", self.reward_weights["format"])
        self.reward_kwargs.setdefault("w_reasoning", self.reward_weights["reasoning"])
        self.reward_kwargs.setdefault("correctness_method", "flexible")
        reward_mode = self.config.get("reward_mode", "legacy_reasoning")
        self.reward_kwargs.setdefault(
            "answer_conditioned_reasoning",
            reward_mode == "answer_conditioned_reasoning",
        )
        self.warmup_use_legacy_reward = bool(
            self.config.get("warmup_use_legacy_reward", reward_mode == "legacy_reasoning")
        )

        if self.is_main:
            print(f"  Train samples: {len(self.train_dataset)}")
            print(f"  Train batch size: {self.train_batch_size} prompts")
            print(f"  Group size: K={self.group_size}")
            print(f"  Max response len: {self.data_config.get('max_response_length', 256)} tokens")
            print(f"  Reward mode: {'legacy_reasoning' if self.warmup_use_legacy_reward else 'answer_conditioned_reasoning'}")
            print(f"  Reward weights: {self.reward_weights}")
            print(f"  Rollout engine: {'vLLM' if self.use_vllm_rollout else 'HF live policy'}")
            if self.use_vllm_rollout:
                print(f"  vLLM sync interval: every {self.vllm_sync_interval} step(s)")
            print(f"  Learning rate: {self.lr}")
            print(f"  Total steps: {self.total_steps}")

    def maybe_sync_vllm_rollout(self, step: int, force: bool = False):
        """Synchronize vLLM rollout weights from the current policy."""
        if not self.use_vllm_rollout:
            return
        if self.vllm_sync_interval <= 0:
            return
        if force or step % self.vllm_sync_interval == 0:
            self.rollout.sync_from_hf_model(
                self.policy,
                self.tokenizer,
                self.vllm_sync_dir,
                step,
            )

    def compute_rewards(self, prompts: List[str], responses: List[str],
                        ground_truths: List[str], data_sources: List[str]) -> List[Dict]:
        """Compute multi-dimensional rewards for all responses."""
        results = []
        for prompt, response, gt, ds in zip(prompts, responses, ground_truths, data_sources):
            reward_fn = compute_score_legacy_reasoning if self.warmup_use_legacy_reward else compute_score
            result = reward_fn(
                data_source=ds,
                solution_str=response,
                ground_truth=gt,
                extra_info=None,
                **self.reward_kwargs,
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

        if not torch.isfinite(loss):
            stats["skipped_step"] = 1.0
            return stats

        # 9. Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        if not torch.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            stats["skipped_step"] = 1.0
            stats["grad_norm"] = float("nan")
            return stats
        self.optimizer.step()
        stats["grad_norm"] = float(grad_norm.item())
        stats["skipped_step"] = 0.0

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
                reward_fn = compute_score_legacy_reasoning if self.warmup_use_legacy_reward else compute_score
                result = reward_fn(
                    data_source=ds,
                    solution_str=response,
                    ground_truth=gt,
                    extra_info=None,
                    **self.reward_kwargs,
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
            batch_size=self.train_batch_size,
            shuffle=True,
        )

        history = defaultdict(list)
        os.makedirs(self.log_dir, exist_ok=True)

        if self.is_main:
            print(f"\nStarting GRPO training ({self.total_steps} steps)...")
            print(f"  {'Step':>6} {'Loss':>8} {'Reward':>8} {'Acc':>7} {'KL':>8} {'Clip%':>8}")

        step = 0
        consecutive_oom = 0
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
                    if step > 0:
                        self.maybe_sync_vllm_rollout(step)
                    stats = self.training_step(
                        prompts, ground_truths, data_sources, prompt_ids,
                    )
                except torch.cuda.OutOfMemoryError:
                    consecutive_oom += 1
                    print(
                        f"  OOM at step {step} (consecutive={consecutive_oom}). "
                        "Try --batch-size 1 --group-size 2 --max-response-length 128."
                    )
                    torch.cuda.empty_cache()
                    if consecutive_oom >= 3:
                        raise RuntimeError(
                            "Repeated CUDA OOM before completing a GRPO step. "
                            "Reduce --batch-size, --group-size, or --max-response-length."
                        )
                    continue
                consecutive_oom = 0

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
                        f"{stats['accuracy']:>7.2%} {stats['seq_kl']:>8.4f} "
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
                    self.maybe_sync_vllm_rollout(step, force=True)
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

        metadata_path = os.path.join(self.log_dir, "training_metadata.json")
        training_metadata = {
            "algorithm": "grpo",
            "base_model": self.config.get("base_model", "Qwen/Qwen2.5-1.5B-Instruct"),
            "sft_checkpoint": self.config.get("sft_checkpoint", "./checkpoints/sft_warmup/final"),
            "grpo_config": self.config.get("grpo_config", "config/grpo_gsm8k.yaml"),
            "reward_mode": "legacy_reasoning" if self.warmup_use_legacy_reward else "answer_conditioned_reasoning",
            "reward_modes": {
                "legacy_reasoning": "Used for warmup/training or comparison; reasoning is heuristic and not gated on correctness.",
                "answer_conditioned_reasoning": "Used for final diagnostics and reward-hacking checks; reasoning reward is gated on correctness.",
            },
            "group_size": self.group_size,
            "batch_size": self.train_batch_size,
            "temperature": self.model_config.get("rollout", {}).get("temperature", 0.7),
            "kl_coef": self.kl_coef,
            "rollout_engine": "vllm" if self.use_vllm_rollout else "hf_policy",
            "vllm_sync_interval": self.vllm_sync_interval if self.use_vllm_rollout else None,
            "vllm_sync_dir": self.vllm_sync_dir if self.use_vllm_rollout else None,
            "vllm_gpu_memory_utilization": self.model_config.get("rollout", {}).get("gpu_memory_utilization", None),
            "vllm_dtype": self.model_config.get("rollout", {}).get("vllm_dtype", None),
            "reward_weights": self.reward_weights,
            "total_steps": self.total_steps,
        }
        with open(metadata_path, "w") as f:
            json.dump(training_metadata, f, indent=2)

        if self.is_main:
            print(f"\nTraining complete. Model saved to {final_path}")
            print(f"Training history saved to {history_path}")
            print(f"Training metadata saved to {metadata_path}")


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GRPO Trainer for GSM8K")
    parser.add_argument("--config", type=str, default="config/grpo_gsm8k.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--w-correctness", type=float, default=None)
    parser.add_argument("--w-format", type=float, default=None)
    parser.add_argument("--w-reasoning", type=float, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Number of unique prompts per GRPO step")
    parser.add_argument("--max-response-length", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--kl-coef", type=float, default=None)
    parser.add_argument("--model", type=str, default=None)
    rollout_group = parser.add_mutually_exclusive_group()
    rollout_group.add_argument("--use-vllm", action="store_true", default=None,
                               help="Use vLLM for rollout generation")
    rollout_group.add_argument("--no-vllm", action="store_true", default=None,
                               help="Use HF live policy generate instead of vLLM")
    parser.add_argument("--vllm-sync-interval", type=int, default=None,
                        help="Reload vLLM from current policy every N GRPO steps")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=None,
                        help="vLLM gpu_memory_utilization")
    parser.add_argument(
        "--reward-mode",
        choices=["legacy_reasoning", "answer_conditioned_reasoning"],
        default=None,
        help="legacy_reasoning for dense training signal; answer_conditioned_reasoning for reward-hacking diagnostics",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config["grpo_config"] = args.config

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
    if args.batch_size is not None:
        config["data"]["train_batch_size"] = args.batch_size
    if args.max_response_length is not None:
        config["data"]["max_response_length"] = args.max_response_length
    if args.total_steps is not None:
        config["trainer"]["total_training_steps"] = args.total_steps
    if args.lr is not None:
        config["actor_rollout_ref"]["actor"]["optim"]["lr"] = args.lr
    if args.kl_coef is not None:
        config["algorithm"].setdefault("kl_penalty", {})["kl_coef"] = args.kl_coef
    if args.model is not None:
        config["actor_rollout_ref"]["model"]["path"] = args.model
    rollout_cfg = config.setdefault("actor_rollout_ref", {}).setdefault("rollout", {})
    if args.use_vllm:
        rollout_cfg["use_vllm"] = True
        rollout_cfg["name"] = "vllm"
    if args.no_vllm:
        rollout_cfg["use_vllm"] = False
        rollout_cfg["name"] = "hf_policy"
    if args.vllm_sync_interval is not None:
        rollout_cfg["vllm_sync_interval"] = args.vllm_sync_interval
    if args.vllm_gpu_memory_utilization is not None:
        rollout_cfg["gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
    if args.reward_mode is not None:
        config["reward_mode"] = args.reward_mode
        config["warmup_use_legacy_reward"] = args.reward_mode == "legacy_reasoning"
        reward_kwargs = config.setdefault("custom_reward_function", {}).setdefault("reward_kwargs", {})
        reward_kwargs["answer_conditioned_reasoning"] = args.reward_mode == "answer_conditioned_reasoning"

    trainer = GRPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
