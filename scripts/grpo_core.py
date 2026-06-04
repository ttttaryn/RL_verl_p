#!/usr/bin/env python3
"""GRPO (Group Relative Policy Optimization) 鈥?Core Algorithm Components.

GRPO is the alignment algorithm behind DeepSeek-R1-style math reasoning
training. It replaces critic-based advantage estimation with group-relative
standardization, eliminating the value model entirely.

Key properties:
  - No critic / value function
  - K responses sampled per prompt (group)
  - Advantage = (reward - group_mean) / group_std
  - KL penalty in the policy objective

Reference: DeepSeek-R1, "DeepSeekMath: Pushing the Limits of Mathematical
Reasoning in Open Language Models" (2024)

GRPO objective:
  J(胃) = E[min(r_i * 脗_i, clip(r_i, 1-蔚, 1+蔚) * 脗_i) - 尾 * D_KL(蟺||蟺_ref)]

Where:
  r_i = 蟺_胃(a_i|s) / 蟺_old(a_i|s)   (probability ratio)
  脗_i = (R_i - 渭_group) / 蟽_group     (group-relative advantage)
  尾   = KL penalty coefficient
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# 鈹€鈹€ Group-Relative Advantage 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def compute_group_relative_advantage(
    rewards: torch.Tensor,
    prompt_ids: torch.Tensor,
    eps: float = 1e-8,
    norm_method: str = "standardize",
) -> torch.Tensor:
    """Compute group-relative advantage: (R - mean) / std within each group.

    Each unique prompt forms a group. For each group of K responses,
    advantages are computed relative to the group statistics.

    Args:
        rewards: [batch_size] scalar reward for each response
        prompt_ids: [batch_size] integer ID mapping each response to its prompt
        eps: Small constant for numerical stability
        norm_method: "standardize" (z-score) or "centered" (mean-only)

    Returns:
        advantages: [batch_size] group-relative advantages
    """
    advantages = torch.zeros_like(rewards)

    for pid in prompt_ids.unique():
        mask = prompt_ids == pid
        group_rewards = rewards[mask]

        if len(group_rewards) < 2:
            # Single response in group 鈥?no relative signal
            advantages[mask] = 0.0
            continue

        group_mean = group_rewards.mean()
        group_std = group_rewards.std()

        if norm_method == "standardize":
            advantages[mask] = (group_rewards - group_mean) / (group_std + eps)
        elif norm_method == "centered":
            advantages[mask] = group_rewards - group_mean
        else:
            raise ValueError(f"Unknown norm_method: {norm_method}")

    return advantages


def compute_global_advantage(
    rewards: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute global (batch-level) relative advantage.

    Fallback when group information is not available. Standardizes
    rewards across the entire batch.
    """
    mean = rewards.mean()
    std = rewards.std()
    return (rewards - mean) / (std + eps)


# 鈹€鈹€ GRPO Loss 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def compute_grpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    kl_divergence: torch.Tensor,
    clip_ratio: float = 0.2,
    kl_coef: float = 0.001,
    loss_agg_mode: str = "seq-mean",
    response_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute sequence-level GRPO loss.

    GRPO assigns one scalar reward/advantage to each sampled completion, so the
    policy objective should operate at sequence level. We first sum log-probs
    over response tokens, then apply the group-relative advantage once per
    completion. This avoids amplifying the advantage by response length.
    """
    if response_mask is None:
        response_mask = torch.ones_like(log_probs)

    response_mask = response_mask.to(log_probs.dtype)
    token_count = response_mask.sum(dim=-1).clamp(min=1.0)

    seq_log_probs = (log_probs * response_mask).sum(dim=-1)
    seq_old_log_probs = (old_log_probs * response_mask).sum(dim=-1)
    seq_log_ratio = torch.clamp(seq_log_probs - seq_old_log_probs, min=-20.0, max=20.0)
    seq_ratio = torch.exp(seq_log_ratio)

    if advantages.dim() > 1:
        advantages = advantages.squeeze(-1)
    advantages = torch.nan_to_num(advantages, nan=0.0, posinf=5.0, neginf=-5.0).clamp(-5.0, 5.0)

    surr1 = seq_ratio * advantages
    surr2 = torch.clamp(seq_ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    token_kl = torch.nan_to_num(kl_divergence, nan=0.0, posinf=1e4, neginf=-1e4)
    seq_kl = (token_kl * response_mask).sum(dim=-1) / token_count
    kl_loss = kl_coef * seq_kl.mean()

    loss = policy_loss + kl_loss

    with torch.no_grad():
        stats = {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "ratio_mean": seq_ratio.mean().item(),
            "ratio_std": seq_ratio.std().item(),
            "ratio_clip_frac": ((seq_ratio < 1.0 - clip_ratio) | (seq_ratio > 1.0 + clip_ratio)).float().mean().item(),
            "approx_kl": (0.5 * (seq_log_ratio ** 2)).mean().item(),
            "seq_kl": seq_kl.mean().item(),
        }

    return loss, stats


# 鈹€鈹€ KL Divergence 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def compute_kl_divergence(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    estimator: str = "k1",
) -> torch.Tensor:
    """Compute per-token KL divergence between policy and reference.

    Estimators:
      k1: KL(蟺||蟺_ref) 鈮?log 蟺 - log 蟺_ref  (forward KL, recommended)
      k2: KL(蟺_ref||蟺) 鈮?log 蟺_ref - log 蟺  (reverse KL)
    """
    if estimator == "k1":
        return log_probs - ref_log_probs
    elif estimator == "k2":
        return ref_log_probs - log_probs
    elif estimator == "k3":
        log_ratio = ref_log_probs - log_probs
        return torch.exp(log_ratio) - log_ratio - 1.0
    else:
        raise ValueError(f"Unknown KL estimator: {estimator}. Use 'k1', 'k2', or 'k3'.")


def collect_group_rollout_stats(
    rewards: torch.Tensor,
    prompt_ids: torch.Tensor,
) -> Dict[str, float]:
    """Collect group-level statistics for monitoring.

    Args:
        rewards: [batch_size] scalar reward tensor
        prompt_ids: [batch_size] integer ID tensor grouping responses by prompt

    Returns:
        Dict with group_mean_reward, group_std_reward, etc.
    """
    with torch.no_grad():
        group_means = []
        group_stds = []
        group_maxs = []
        group_mins = []

        for pid in prompt_ids.unique():
            mask = prompt_ids == pid
            gr = rewards[mask]
            if len(gr) >= 2:
                group_means.append(gr.mean().item())
                group_stds.append(gr.std().item())
                group_maxs.append(gr.max().item())
                group_mins.append(gr.min().item())

        return {
            "n_groups": len(group_means),
            "avg_group_mean": sum(group_means) / len(group_means) if group_means else 0.0,
            "avg_group_std": sum(group_stds) / len(group_stds) if group_stds else 0.0,
            "avg_group_max": sum(group_maxs) / len(group_maxs) if group_maxs else 0.0,
            "avg_group_min": sum(group_mins) / len(group_mins) if group_mins else 0.0,
            "avg_reward": rewards.mean().item(),
            "std_reward": rewards.std().item(),
        }
