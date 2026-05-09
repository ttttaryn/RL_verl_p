#!/usr/bin/env python3
"""GRPO (Group Relative Policy Optimization) — Core Algorithm Components.

GRPO is the alignment algorithm behind DeepSeek-R1. It replaces PPO's
critic-based GAE advantage with group-relative standardization, eliminating
the value model entirely (~50% VRAM savings).

Key differences from PPO:
  - No critic / value function
  - K responses sampled per prompt (group)
  - Advantage = (reward - group_mean) / group_std
  - KL penalty baked into the reward (no separate KL controller)

Reference: DeepSeek-R1, "DeepSeekMath: Pushing the Limits of Mathematical
Reasoning in Open Language Models" (2024)

GRPO objective:
  J(θ) = E[min(r_i * Â_i, clip(r_i, 1-ε, 1+ε) * Â_i) - β * D_KL(π||π_ref)]

Where:
  r_i = π_θ(a_i|s) / π_old(a_i|s)   (probability ratio)
  Â_i = (R_i - μ_group) / σ_group     (group-relative advantage)
  β   = KL penalty coefficient
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ── Group-Relative Advantage ──────────────────────────────────────────────

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
            # Single response in group — no relative signal
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


# ── GRPO Loss ─────────────────────────────────────────────────────────────

def compute_grpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    kl_divergence: torch.Tensor,
    clip_ratio: float = 0.2,
    kl_coef: float = 0.001,
    loss_agg_mode: str = "token-mean",
    response_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the GRPO policy loss.

    Args:
        log_probs: [batch, seq_len] current policy log probs
        old_log_probs: [batch, seq_len] old policy log probs (from rollout)
        advantages: [batch, 1] or [batch] group-relative advantages
        kl_divergence: [batch, seq_len] per-token KL from reference model
        clip_ratio: ε for probability ratio clipping
        kl_coef: β weight for KL penalty
        loss_agg_mode: how to aggregate across tokens/sequences
        response_mask: [batch, seq_len] 1 for response tokens, 0 for padding

    Returns:
        loss: scalar GRPO loss
        stats: dict with diagnostic values
    """
    # Probability ratio: π_θ / π_old
    log_ratio = log_probs - old_log_probs
    ratio = torch.exp(log_ratio)  # r_i(θ)

    # Ensure advantages broadcast correctly
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1)  # [batch, 1]

    # Clipped surrogate objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -torch.min(surr1, surr2)  # negative because we minimize

    # KL penalty (added to the loss — GRPO bakes KL into the objective)
    kl_loss = kl_coef * kl_divergence

    # Per-token loss
    per_token_loss = policy_loss + kl_loss

    # Aggregate loss
    if response_mask is not None:
        if loss_agg_mode == "token-mean":
            loss = (per_token_loss * response_mask).sum() / response_mask.sum().clamp(min=1)
        elif loss_agg_mode == "token-sum":
            loss = (per_token_loss * response_mask).sum()
        elif loss_agg_mode == "seq-mean":
            seq_loss = (per_token_loss * response_mask).sum(dim=-1)
            loss = seq_loss.mean()
        else:
            raise ValueError(f"Unknown loss_agg_mode: {loss_agg_mode}")
    else:
        loss = per_token_loss.mean()

    # Diagnostic statistics
    with torch.no_grad():
        stats = {
            "loss": loss.item(),
            "policy_loss": policy_loss.mean().item(),
            "kl_loss": kl_loss.mean().item(),
            "ratio_mean": ratio.mean().item(),
            "ratio_std": ratio.std().item(),
            "ratio_clip_frac": ((ratio < 1.0 - clip_ratio) | (ratio > 1.0 + clip_ratio)).float().mean().item(),
            "approx_kl": (0.5 * (log_ratio ** 2)).mean().item(),
        }

    return loss, stats


# ── KL Divergence ─────────────────────────────────────────────────────────

def compute_kl_divergence(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    estimator: str = "k1",
) -> torch.Tensor:
    """Compute per-token KL divergence between policy and reference.

    Estimators:
      k1: KL(π||π_ref) ≈ log π - log π_ref  (forward KL, recommended)
      k2: KL(π_ref||π) ≈ log π_ref - log π  (reverse KL)
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
