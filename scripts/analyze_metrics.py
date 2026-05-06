#!/usr/bin/env python3
# Copyright 2024 Custom Implementation
# Metrics Analysis and Visualization Tool
#
# This script provides utilities for analyzing and comparing experiments.

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


class MetricsLogger:
    """Logger for recording detailed metrics during training.
    
    This class can be used alongside verl's built-in logging to record
    additional metrics for later analysis.
    """
    
    def __init__(self, log_dir: str, experiment_name: str):
        """Initialize the metrics logger.
        
        Args:
            log_dir: Directory to save log files
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}_metrics.jsonl"
        
        # Create directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics_history = defaultdict(list)
        
    def log(self, step: int, metrics: Dict[str, Any]):
        """Log metrics for a training step.
        
        Args:
            step: Training step number
            metrics: Dictionary of metrics to log
        """
        # Add timestamp and step
        record = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            **metrics,
        }
        
        # Store in memory
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.metrics_history[key].append((step, value))
        
        # Append to file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
    
    def get_metric(self, name: str) -> List[tuple]:
        """Get history of a specific metric.
        
        Args:
            name: Name of the metric
            
        Returns:
            List of (step, value) tuples
        """
        return self.metrics_history.get(name, [])
    
    def save_summary(self):
        """Save a summary of all metrics."""
        summary = {}
        for key, values in self.metrics_history.items():
            if values:
                vals = [v for _, v in values]
                summary[key] = {
                    "mean": np.mean(vals),
                    "std": np.std(vals),
                    "min": np.min(vals),
                    "max": np.max(vals),
                    "final": vals[-1],
                }
        
        summary_file = self.log_dir / f"{self.experiment_name}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)


def load_metrics_from_jsonl(file_path: str) -> Dict[str, List[tuple]]:
    """Load metrics from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        Dictionary mapping metric names to lists of (step, value) tuples
    """
    metrics = defaultdict(list)
    
    with open(file_path, "r") as f:
        for line in f:
            record = json.loads(line.strip())
            step = record.get("step", 0)
            
            for key, value in record.items():
                if key not in ["timestamp", "step"] and isinstance(value, (int, float)):
                    metrics[key].append((step, value))
    
    return dict(metrics)


def plot_metric_comparison(
    experiment_metrics: Dict[str, Dict[str, List[tuple]]],
    metric_name: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: str = "Training Step",
    ylabel: Optional[str] = None,
    smoothing: float = 0.0,
):
    """Plot comparison of a metric across experiments.
    
    Args:
        experiment_metrics: Dictionary mapping experiment names to their metrics
        metric_name: Name of the metric to plot
        output_path: Path to save the plot (optional)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        smoothing: Exponential smoothing factor (0 = no smoothing)
    """
    plt.figure(figsize=(10, 6))
    
    for exp_name, metrics in experiment_metrics.items():
        if metric_name not in metrics:
            print(f"Warning: Metric '{metric_name}' not found in experiment '{exp_name}'")
            continue
        
        data = metrics[metric_name]
        steps = [d[0] for d in data]
        values = [d[1] for d in data]
        
        # Apply smoothing if requested
        if smoothing > 0:
            smoothed = []
            current = values[0]
            for v in values:
                current = smoothing * current + (1 - smoothing) * v
                smoothed.append(current)
            values = smoothed
        
        plt.plot(steps, values, label=exp_name)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel or metric_name)
    plt.title(title or f"Comparison of {metric_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_multi_reward_breakdown(
    metrics: Dict[str, List[tuple]],
    output_path: Optional[str] = None,
    title: str = "Multi-Reward Breakdown",
):
    """Plot breakdown of different reward components.
    
    Args:
        metrics: Dictionary of metrics
        output_path: Path to save the plot
        title: Plot title
    """
    reward_keys = [
        "correctness_score",
        "format_score", 
        "reasoning_score",
        "score",  # total score
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]
    titles = ["Correctness Reward", "Format Reward", "Reasoning Reward", "Total Reward"]
    
    for ax, key, color, subplot_title in zip(axes, reward_keys, colors, titles):
        if key in metrics:
            data = metrics[key]
            steps = [d[0] for d in data]
            values = [d[1] for d in data]
            
            ax.plot(steps, values, color=color, alpha=0.7)
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Reward")
            ax.set_title(subplot_title)
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_kl_entropy_analysis(
    metrics: Dict[str, List[tuple]],
    output_path: Optional[str] = None,
    title: str = "KL & Entropy Analysis",
):
    """Plot KL divergence and entropy metrics.
    
    Args:
        metrics: Dictionary of metrics
        output_path: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # KL divergence
    kl_keys = ["kl_divergence", "kl", "critic/kl_mean"]
    for key in kl_keys:
        if key in metrics:
            data = metrics[key]
            steps = [d[0] for d in data]
            values = [d[1] for d in data]
            axes[0].plot(steps, values, label=key)
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("KL Divergence")
    axes[0].set_title("KL Divergence")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Entropy
    entropy_keys = ["entropy", "actor/entropy_mean", "policy_entropy"]
    for key in entropy_keys:
        if key in metrics:
            data = metrics[key]
            steps = [d[0] for d in data]
            values = [d[1] for d in data]
            axes[1].plot(steps, values, label=key)
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Entropy")
    axes[1].set_title("Policy Entropy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Advantage
    adv_keys = ["advantage", "critic/advantages/mean"]
    for key in adv_keys:
        if key in metrics:
            data = metrics[key]
            steps = [d[0] for d in data]
            values = [d[1] for d in data]
            axes[2].plot(steps, values, label=key)
    axes[2].set_xlabel("Training Step")
    axes[2].set_ylabel("Advantage")
    axes[2].set_title("Advantage Estimation")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_experiment_report(
    experiment_dir: str,
    output_dir: str,
):
    """Generate a comprehensive report for an experiment.
    
    Args:
        experiment_dir: Directory containing experiment logs
        output_dir: Directory to save the report
    """
    exp_path = Path(experiment_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Find all metrics files
    metrics_files = list(exp_path.glob("*_metrics.jsonl"))
    
    if not metrics_files:
        print(f"No metrics files found in {experiment_dir}")
        return
    
    all_metrics = {}
    for f in metrics_files:
        exp_name = f.stem.replace("_metrics", "")
        all_metrics[exp_name] = load_metrics_from_jsonl(str(f))
    
    # Generate comparison plots
    common_metrics = ["score", "is_correct", "response_length"]
    
    for metric in common_metrics:
        plot_metric_comparison(
            all_metrics,
            metric,
            output_path=str(out_path / f"comparison_{metric}.png"),
            smoothing=0.9,
        )
    
    # Generate individual experiment reports
    for exp_name, metrics in all_metrics.items():
        plot_multi_reward_breakdown(
            metrics,
            output_path=str(out_path / f"{exp_name}_reward_breakdown.png"),
            title=f"Reward Breakdown: {exp_name}",
        )
        
        plot_kl_entropy_analysis(
            metrics,
            output_path=str(out_path / f"{exp_name}_kl_entropy.png"),
            title=f"KL & Entropy: {exp_name}",
        )
    
    print(f"Report generated in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize training metrics")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate experiment report")
    report_parser.add_argument("--experiment-dir", required=True, help="Directory containing experiment logs")
    report_parser.add_argument("--output-dir", required=True, help="Directory to save the report")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare specific metrics")
    compare_parser.add_argument("--files", nargs="+", required=True, help="Metrics files to compare")
    compare_parser.add_argument("--metric", required=True, help="Metric to compare")
    compare_parser.add_argument("--output", help="Output file path")
    compare_parser.add_argument("--smoothing", type=float, default=0.0, help="Smoothing factor")
    
    args = parser.parse_args()
    
    if args.command == "report":
        generate_experiment_report(args.experiment_dir, args.output_dir)
    
    elif args.command == "compare":
        all_metrics = {}
        for f in args.files:
            exp_name = Path(f).stem.replace("_metrics", "")
            all_metrics[exp_name] = load_metrics_from_jsonl(f)
        
        plot_metric_comparison(
            all_metrics,
            args.metric,
            output_path=args.output,
            smoothing=args.smoothing,
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
