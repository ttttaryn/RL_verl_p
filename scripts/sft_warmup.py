#!/usr/bin/env python3
"""SFT Warmup — establish a proper baseline before RL training.

Problem: Qwen2.5-0.5B-Instruct has ~6% GSM8K accuracy. RL training
boosts this to ~50%, but most of the gain comes from learning the
output format (27% → 97% format score), not math reasoning.

This script performs supervised fine-tuning on GSM8K training data
to establish a reasonable math-capable baseline BEFORE applying RL.
The SFT model learns BOTH math reasoning AND output format from
demonstrations, so the subsequent RL phase can focus on improving
reasoning quality rather than format compliance.

Usage:
    python scripts/sft_warmup.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --train-data ~/data/gsm8k/train.parquet \
        --output ./checkpoints/sft_warmup \
        --epochs 3 \
        --lr 2e-5
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)


def load_gsm8k_data(parquet_path: str) -> Dataset:
    """Load GSM8K data from parquet and format for SFT."""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    dataset = Dataset.from_pandas(df)

    def extract_prompt(prompt_data) -> str:
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

    def extract_ground_truth(example) -> str:
        reward_model = example.get("reward_model", {})
        if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
            return str(reward_model["ground_truth"])
        return str(example.get("ground_truth", example.get("answer", "")))

    def format_sft_example(example):
        """Create a properly formatted SFT example with step-by-step reasoning."""
        prompt = extract_prompt(example.get("prompt", example.get("question", "")))
        answer = extract_ground_truth(example)

        # Build a demonstration with proper step-by-step structure
        # The prompt already includes "Let's think step by step and output
        # the final answer after ####"
        response = f" {answer}"

        # If we have a full solution (some GSM8K versions include this)
        solution = example.get("solution", example.get("answer_detail", None))
        if solution and len(str(solution)) > 10:
            response = f" {solution}\n#### {answer}"

        return {"prompt": prompt, "response": response}

    return dataset.map(format_sft_example)


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int = 1024):
    """Tokenize the dataset and mask prompt tokens from the loss."""

    def tokenize(example):
        prompt_ids = tokenizer(
            example["prompt"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )["input_ids"]
        result = tokenizer(
            example["prompt"] + example["response"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        labels = result["input_ids"].copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        result["labels"] = labels
        return result

    return dataset.map(tokenize, remove_columns=dataset.column_names)


def main():
    parser = argparse.ArgumentParser(description="SFT warmup for GSM8K baseline")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-data", type=str, required=True,
                        help="Path to GSM8K train.parquet")
    parser.add_argument("--val-data", type=str, default=None,
                        help="Path to GSM8K test.parquet")
    parser.add_argument("--output", type=str, default="./checkpoints/sft_warmup")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--fp16", action="store_true", default=True)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
        device_map="auto",
    )

    print(f"Loading data: {args.train_data}")
    dataset = load_gsm8k_data(args.train_data)
    print(f"  Train samples: {len(dataset)}")

    tokenized = tokenize_dataset(dataset, tokenizer, max_length=args.max_length)

    val_dataset = None
    if args.val_data:
        val_data = load_gsm8k_data(args.val_data)
        val_dataset = tokenize_dataset(val_data, tokenizer, max_length=args.max_length)
        print(f"  Val samples: {len(val_data)}")

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=args.save_steps,
        eval_steps=args.save_steps if val_dataset else None,
        eval_strategy="steps" if val_dataset else "no",
        save_total_limit=3,
        fp16=args.fp16,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
        ),
    )

    print(f"Starting SFT training ({args.epochs} epochs, lr={args.lr})...")
    trainer.train()

    # Save the final model
    final_path = os.path.join(args.output, "final")
    print(f"Saving SFT model to {final_path}")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # Save a metadata file
    metadata = {
        "base_model": args.model,
        "train_data": args.train_data,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "description": "SFT warmup baseline for GSM8K multi-reward RL training",
    }
    with open(os.path.join(final_path, "sft_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("SFT warmup complete. Use this model as actor_rollout_ref.model.path")
    print(f"  in your RL config: model_path={final_path}")


if __name__ == "__main__":
    main()
