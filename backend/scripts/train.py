#!/usr/bin/env python3
"""
Master training script for all task/method combinations.
Run from backend/ directory.

Usage:
    python -m scripts.train --task summarization --method lora
    python -m scripts.train --task question_answering --method qlora --epochs 3
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train import main as train_main


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train NLP model for a specific task and method."
    )

    parser.add_argument(
        "--task",
        required=True,
        choices=["summarization", "question_answering", "completion"],
        help="Task to train on"
    )

    parser.add_argument(
        "--method",
        default="lora",
        choices=["base", "lora", "qlora"],
        help="Training method (default: lora)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help="Custom output directory (optional)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Build sys.argv for train.py
    train_args = [
        "--task", args.task,
        "--method", args.method,
        "--epochs", str(args.epochs),
    ]

    if args.output_dir:
        train_args.extend(["--output_dir", args.output_dir])

    # Override sys.argv and run training
    sys.argv = ["train.py"] + train_args
    train_main()
