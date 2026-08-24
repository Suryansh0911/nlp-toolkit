#!/usr/bin/env python3
"""
Train all task/method combinations sequentially.
Run from backend/ directory.

Usage:
    python -m scripts.train_all
"""

import subprocess
import sys
from pathlib import Path


TRAINING_CONFIGS = [
    # Summarization
    {"task": "summarization", "method": "base", "epochs": 3},
    {"task": "summarization", "method": "lora", "epochs": 3},
    {"task": "summarization", "method": "qlora", "epochs": 3},

    # Question Answering
    {"task": "question_answering", "method": "base", "epochs": 3},
    {"task": "question_answering", "method": "lora", "epochs": 3},
    {"task": "question_answering", "method": "qlora", "epochs": 3},

    # Completion
    {"task": "completion", "method": "base", "epochs": 3},
    {"task": "completion", "method": "lora", "epochs": 3},
    {"task": "completion", "method": "qlora", "epochs": 3},
]


def main():
    print("=" * 50)
    print("NLP Toolkit - Training All Tasks")
    print("=" * 50)

    total = len(TRAINING_CONFIGS)

    for i, config in enumerate(TRAINING_CONFIGS, 1):
        task = config["task"]
        method = config["method"]
        epochs = config["epochs"]

        print(f"\n[{i}/{total}] Training {task} - {method}...")

        cmd = [
            sys.executable,
            "-m", "scripts.train",
            "--task", task,
            "--method", method,
            "--epochs", str(epochs),
        ]

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"\nError: Training failed for {task} - {method}")
            sys.exit(1)

    print("\n" + "=" * 50)
    print("All training complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
