#!/bin/bash
# Train completion task with base fine-tuning (full parameter update)

python src/training/train.py \
    --task completion \
    --method base \
    --epochs 3 \
    --output_dir adapters/completion/base
