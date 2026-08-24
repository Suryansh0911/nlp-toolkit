#!/bin/bash
# Train summarization task with base fine-tuning (full parameter update)

python src/training/train.py \
    --task summarization \
    --method base \
    --epochs 3 \
    --output_dir adapters/summarization/base
