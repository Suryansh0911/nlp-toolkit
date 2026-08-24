#!/bin/bash
# Train question answering task with base fine-tuning (full parameter update)

python src/training/train.py \
    --task question_answering \
    --method base \
    --epochs 3 \
    --output_dir adapters/question_answering/base
