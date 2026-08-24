#!/bin/bash
# Train question answering task with LoRA (Low-Rank Adaptation)

python src/training/train.py \
    --task question_answering \
    --method lora \
    --epochs 3 \
    --output_dir adapters/question_answering/lora
