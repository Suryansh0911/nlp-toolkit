#!/bin/bash
# Train summarization task with LoRA (Low-Rank Adaptation)

python src/training/train.py \
    --task summarization \
    --method lora \
    --epochs 3 \
    --output_dir adapters/summarization/lora
