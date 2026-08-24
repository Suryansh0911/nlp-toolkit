#!/bin/bash
# Train completion task with LoRA (Low-Rank Adaptation)

python src/training/train.py \
    --task completion \
    --method lora \
    --epochs 3 \
    --output_dir adapters/completion/lora
