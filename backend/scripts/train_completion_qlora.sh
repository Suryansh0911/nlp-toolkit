#!/bin/bash
# Train completion task with QLoRA (Quantized Low-Rank Adaptation)
# Requires GPU with CUDA support

python src/training/train.py \
    --task completion \
    --method qlora \
    --epochs 3 \
    --output_dir adapters/completion/qlora
