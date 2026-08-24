#!/bin/bash
# Train summarization task with QLoRA (Quantized Low-Rank Adaptation)
# Requires GPU with CUDA support

python src/training/train.py \
    --task summarization \
    --method qlora \
    --epochs 3 \
    --output_dir adapters/summarization/qlora
