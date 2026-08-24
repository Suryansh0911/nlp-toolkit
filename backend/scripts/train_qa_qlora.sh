#!/bin/bash
# Train question answering task with QLoRA (Quantized Low-Rank Adaptation)
# Requires GPU with CUDA support

python src/training/train.py \
    --task question_answering \
    --method qlora \
    --epochs 3 \
    --output_dir adapters/question_answering/qlora
