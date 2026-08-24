#!/bin/bash
# Master script to train all tasks with all methods
# Run on Google Colab with GPU for QLoRA
# Run from backend/ directory

echo "=========================================="
echo "NLP Toolkit - Training All Tasks"
echo "=========================================="

python -m scripts.train_all

echo -e "\n=========================================="
echo "All training complete!"
echo "=========================================="
