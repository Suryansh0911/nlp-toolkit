#!/bin/bash
# Master script to train all tasks with all methods
# Run on Google Colab with GPU for QLoRA
# Run from backend/ directory: bash scripts/train_all.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "NLP Toolkit - Training All Tasks"
echo "=========================================="

# Summarization
echo -e "\n[1/9] Training Summarization - Base..."
bash "$SCRIPT_DIR/train_summarization_base.sh"

echo -e "\n[2/9] Training Summarization - LoRA..."
bash "$SCRIPT_DIR/train_summarization_lora.sh"

echo -e "\n[3/9] Training Summarization - QLoRA..."
bash "$SCRIPT_DIR/train_summarization_qlora.sh"

# Question Answering
echo -e "\n[4/9] Training Question Answering - Base..."
bash "$SCRIPT_DIR/train_qa_base.sh"

echo -e "\n[5/9] Training Question Answering - LoRA..."
bash "$SCRIPT_DIR/train_qa_lora.sh"

echo -e "\n[6/9] Training Question Answering - QLoRA..."
bash "$SCRIPT_DIR/train_qa_qlora.sh"

# Completion
echo -e "\n[7/9] Training Completion - Base..."
bash "$SCRIPT_DIR/train_completion_base.sh"

echo -e "\n[8/9] Training Completion - LoRA..."
bash "$SCRIPT_DIR/train_completion_lora.sh"

echo -e "\n[9/9] Training Completion - QLoRA..."
bash "$SCRIPT_DIR/train_completion_qlora.sh"

echo -e "\n=========================================="
echo "All training complete!"
echo "=========================================="
