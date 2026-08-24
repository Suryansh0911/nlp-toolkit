# NLP Toolkit - LoRA/QLoRA Fine-Tuning

A fine-tuning project for training language models on Computer Science tasks using LoRA and QLoRA techniques.

## Overview

This project fine-tunes the **Qwen2.5-0.5B-Instruct** model on three CS-focused tasks:

- **Summarization** - Summarizing CS research papers and technical content
- **Question Answering** - Answering CS questions from MMLU
- **Completion** - Code and text completion from CodeAlpaca

## Project Structure

```
nlp-toolkit/
├── backend/
│   ├── data/
│   │   └── raw/
│   │       └── cs_examples_unified.json    # Training dataset (657 examples)
│   ├── scripts/
│   │   └── build_unified_dataset.py        # Dataset creation script
│   ├── src/
│   │   ├── data/                           # Data loading & formatting
│   │   ├── evaluation/                     # Metrics (ROUGE, F1, EM)
│   │   ├── inference/                      # Model inference & adapters
│   │   └── training/                       # LoRA/QLoRA training scripts
│   └── tests/
│       └── smoke_test.py
├── requirements.txt
└── README.md
```

## Training

### On Google Colab

1. **Clone the repository:**
   ```bash
   !git clone <your-repo-url>
   %cd nlp-toolkit/backend
   ```

2. **Install dependencies:**
   ```bash
   !pip install -r ../requirements.txt
   ```

3. **Run training:**
   ```bash
   # LoRA training
   !python src/training/train.py --task summarization --method lora --epochs 3

   # QLoRA training (requires GPU)
   !python src/training/train.py --task summarization --method qlora --epochs 3
   ```

### Training Arguments

- `--task`: One of `summarization`, `question_answering`, `completion`
- `--method`: `lora` (default) or `qlora` (4-bit quantization)
- `--epochs`: Number of training epochs (default: 1)
- `--output_dir`: Custom output directory (optional)

## LoRA Configuration

Default LoRA settings (in `src/training/lora_config.py`):

| Parameter | Value |
|-----------|-------|
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target Modules | q_proj, k_proj, v_proj, o_proj |

## Dataset

The unified dataset (`cs_examples_unified.json`) contains:

| Task | Examples |
|------|----------|
| Summarization | 305 |
| Completion | 305 |
| Question Answering | 47 |
| **Total** | **657** |

Sources:
- Summarization: ccdv/arxiv-summarization
- QA: MMLU CS subjects
- Completion: flwrlabs/code-alpaca-20k

## Evaluation

```bash
python src/evaluation/evaluate.py --task summarization --adapter adapters/summarization/lora
```

## Inference

```python
from src.inference.toolkit import NLPToolkit

toolkit = NLPToolkit()
toolkit.load_task("summarization", method="lora")

summary = toolkit.summarize("Your CS text here...")
```

## Requirements

- Python >= 3.11
- PyTorch
- Transformers >= 5.15.0
- PEFT >= 0.20.0
- TRL >= 1.10.0
- bitsandbytes (for QLoRA)

See `requirements.txt` for full dependencies.

## License

MIT
