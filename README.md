# CS-NLP Toolkit

A domain-specific NLP toolkit for Computer Science built with Hugging Face Transformers and Parameter-Efficient Fine-Tuning (PEFT). The project adapts a shared instruction-tuned language model using task-specific LoRA and QLoRA adapters for summarization, question answering, and text completion.

## Overview

The project investigates how parameter-efficient fine-tuning can adapt a small instruction-tuned causal language model to multiple Computer Science NLP tasks without full-model fine-tuning.

## Tasks

- Text Summarization
- Question Answering
- Text Completion

## Fine-Tuning Methods

- PEFT
- LoRA
- QLoRA
- Supervised Fine-Tuning (SFT)

## Base Model

**Qwen/Qwen2.5-0.5B-Instruct**

The same base model is used across tasks, with separate adapters for each task and fine-tuning method.

```
                         Qwen2.5-0.5B-Instruct
                                  |
                         Parameter-Efficient
                           Fine-Tuning
                                  |
                    +-------------+-------------+
                    |             |             |
                  LoRA          QLoRA          Base
                    |             |             |
             +------+-----+       |       Zero-shot baseline
             |            |
        Summarization     QA      Completion
```


## How It Works

### 1. Dataset Preparation

Examples are normalized into:

```
task | input | output
```

Each task is converted into an instruction/chat representation compatible with the instruction-tuned base model.

### 2. Baseline

The untouched Qwen model is evaluated before fine-tuning. This establishes the baseline that the PEFT adapters need to beat.

### 3. LoRA

The base model is frozen and low-rank adapter parameters are trained on task-specific data.

```
Base Model
    |
    +-- frozen weights
    |
    +-- LoRA parameters
          |
        trainable
```

Separate adapters are used for:

- summarization
- question_answering
- completion

### 4. QLoRA

QLoRA combines a quantized base model with LoRA adapters to reduce the memory required for fine-tuning.

### 5. Inference

The toolkit can load a task-specific adapter on top of the shared base model.

```python
from src.inference.toolkit import NLPToolkit

toolkit = NLPToolkit()

toolkit.load_task("summarization")

result = toolkit.summarize(
    "A transformer uses self-attention to model relationships between tokens."
)

print(result)
```

Question answering:

```python
toolkit.load_task("question_answering")

answer = toolkit.answer(
    context="A stack follows the LIFO principle.",
    question="What principle does a stack follow?"
)

print(answer)
```

Completion:

```python
toolkit.load_task("completion")

result = toolkit.complete(
    "Binary search is an efficient searching algorithm that"
)

print(result)
```

## Training

The project uses one reusable training pipeline for all tasks.

**LoRA**

```bash
python -m src.training.train --task summarization --method lora

python -m src.training.train --task question_answering --method lora

python -m src.training.train --task completion --method lora
```

**QLoRA**

Run on a compatible NVIDIA GPU environment:

```bash
python -m src.training.train --task summarization --method qlora
```

Repeat for the other tasks.

## Evaluation

| Task | Metrics |
|---|---|
| Summarization | ROUGE-1, ROUGE-2, ROUGE-L |
| Question Answering | Exact Match, Token F1 |
| Completion | ROUGE-1, ROUGE-2, ROUGE-L |

Example:

```bash
python -m src.evaluation.evaluate \
    --task summarization \
    --adapter adapters/summarization/lora
```
change the task and adapter paths for qlora, kepp it blank foe base model

## Results

### Experimental Setup

| Parameter | Value |
|---|---|
| Base model | Qwen2.5-0.5B-Instruct |
| Training examples | 305 |
| Held-out evaluation examples | 50 |
| Domain | Computer Science |
| Methods | Base, LoRA, QLoRA |

The results below come from the 50-example held-out evaluation set.

### Results Summary

| Task / Metric | Base | LoRA | QLoRA |
|---|---|---|---|
| Summarization ROUGE-1 | 0.2926 | 0.3300 | 0.3187 |
| Summarization ROUGE-2 | 0.0701 | 0.1019 | 0.0884 |
| Summarization ROUGE-L | 0.1794 | 0.2078 | 0.1970 |
| QA Exact Match | 0.0000 | 0.4043 | 0.2979 |
| QA Token F1 | 0.1339 | 0.6797 | 0.5766 |
| Completion ROUGE-1 | 0.2322 | 0.5722 | 0.5110 |
| Completion ROUGE-2 | 0.1146 | 0.4038 | 0.3429 |
| Completion ROUGE-L | 0.1907 | 0.5418 | 0.4788 |

### Key Findings

- LoRA achieved the best score on every reported metric.
- QLoRA consistently outperformed the base model, while remaining below standard LoRA in this experiment.
- The largest gains appeared in question answering and text completion, indicating strong task adaptation.
- Summarization also improved, but the gain was smaller than for QA and completion.
- Because the evaluation set contains 50 examples, these figures should be treated as a development-scale evaluation rather than a large benchmark.

### Relative Improvement: LoRA vs Base

| Metric | Improvement |
|---|---|
| Summarization ROUGE-1 | +12.8% |
| Summarization ROUGE-2 | +45.4% |
| Summarization ROUGE-L | +15.8% |
| QA Token F1 | +408% |
| Completion ROUGE-1 | +146% |
| Completion ROUGE-2 | +252% |
| Completion ROUGE-L | +184% |

> For QA Exact Match, the base score is 0, so a relative percentage increase is not meaningful.

## Adapter Layout

```
adapters/
|
+-- summarization/
|   +-- lora/
|   +-- qlora/
|
+-- question_answering/
|   +-- lora/
|   +-- qlora/
|
+-- completion/
    +-- lora/
    +-- qlora/
```

The adapter-based design allows multiple specialized behaviors to share one base model without storing a separate full model for every task.

## Technology Stack

| Category | Tools |
|---|---|
| Language | Python |
| Modeling | PyTorch, Hugging Face Transformers, Hugging Face TRL, PEFT |
| Fine-tuning | LoRA, QLoRA, BitsAndBytes |
| Data | Hugging Face Datasets |
| Evaluation | Evaluate, ROUGE |
| Infrastructure | Git / GitHub, Google Colab, NVIDIA T4 |

## Development Workflow

Local development was used for data preparation, inference, debugging, and pipeline development. GPU-intensive fine-tuning was moved to Google Colab with an NVIDIA T4.

```
Local development
      |
      v
Private GitHub repository
      |
      v
Google Colab + NVIDIA T4
      |
      +-- LoRA training
      +-- QLoRA training
      |
      v
Evaluation
      |
      v
Results / README
```

## Usage

**Clone**

```bash
git clone <YOUR_REPOSITORY_URL>
cd nlp-toolkit/backend
```

**Install**

```bash
pip install -r requirements.txt
```

**Train**

```bash
python -m src.training.train \
    --task summarization \
    --method lora
```

**Evaluate**

```bash
python -m src.evaluation.evaluate \
    --task summarization \
    --adapter adapters/summarization/lora
```

## Future Improvements

- Expand the Computer Science training and evaluation datasets
- Use formal train/validation/test splits at larger scale
- Add code explanation and code generation
- Add experiment tracking
- Compare additional base models
- Systematically tune LoRA rank and target modules
- Publish trained adapters to the Hugging Face Hub
- Add a FastAPI inference service
- Add a lightweight web interface

## Notes

The current reported evaluation uses 305 training examples and 50 held-out examples. The 15-example dataset used during early local development was a pipeline-validation dataset and is not the basis for the final reported 305/50 experiment.