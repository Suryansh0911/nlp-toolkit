import json
from collections import Counter

from datasets import Dataset
from transformers import AutoTokenizer

from src.data.formatting import (
    format_example,
    format_chat,
    tokenize_chat_example
)


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DATA_PATH = "data/raw/cs_examples.json"


# --------------------------------------------------
# 1. Load raw JSON
# --------------------------------------------------

with open(
    DATA_PATH,
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)


print(f"Loaded {len(data)} raw examples.")


# --------------------------------------------------
# 2. Convert to Hugging Face Dataset
# --------------------------------------------------

dataset = Dataset.from_list(data)

print("\nDataset:")
print(dataset)


# --------------------------------------------------
# 3. Inspect examples
# --------------------------------------------------

print("\nFirst example:")
print(dataset[0])


# --------------------------------------------------
# 4. Count tasks
# --------------------------------------------------

task_counts = Counter(dataset["task"])

print("\nTask distribution:")
print(task_counts)


# --------------------------------------------------
# 5. Separate tasks
# --------------------------------------------------

summarization_dataset = dataset.filter(
    lambda example: example["task"] == "summarization"
)

qa_dataset = dataset.filter(
    lambda example: example["task"] == "question_answering"
)

completion_dataset = dataset.filter(
    lambda example: example["task"] == "completion"
)


print("\nTask sizes:")
print("Summarization:", len(summarization_dataset))
print("QA:", len(qa_dataset))
print("Completion:", len(completion_dataset))


# --------------------------------------------------
# 6. Create train/test split
# --------------------------------------------------

# This is only for testing the pipeline.
# Our real dataset will be much larger.

split = dataset.train_test_split(
    test_size=0.2,
    seed=42
)

train_dataset = split["train"]
test_dataset = split["test"]

print("\nTrain size:", len(train_dataset))
print("Test size:", len(test_dataset))


# --------------------------------------------------
# 7. Instruction formatting
# --------------------------------------------------

formatted_dataset = dataset.map(
    format_example
)

print("\nInstruction-formatted example:")
print(formatted_dataset[0])


# --------------------------------------------------
# 8. Chat formatting
# --------------------------------------------------

chat_dataset = dataset.map(
    format_chat
)

print("\nChat-formatted example:")
print(chat_dataset[0]["messages"])


# --------------------------------------------------
# 9. Load tokenizer
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# --------------------------------------------------
# 10. Apply chat template
# --------------------------------------------------

tokenized_text_dataset = chat_dataset.map(
    lambda example: tokenize_chat_example(
        example,
        tokenizer
    )
)

print("\nFormatted model text:")
print(tokenized_text_dataset[0]["text"])