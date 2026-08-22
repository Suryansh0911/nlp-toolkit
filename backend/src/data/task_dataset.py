from datasets import Dataset
import json

def load_raw_dataset(path: str = "C:/Users/surya/OneDrive/Desktop/nlp-toolkit/backend/data/raw/cs_examples.json") -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def load_task_dataset(dataset : Dataset, task: str) -> Dataset:
    return dataset.filter(lambda example: example["task"] == task)