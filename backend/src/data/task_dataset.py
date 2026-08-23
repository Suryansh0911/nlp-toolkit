from datasets import Dataset
import json
from pathlib import Path
def load_raw_dataset():
    base_dir = Path(__file__).resolve().parents[2]
    path = base_dir / "data" / "raw" / "cs_examples.json"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def load_task_dataset(dataset : Dataset, task: str) -> Dataset:
    return dataset.filter(lambda example: example["task"] == task)