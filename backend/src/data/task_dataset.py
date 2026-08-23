from datasets import Dataset
import json

def load_raw_dataset():
    base_dir = Path(__file__).resolve().parents[2]
    path = base_dir / "data" / "raw" / "cs_examples.json"

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_task_dataset(dataset : Dataset, task: str) -> Dataset:
    return dataset.filter(lambda example: example["task"] == task)