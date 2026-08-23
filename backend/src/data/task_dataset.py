from datasets import Dataset
import json
from pathlib import Path


def load_raw_dataset() -> Dataset:
    base_dir = Path(__file__).resolve().parents[2]
    path = base_dir / "data" / "raw" / "cs_examples.json"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return Dataset.from_list(data)


def load_task_dataset(dataset: Dataset, task: str) -> Dataset:
    return dataset.filter(
        lambda example: example["task"] == task
    )


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Dataset, Dataset, Dataset]:

    if train_ratio + validation_ratio >= 1.0:
        raise ValueError(
            "train_ratio + validation_ratio must be less than 1.0"
        )

    # First split: train vs remaining (validation + test)
    train_split = dataset.train_test_split(
        test_size=1.0 - train_ratio,
        seed=seed,
    )

    train_dataset = train_split["train"]
    remaining_dataset = train_split["test"]

    # Second split: validation vs test
    validation_fraction = validation_ratio / (
        1.0 - train_ratio
    )

    validation_test_split = remaining_dataset.train_test_split(
        test_size=1.0 - validation_fraction,
        seed=seed,
    )

    validation_dataset = validation_test_split["train"]
    test_dataset = validation_test_split["test"]

    return train_dataset, validation_dataset, test_dataset