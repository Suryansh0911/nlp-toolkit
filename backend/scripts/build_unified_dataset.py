import json
import random
from pathlib import Path

from datasets import load_dataset


SEED = 42

# Target number of examples from each task.
SUMMARIZATION_TARGET = 300
QA_TARGET = 300
COMPLETION_TARGET = 300

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"

LOCAL_DATASET = RAW_DIR / "cs_examples.json"

OUTPUT_DATASET = RAW_DIR / "cs_examples_unified.json"


def clean_text(text):
    """
    Normalize whitespace while preserving the actual content.
    """
    if text is None:
        return ""

    return " ".join(str(text).split()).strip()


def make_key(example):
    """
    Used for exact duplicate detection.
    """
    return (
        example["task"],
        clean_text(example["input"]).lower(),
        clean_text(example["output"]).lower(),
    )


def load_local_examples():
    """
    Load your existing manually-created examples.
    """
    if not LOCAL_DATASET.exists():
        print("No local dataset found.")
        return []

    with open(LOCAL_DATASET, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded local examples: {len(data)}")

    return data


# ============================================================
# SUMMARIZATION
# ============================================================

def build_summarization_dataset(target_size):
    """
    Build technical summarization examples from the native Parquet
    version of ccdv/arxiv-summarization.

    We intentionally load only one training shard for the first
    dataset build. The full dataset contains ~203k training examples,
    which is unnecessary for our initial 300-example target.
    """

    print("\nLoading technical summarization dataset...")

    dataset = load_dataset(
        "parquet",
        data_files={
            "train": (
                "https://huggingface.co/datasets/"
                "ccdv/arxiv-summarization/resolve/main/"
                "document/train-00000-of-00015.parquet"
            )
        },
        split="train",
    )

    print(f"Loaded summarization source: {len(dataset)} examples")

    candidates = []

    for example in dataset:

        article = clean_text(
            example.get("article", "")
        )

        abstract = clean_text(
            example.get("abstract", "")
        )

        if not article or not abstract:
            continue

        # Our current SFT configuration uses max_length=512.
        # Keep examples reasonably sized.
        if len(article) > 12000:
            continue

        if len(abstract) > 2000:
            continue

        candidates.append(
            {
                "task": "summarization",
                "input": article,
                "output": abstract,
            }
        )

    return sample_examples(
        candidates,
        target_size,
        seed=SEED,
    )

# ============================================================
# QUESTION ANSWERING
# ============================================================

def build_qa_dataset(target_size):
    """
    Build CS-oriented QA examples from MMLU.

    MMLU is multiple-choice QA, so we preserve the question and
    choices in the input and convert the correct choice into the
    output text.

    We intentionally DO NOT use MMLU test data.
    """

    print("\nLoading MMLU...")

    subjects = [
        "college_computer_science",
        "high_school_computer_science",
        "machine_learning",
        "computer_security",
    ]

    candidates = []

    for subject in subjects:

        print(f"  Loading: {subject}")

        dataset = load_dataset(
            "cais/mmlu",
            subject,
            split="validation",
        )

        for example in dataset:

            question = clean_text(example["question"])
            choices = example["choices"]

            answer_index = int(example["answer"])

            if not question:
                continue

            if not choices:
                continue

            if answer_index < 0 or answer_index >= len(choices):
                continue

            formatted_choices = []

            for index, choice in enumerate(choices):

                choice = clean_text(choice)

                formatted_choices.append(
                    f"{chr(65 + index)}. {choice}"
                )

            answer = clean_text(
                choices[answer_index]
            )

            input_text = (
                f"Question: {question}\n"
                f"Choices:\n"
                + "\n".join(formatted_choices)
            )

            candidates.append(
                {
                    "task": "question_answering",
                    "input": input_text,
                    "output": answer,
                }
            )

    return sample_examples(
        candidates,
        target_size,
        seed=SEED + 1,
    )


# ============================================================
# COMPLETION
# ============================================================

def build_completion_dataset(target_size):
    """
    Build completion/code-generation examples from CodeAlpaca.

    CodeAlpaca uses:

        instruction
        input
        output

    We convert these into:

        input -> output

    for our unified dataset.
    """

    print("\nLoading CodeAlpaca...")

    dataset = load_dataset(
        "flwrlabs/code-alpaca-20k",
        split="train",
    )

    candidates = []

    for example in dataset:

        instruction = clean_text(
            example.get("instruction", "")
        )

        context = clean_text(
            example.get("input", "")
        )

        output = clean_text(
            example.get("output", "")
        )

        if not instruction:
            continue

        if not output:
            continue

        if context:

            input_text = (
                f"{instruction}\n"
                f"Input: {context}"
            )

        else:

            input_text = instruction

        # Avoid extremely large code outputs.
        if len(input_text) > 6000:
            continue

        if len(output) > 6000:
            continue

        candidates.append(
            {
                "task": "completion",
                "input": input_text,
                "output": output,
            }
        )

    return sample_examples(
        candidates,
        target_size,
        seed=SEED + 2,
    )


# ============================================================
# SAMPLING / DEDUPLICATION
# ============================================================

def sample_examples(examples, target_size, seed):
    """
    Shuffle, remove exact duplicates, and select target_size examples.
    """

    rng = random.Random(seed)

    examples = list(examples)

    rng.shuffle(examples)

    selected = []

    seen = set()

    for example in examples:

        key = make_key(example)

        if key in seen:
            continue

        seen.add(key)

        selected.append(example)

        if len(selected) >= target_size:
            break

    print(
        f"Selected {len(selected)} / "
        f"{target_size} requested examples"
    )

    return selected


# ============================================================
# MERGING
# ============================================================

def merge_datasets(
    local_examples,
    summarization,
    qa,
    completion,
):
    """
    Merge all sources while removing exact duplicates.

    Local examples are preserved first.
    """

    all_examples = []

    seen = set()

    sources = [
        ("local", local_examples),
        ("summarization", summarization),
        ("question_answering", qa),
        ("completion", completion),
    ]

    for source_name, examples in sources:

        for example in examples:

            task = example.get("task")
            input_text = example.get("input")
            output_text = example.get("output")

            if not task or not input_text or not output_text:
                continue

            normalized = {
                "task": task,
                "input": clean_text(input_text),
                "output": clean_text(output_text),
            }

            key = make_key(normalized)

            if key in seen:
                continue

            seen.add(key)

            all_examples.append(normalized)

    return all_examples


# ============================================================
# VALIDATION
# ============================================================

def validate_dataset(dataset):
    """
    Basic structural validation.
    """

    required_fields = {
        "task",
        "input",
        "output",
    }

    valid_tasks = {
        "summarization",
        "question_answering",
        "completion",
    }

    for index, example in enumerate(dataset):

        if set(example.keys()) != required_fields:

            raise ValueError(
                f"Example {index} has invalid fields: "
                f"{example.keys()}"
            )

        if example["task"] not in valid_tasks:

            raise ValueError(
                f"Example {index} has invalid task: "
                f"{example['task']}"
            )

        if not example["input"].strip():

            raise ValueError(
                f"Example {index} has empty input."
            )

        if not example["output"].strip():

            raise ValueError(
                f"Example {index} has empty output."
            )


# ============================================================
# STATISTICS
# ============================================================

def print_statistics(dataset):

    counts = {}

    for example in dataset:

        task = example["task"]

        counts[task] = counts.get(task, 0) + 1

    print("\n==============================")
    print("DATASET STATISTICS")
    print("==============================")

    print(f"Total examples: {len(dataset)}")

    for task, count in sorted(counts.items()):

        print(f"{task}: {count}")

    print("==============================\n")


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print("CS-NLP UNIFIED DATASET BUILDER")
    print("========================================")

    random.seed(SEED)

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Existing examples
    # --------------------------------------------------------

    local_examples = load_local_examples()

    # --------------------------------------------------------
    # Public datasets
    # --------------------------------------------------------

    summarization = build_summarization_dataset(
        SUMMARIZATION_TARGET
    )

    qa = build_qa_dataset(
        QA_TARGET
    )

    completion = build_completion_dataset(
        COMPLETION_TARGET
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    dataset = merge_datasets(
        local_examples,
        summarization,
        qa,
        completion,
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_dataset(dataset)

    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    random.Random(SEED).shuffle(dataset)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(
        OUTPUT_DATASET,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            dataset,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print_statistics(dataset)

    print(
        f"Dataset saved to:\n"
        f"{OUTPUT_DATASET}"
    )


if __name__ == "__main__":
    main()