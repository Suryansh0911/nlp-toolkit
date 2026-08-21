from datasets import Dataset
def load_task_dataset(data, task):

    filtered = [example for example in data if example["task"] == task]
    return Dataset.from_list(filtered)