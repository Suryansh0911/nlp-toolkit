from src.data.task_dataset import load_raw_dataset
from src.data.formatting import format_chat, format_inference_messages

dataset = load_raw_dataset()

assert len(dataset) > 0

tasks = set(
    dataset["task"]
)

assert "summarization" in tasks
assert "question_answering" in tasks
assert "completion" in tasks


example = dataset[0]

chat = format_chat(
    example
)

assert len(
    chat["messages"]
) == 3


messages = (
    format_inference_messages(
        example
    )
)

assert len(messages) == 2

assert messages[0]["role"] == "system"
assert messages[1]["role"] == "user"

print(
    "All smoke tests passed."
)