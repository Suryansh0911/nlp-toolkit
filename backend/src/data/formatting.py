def format_example(example):

    task = example["task"]
    user_input = example["input"]
    output = example["output"]

    if task == "summarization":
        return {
            "instruction": (
                "Summarize the following Computer Science text."
            ),
            "input": user_input,
            "output": output
        }

    elif task == "question_answering":
        return {
            "instruction": (
                "Answer the question using the provided context."
            ),
            "input": user_input,
            "output": output
        }

    elif task == "completion":
        return {
            "instruction": (
                "Complete the following Computer Science text."
            ),
            "input": user_input,
            "output": output
        }

    else:
        raise ValueError(f"Unknown task: {task}")

def format_chat(example):

    task = example["task"]

    system_message = (
        "You are an expert Computer Science assistant."
    )

    if task == "summarization":
        user_message = (
            "Summarize the following Computer Science text:\n\n"
            + example["input"]
        )

    elif task == "question_answering":
        user_message = (
            "Answer the question using the provided context.\n\n"
            + example["input"]
        )

    elif task == "completion":
        user_message = (
            "Complete the following Computer Science text:\n\n"
            + example["input"]
        )

    else:
        raise ValueError(f"Unknown task: {task}")

    return {
        "messages": [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": user_message
            },
            {
                "role": "assistant",
                "content": example["output"]
            }
        ]
    }

def tokenize_chat_example(example, tokenizer):

    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )

    return {
        "text": text
    }