from src.data.formatting import format_chat

def create_training_text(example, tokenizer):

    chat = format_chat(example)

    text = tokenizer.apply_chat_template(
        chat["messages"],
        tokenize = False,
        add_generation_prompt = False
    )

    return {"text" : text}