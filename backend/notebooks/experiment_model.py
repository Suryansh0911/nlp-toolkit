from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

messages = [
    {
        "role" : "system",
        "content" : "you are an experienced computer science professor"
    },
    {
        "role" : "user",
        "content" : "explain sliding window in very simple terms along with tricks to apply it"
    }
]
inputs = tokenizer.apply_chat_template(
    messages, 
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)

outputs = model.generate(
    **inputs, max_new_tokens=512, do_sample=True, temperature=1
)

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

print(response)

input = tokenizer.tokenize("hash tables provide O(1) lookup")
print(input)
input_ids = tokenizer.convert_tokens_to_ids(input)
print(input_ids)
outputs = tokenizer.decode(input_ids, skip_special_tokens=True)
print(outputs)