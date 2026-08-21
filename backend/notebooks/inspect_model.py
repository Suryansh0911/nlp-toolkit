from transformers import AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# print(model)

for name, module in model.named_modules():
    if "proj" in name.lower():
        print(name, "->", module.__class__.__name__)