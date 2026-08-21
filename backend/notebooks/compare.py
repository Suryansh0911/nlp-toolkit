import json
from src.inference.base_model import BaseModel
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADATPER_PATH = "adapters/summarization"

with open("data/raw/cs_examples.json", "r", encoding="utf-8") as f:
    data = json.load(f)

example = next(item for item in data if item["task"] == "summarization")

base_model = BaseModel(MODEL_NAME)
prompt = (
    "Summarize the following"
    "Computer Science text : \n\n"
    + example["input"]
)

base_response = base_model.generate(prompt, max_new_tokens=100)

print("="*60)
print("BASE_MODEL")
print("="*60)

print(base_response)


base_model.model = PeftModel.from_pretrained(base_model.model, ADATPER_PATH)

adapter_response = base_model.generate(prompt, max_new_tokens=100)

print("\n" + "=" * 70)
print("BASE + LoRA")
print("=" * 70)

print(adapter_response)


print("\n" + "=" * 70)
print("REFERENCE")
print("=" * 70)

print(example["output"])