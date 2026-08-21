import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import get_peft_model
from trl import SFTTrainer
from src.training.lora_config import create_lora_config
from src.data.training_format import create_training_text

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

with open("data/raw/cs_examples.json", "r", encoding="utf-8") as f:
    data = json.load(f)

summarization_examples = [example for example in data if example["task"]=="summarization"]

dataset = Dataset.from_list(summarization_examples)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

dataset = dataset.map(
    lambda example: create_training_text(example, tokenizer)
)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

lora_config = create_lora_config()

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="adapters/summarization",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer
)

trainer.train()

trainer.save_model("adapters/summarization")

print("Summarization LoRA adapter saved")