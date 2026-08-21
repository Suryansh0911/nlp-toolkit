from transformers import AutoModelForCausalLM
from peft import get_peft_model
from src.training.lora_config import create_lora_config

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
lora_config = create_lora_config()
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        print(name)