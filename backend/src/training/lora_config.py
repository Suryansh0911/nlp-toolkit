from peft import LoraConfig

DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

def create_lora_config(
        r: int=16,
        lora_alpha: int=32,
        lora_dropout: float=0.05) -> LoraConfig:

    return LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=DEFAULT_TARGET_MODULES
    )