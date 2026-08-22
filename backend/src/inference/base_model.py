from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class BaseModel:

    DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

    def __init__(self, model_name=None):
        self.model_name = model_name or self.DEFAULT_MODEL

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (self.tokenizer.eos_token)

        self.model.eval()
        self.device = next(self.model.parameters()).device

    def generate_messages(
        self,
        messages,
        max_new_tokens=512
    ):

        inputs = (
            self.tokenizer.apply_chat_template(
                messages, tokenize=True,
                add_generation_prompt=True,
                return_dict = True,
                return_tensors = "pt"
            )
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        input_length = (inputs["input_ids"].shape[-1])

        generated_tokens = (outputs[0][input_length:])

        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

    def generate(self, prompt, max_new_tokens = 512):
        messages = [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]

        return self.generate_messages(messages, max_new_tokens)