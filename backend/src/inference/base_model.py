from transformers import AutoTokenizer, AutoModelForCausalLM

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

    def generate(
        self,
        prompt,
        max_new_tokens=512
    ):

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

        input_length = inputs["input_ids"].shape[-1]

        generated_tokens = outputs[0][input_length:]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return response.strip()