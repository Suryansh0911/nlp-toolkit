from src.inference.base_model import BaseModel
from src.inference.adapter_manager import AdapterManager
from src.data.formatting import format_inference_messages

class NLPToolkit:

    def __init__(self, model_name=None):
        self.base_model = BaseModel(model_name=model_name)
        self.adapters = AdapterManager(self.base_model)

    def load_task(self, task, method="lora"):
        self.adapters.load(task, method)

    def unload_task(self):
        self.adapters.unload()

    def generate(self, example, max_new_tokens=512):
        messages = (format_inference_messages(example))

        return self.base_model.generate_messages(messages, max_new_tokens)

    def ask(self, question, max_new_tokens=512):
        return self.base_model.generate(question, max_new_tokens)

    def summarize(self, text, max_new_tokens=512):
        example = {
            "task": "summarization",
            "input": text,
            "output": ""
        }

        return self.generate(example, max_new_tokens)

    def answer(self, context, question, max_new_tokens=512):
        example = {
            "task": "question_answering",
            "input": (
                f"Context: {context}\n"
                f"Question: {question}"
            ),
            "output": "",
        }

        return self.generate(example, max_new_tokens)

    def complete(self, text, max_new_tokens=100):

        example = {
            "task": "completion",
            "input": text,
            "output": "",
        }

        return self.generate(example, max_new_tokens)