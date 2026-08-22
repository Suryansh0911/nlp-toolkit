from pathlib import Path
from peft import PeftModel

class AdapterManager:

    def __init__(self, base_model, adapter_root="adapters"):

        self.base_model = base_model
        self.adapter_root = Path(adapter_root)

        self.active_task = None

    def load(self, task, method="lora"):
        adapter_path = (self.adapter_root/task/method)

        if not adapter_path.exists():
            raise FileNotFoundError(
                f"Adapter not found: "
                f"{self.adapter_path}"
            )

        self.base_model.model = (PeftModel.from_pretrained(self.base_model.model, str(adapter_path)))
        self.base_model.model.eval()
        self.active_task = task
        return self.base_model

    def unload(self):
        if hasattr(self.base_model.model, "unload"):
            self.base_model.model = (self.base_model.model.unload())

        self.active_task = None