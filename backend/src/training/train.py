import argparse
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
import torch
from src.data.task_dataset import load_raw_dataset, load_task_dataset, split_dataset
from src.training.training_format import create_training_text
from src.training.lora_config import create_lora_config

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", required=True, 
        choices=["summarization", "question_answering", "completion"]
    )

    parser.add_argument(
        "--method", default="lora",
        choices=["base", "lora", "qlora"]
    )

    parser.add_argument(
        "--output_dir", default=None
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )

    return parser.parse_args()

def load_model(method: str):

    if method.strip().lower() == "base":
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        return model

    if method.strip().lower() == "lora":
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        return model

    if method.strip().lower() == "qlora":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "QLoRA requires a CUDA GPU in this setup." \
                "Run this script on a Colab NVIDIA GPU"
            )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=quantization_config,
            dtype=torch.float16,
            device_map="auto"
        )

        model = prepare_model_for_kbit_training(model)

        return model
    raise ValueError(
        f"Unsupported method: {method}"
    )

def main():

    args = parse_args()

    output_dir = (
        args.output_dir
        or f"adapters/{args.task}/{args.method}"
    )

    dataset = load_raw_dataset()
    dataset = load_task_dataset(dataset, args.task)

    train_dataset, validation_dataset, test_dataset = split_dataset(
        dataset,
        seed=42,
    )

    print("\nDataset split:")
    print(f"Total:      {len(dataset)}")
    print(f"Train:      {len(train_dataset)}")
    print(f"Validation: {len(validation_dataset)}")
    print(f"Test:       {len(test_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = train_dataset.map(
        lambda example: create_training_text(
            example,
            tokenizer
        )
    )

    validation_dataset = validation_dataset.map(
        lambda example: create_training_text(
            example,
            tokenizer
        )
    )

    model = load_model(args.method)

    # Apply LoRA only for lora and qlora methods
    if args.method in ["lora", "qlora"]:
        lora_config = create_lora_config()
        model = get_peft_model(
            model,
            lora_config
        )
        model.print_trainable_parameters()
    else:
        print("Training with full fine-tuning (base method)")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} (100%)")

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        dataset_text_field="text",
        max_length=512,
        packing=False,
        gradient_checkpointing=True,
        fp16=False,
        bf16=False,
        seed=42,
    )


    trainer = SFTTrainer(
        model=model,
        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=validation_dataset,

        processing_class=tokenizer,
    )

    trainer.train()


    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nAdapter saved to: {output_dir}")


if __name__ == "__main__":
    main()
