import argparse
import json
import torch
from collections import defaultdict
from src.data.task_dataset import load_raw_dataset
from src.data.formatting import format_inference_messages
from src.evaluation.metrics import exact_match, token_f1, rouge_scores
from src.inference.base_model import BaseModel
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a CS-NLP task model."
    )

    parser.add_argument(
        "--task",
        required=True,
        choices=[
            "summarization",
            "question_answering",
            "completion",
        ],
    )

    parser.add_argument(
        "--adapter",
        default=None,
        help="Path to a LoRA/QLoRA adapter.",
    )

    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test"],
        help="Dataset split to evaluate (default: test)",
    )

    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (default: all)",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate (default: 128)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )

    parser.add_argument(
        "--save_results",
        default=None,
        help="Path to save evaluation results as JSON (optional)",
    )

    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=True,
        help="Use GPU if available (default: True)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    print(f"\nUsing device: {device.upper()}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load dataset
    dataset = load_raw_dataset()
    examples = [example for example in dataset if example["task"] == args.task]

    if not examples:
        print(f"No examples found for task: {args.task}")
        return

    # Limit examples if specified
    if args.max_examples:
        examples = examples[:args.max_examples]

    print(f"\nEvaluating {len(examples)} examples for task: {args.task}")
    print(f"Max new tokens: {args.max_new_tokens}")

    # Load model
    model = BaseModel()

    # Move model to device
    if device == "cuda":
        model.model = model.model.to(device)
        model.device = device

    # Load adapter if provided
    if args.adapter:
        print(f"Loading adapter from: {args.adapter}")
        model.model = PeftModel.from_pretrained(model.model, args.adapter)
        model.model.eval()

        # Move to GPU after loading adapter
        if device == "cuda":
            model.model = model.model.to(device)

    predictions = []

    # Run inference
    print(f"\nStarting evaluation...")

    for i, example in enumerate(examples):
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Processing {i + 1}/{len(examples)}...")

        messages = format_inference_messages(example)
        prediction = model.generate_messages(
            messages,
            max_new_tokens=args.max_new_tokens
        )

        predictions.append({
            "input": example["input"],
            "reference": example["output"],
            "prediction": prediction
        })

    # Extract references and outputs
    references = [item["reference"] for item in predictions]
    outputs = [item["prediction"] for item in predictions]

    # Calculate metrics
    print("\n" + "=" * 50)
    print(f"Task: {args.task}")
    print("=" * 50)

    results = {
        "task": args.task,
        "num_examples": len(examples),
        "adapter": args.adapter,
        "max_new_tokens": args.max_new_tokens,
        "device": device,
    }

    if args.task == "question_answering":
        em_scores = [exact_match(pred, ref) for pred, ref in zip(outputs, references)]
        f1_scores = [token_f1(pred, ref) for pred, ref in zip(outputs, references)]

        em_avg = sum(em_scores) / len(em_scores)
        f1_avg = sum(f1_scores) / len(f1_scores)

        print(f"Exact Match: {em_avg:.4f}")
        print(f"Token F1:    {f1_avg:.4f}")

        results["exact_match"] = em_avg
        results["token_f1"] = f1_avg

    else:
        scores = rouge_scores(outputs, references)
        print(f"ROUGE-1: {scores['rouge1']:.4f}")
        print(f"ROUGE-2: {scores['rouge2']:.4f}")
        print(f"ROUGE-L: {scores['rougeL']:.4f}")

        results["rouge1"] = float(scores["rouge1"])
        results["rouge2"] = float(scores["rouge2"])
        results["rougeL"] = float(scores["rougeL"])

    print("=" * 50)

    # Print sample predictions
    print("\nSample Predictions:")
    print("-" * 50)

    for i, item in enumerate(predictions[:3]):
        print(f"\n[Example {i + 1}]")
        print(f"INPUT: {item['input'][:200]}...")
        print(f"REFERENCE: {item['reference'][:200]}...")
        print(f"PREDICTION: {item['prediction'][:200]}...")

    # Save results if specified
    if args.save_results:
        with open(args.save_results, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.save_results}")


if __name__ == "__main__":
    main()
