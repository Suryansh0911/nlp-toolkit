import argparse
import json
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

    return parser.parse_args()

def main():
    args = parse_args()

    dataset = load_raw_dataset()
    examples = [example for example in dataset if example["task"] == args.task]

    model = BaseModel()

    if args.adapter:
        model.model = (PeftModel.from_pretrained(model.model, args.adapter))
        model.model.eval()

    predictions = []

    for example in examples:
        messages = (format_inference_messages(example))
        prediction = (model.generate_messages(messages, max_new_tokens=512))

        predictions.append({
            "input": example["input"],
            "reference": example["output"],
            "prediction": prediction
        })

    references = [item["reference"] for item in predictions]

    outputs = [
        item["prediction"] for item in predictions  
    ]

    print("\nTask: ", args.task)

    if args.task == "question_answering":
        en = [exact_match(pred, ref) for pred, ref in zip(outputs, references)]

        f1 = [token_f1(pred, ref) for pred, ref in zip(outputs, references)]

        print(
            "Exact match: ", sum(f1) / len(f1)
        )

    else:
        scores = rouge_scores(outputs, references)
        print("ROUGE-1 : ", scores["rouge1"])

        print("ROUGE-2 : ", scores["rouge2"])

        print("ROUGE-L : ", scores["rougeL"])

    print("\nPredictions: ")
    for item in predictions:
        print("\nINPUT:")
        print(item["input"])

        print("\nREFERENCE:")
        print(item["reference"])

        print("\nPREDICTION:")
        print(item["prediction"])

        print("-" * 60)
if __name__ == "__main__":
    main()