import re
from evaluate import load

rouge = load("rouge")

def normalize_text(text):

    text = text.lower()

    text = re.sub(
        r"[^\w\s]",
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def exact_match(prediction, reference):

    prediction = normalize_text(prediction)

    reference = normalize_text(reference)

    return int(prediction == reference)


def token_f1(prediction, reference):

    prediction_tokens = (
        normalize_text(
            prediction
        ).split()
    )

    reference_tokens = (
        normalize_text(
            reference
        ).split()
    )

    if not prediction_tokens:
        return 0.0

    if not reference_tokens:
        return 0.0

    prediction_counts = {}

    reference_counts = {}

    for token in prediction_tokens:

        prediction_counts[token] = (
            prediction_counts.get(token, 0)
            + 1
        )

    for token in reference_tokens:

        reference_counts[token] = (
            reference_counts.get(token, 0)
            + 1
        )

    common = 0

    for token, count in prediction_counts.items():

        common += min(
            count,
            reference_counts.get(
                token,
                0,
            ),
        )

    if common == 0:
        return 0.0

    precision = (
        common
        / len(prediction_tokens)
    )

    recall = (
        common
        / len(reference_tokens)
    )

    return (2 * precision * recall/ (precision + recall))


def rouge_scores(
    predictions,
    references,
):

    return rouge.compute(
        predictions=predictions,
        references=references,
    )