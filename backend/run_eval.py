"""Feedy evaluation script.

Evaluates the feedback classifier against data/eval_set.jsonl.

Run from backend:

    uv run python run_eval.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
EVAL_FILE = BASE_DIR / "data" / "eval_set.jsonl"


# ============================================================
# Expected labels
# ============================================================

LABELS = [
    "Excellent",
    "Good",
    "Need Improvements",
    "Poor",
]


# ============================================================
# Load evaluation data
# ============================================================

def load_eval_data() -> list[dict[str, Any]]:
    """Load evaluation examples from JSONL."""

    if not EVAL_FILE.exists():
        raise FileNotFoundError(
            f"\nEvaluation file does not exist:\n{EVAL_FILE}\n\n"
            "Expected file:\n"
            "backend/data/eval_set.jsonl"
        )

    examples: list[dict[str, Any]] = []

    with EVAL_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"[eval] WARNING: invalid JSON on line "
                    f"{line_number}: {exc}"
                )
                continue

            if not isinstance(data, dict):
                print(
                    f"[eval] WARNING: line {line_number} "
                    "is not a JSON object"
                )
                continue

            text = data.get("text")
            label = data.get("label")

            if not text:
                print(
                    f"[eval] WARNING: line {line_number} "
                    "has no text"
                )
                continue

            if not label:
                print(
                    f"[eval] WARNING: line {line_number} "
                    "has no label"
                )
                continue

            examples.append(data)

    return examples


# ============================================================
# Import classifier
# ============================================================

def load_classifier():
    """Import the classifier safely."""

    try:
        from app.services.classifier import classify

        print("[eval] classifier loaded: app.services.classifier.classify")

        return classify

    except ImportError as exc:
        raise ImportError(
            "\nCould not import the Feedy classifier.\n\n"
            "Expected:\n"
            "app/services/classifier.py\n\n"
            "with a function named:\n"
            "classify\n\n"
            f"Original error:\n{exc}"
        ) from exc


# ============================================================
# Extract category
# ============================================================

def extract_category(result: Any) -> str | None:
    """Extract classification category from any common result format."""

    if result is None:
        return None

    # --------------------------------------------------------
    # Pydantic / normal object
    # --------------------------------------------------------

    for attribute in [
        "category",
        "label",
        "classification",
        "sentiment",
    ]:
        if hasattr(result, attribute):

            value = getattr(result, attribute)

            if value is not None:
                return clean_label(value)

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(result, dict):

        for key in [
            "category",
            "label",
            "classification",
            "sentiment",
        ]:

            if key in result and result[key] is not None:
                return clean_label(result[key])

    # --------------------------------------------------------
    # JSON string
    # --------------------------------------------------------

    if isinstance(result, str):

        text = result.strip()

        # Try JSON first
        try:
            parsed = json.loads(text)

            if isinstance(parsed, dict):
                return extract_category(parsed)

        except Exception:
            pass

        # Check whether the raw string itself is a label
        cleaned = clean_label(text)

        if cleaned in LABELS:
            return cleaned

        # Search for a known label inside the response
        lower_text = text.lower()

        for label in LABELS:
            if label.lower() in lower_text:
                return label

    return None


# ============================================================
# Clean category
# ============================================================

def clean_label(value: Any) -> str:
    """Normalize category text."""

    text = str(value).strip()

    # Remove common formatting
    text = text.replace("_", " ")
    text = text.replace("-", " ")

    normalized = text.lower().strip()

    aliases = {
        "excellent": "Excellent",
        "good": "Good",
        "need improvement": "Need Improvements",
        "needs improvement": "Need Improvements",
        "need improvements": "Need Improvements",
        "needs improvements": "Need Improvements",
        "poor": "Poor",
    }

    if normalized in aliases:
        return aliases[normalized]

    return text


# ============================================================
# Extract confidence
# ============================================================

def extract_confidence(result: Any) -> float | None:
    """Extract confidence from classifier output."""

    possible_names = [
        "self_confidence",
        "confidence",
        "score",
        "probability",
    ]

    value = None

    # Object
    for name in possible_names:

        if hasattr(result, name):
            value = getattr(result, name)

            if value is not None:
                break

    # Dictionary
    if value is None and isinstance(result, dict):

        for name in possible_names:

            if name in result and result[name] is not None:
                value = result[name]
                break

    if value is None:
        return None

    try:
        confidence = float(value)

        # Convert percentages such as 85 -> 0.85
        if confidence > 1:
            confidence = confidence / 100

        return max(0.0, min(1.0, confidence))

    except (TypeError, ValueError):
        return None


# ============================================================
# Calculate accuracy
# ============================================================

def calculate_accuracy(
    actual: list[str],
    predicted: list[str],
) -> float:

    if not actual:
        return 0.0

    correct = sum(
        1
        for a, p in zip(actual, predicted)
        if a == p
    )

    return correct / len(actual)


# ============================================================
# Calculate precision / recall / F1
# ============================================================

def calculate_metrics(
    actual: list[str],
    predicted: list[str],
) -> dict[str, dict[str, float]]:

    results: dict[str, dict[str, float]] = {}

    for label in LABELS:

        tp = 0
        fp = 0
        fn = 0

        for true_label, predicted_label in zip(
            actual,
            predicted,
        ):

            if true_label == label and predicted_label == label:
                tp += 1

            elif true_label != label and predicted_label == label:
                fp += 1

            elif true_label == label and predicted_label != label:
                fn += 1

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        results[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return results


# ============================================================
# Confusion matrix
# ============================================================

def create_confusion_matrix(
    actual: list[str],
    predicted: list[str],
) -> dict[str, dict[str, int]]:

    matrix = {
        actual_label: {
            predicted_label: 0
            for predicted_label in LABELS
        }
        for actual_label in LABELS
    }

    for true_label, predicted_label in zip(
        actual,
        predicted,
    ):

        if true_label not in matrix:
            continue

        if predicted_label not in LABELS:
            continue

        matrix[true_label][predicted_label] += 1

    return matrix


# ============================================================
# Print confusion matrix
# ============================================================

def print_confusion_matrix(
    matrix: dict[str, dict[str, int]],
) -> None:

    print()
    print("=" * 90)
    print("CONFUSION MATRIX")
    print("=" * 90)

    print(
        f"{'Actual / Predicted':<22}"
        f"{'Excellent':<17}"
        f"{'Good':<17}"
        f"{'Need Improvements':<20}"
        f"{'Poor':<10}"
    )

    print("-" * 90)

    for actual_label in LABELS:

        print(
            f"{actual_label:<22}"
            f"{matrix[actual_label]['Excellent']:<17}"
            f"{matrix[actual_label]['Good']:<17}"
            f"{matrix[actual_label]['Need Improvements']:<20}"
            f"{matrix[actual_label]['Poor']:<10}"
        )

    print("-" * 90)


# ============================================================
# Calibration
# ============================================================

def print_calibration(
    actual: list[str],
    predicted: list[str],
    confidence: list[float | None],
) -> None:

    data = []

    for true_label, predicted_label, score in zip(
        actual,
        predicted,
        confidence,
    ):

        if score is not None:
            data.append(
                (
                    true_label == predicted_label,
                    score,
                )
            )

    print()
    print("=" * 90)
    print("CONFIDENCE CALIBRATION")
    print("=" * 90)

    if not data:
        print("No confidence scores were returned by the classifier.")
        return

    buckets = [
        ("0.00 - 0.49", 0.00, 0.50),
        ("0.50 - 0.69", 0.50, 0.70),
        ("0.70 - 0.84", 0.70, 0.85),
        ("0.85 - 1.00", 0.85, 1.01),
    ]

    print(
        f"{'Confidence':<20}"
        f"{'Samples':<12}"
        f"{'Accuracy':<15}"
        f"{'Avg Confidence':<18}"
    )

    print("-" * 90)

    for name, low, high in buckets:

        bucket = [
            item
            for item in data
            if low <= item[1] < high
        ]

        if not bucket:
            print(
                f"{name:<20}"
                f"{0:<12}"
                f"{'N/A':<15}"
                f"{'N/A':<18}"
            )
            continue

        accuracy = sum(
            1
            for correct, _ in bucket
            if correct
        ) / len(bucket)

        average = sum(
            score
            for _, score in bucket
        ) / len(bucket)

        print(
            f"{name:<20}"
            f"{len(bucket):<12}"
            f"{accuracy * 100:>7.2f}%       "
            f"{average:>8.3f}"
        )


# ============================================================
# Main evaluation
# ============================================================

def main() -> None:

    print()
    print("=" * 90)
    print("FEEDY CLASSIFIER EVALUATION")
    print("=" * 90)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    examples = load_eval_data()

    print()
    print(f"[eval] loaded {len(examples)} evaluation examples")
    print(f"[eval] file: {EVAL_FILE}")

    if not examples:
        print("[eval] No evaluation examples found.")
        return

    # --------------------------------------------------------
    # Load classifier
    # --------------------------------------------------------

    classify = load_classifier()

    print()
    print("[eval] starting evaluation...")
    print()

    actual: list[str] = []
    predicted: list[str] = []
    confidence: list[float | None] = []

    errors = 0

    # --------------------------------------------------------
    # Evaluate each example
    # --------------------------------------------------------

    for index, example in enumerate(
        examples,
        start=1,
    ):

        text = str(example["text"]).strip()
        expected = clean_label(example["label"])

        try:

            result = classify(text)

            prediction = extract_category(result)
            score = extract_confidence(result)

            if prediction is None:

                print(
                    f"[eval] {index:02d}/{len(examples):02d} "
                    f"ERROR - could not extract category"
                )

                prediction = "ERROR"
                errors += 1

            status = (
                "PASS"
                if prediction == expected
                else "MISS"
            )

            confidence_text = (
                f"{score:.3f}"
                if score is not None
                else "N/A"
            )

            print(
                f"[eval] {index:02d}/{len(examples):02d} "
                f"{status:<5} "
                f"expected={expected:<20} "
                f"predicted={prediction:<20} "
                f"confidence={confidence_text}"
            )

        except Exception as exc:

            prediction = "ERROR"
            score = None
            errors += 1

            print(
                f"[eval] {index:02d}/{len(examples):02d} "
                f"ERROR"
            )

            print(
                f"        {type(exc).__name__}: {exc}"
            )

        actual.append(expected)
        predicted.append(prediction)
        confidence.append(score)

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = calculate_accuracy(
        actual,
        predicted,
    )

    correct = sum(
        a == p
        for a, p in zip(actual, predicted)
    )

    incorrect = len(actual) - correct

    print()
    print("=" * 90)
    print("OVERALL RESULTS")
    print("=" * 90)

    print(f"Total examples : {len(actual)}")
    print(f"Correct        : {correct}")
    print(f"Incorrect      : {incorrect}")
    print(f"Errors         : {errors}")
    print(f"Accuracy       : {accuracy * 100:.2f}%")

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        actual,
        predicted,
    )

    print()
    print("=" * 90)
    print("PER-CLASS METRICS")
    print("=" * 90)

    print(
        f"{'Category':<22}"
        f"{'Precision':<18}"
        f"{'Recall':<18}"
        f"{'F1 Score':<18}"
    )

    print("-" * 90)

    for label in LABELS:

        values = metrics[label]

        print(
            f"{label:<22}"
            f"{values['precision'] * 100:>7.2f}%         "
            f"{values['recall'] * 100:>7.2f}%         "
            f"{values['f1'] * 100:>7.2f}%"
        )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    matrix = create_confusion_matrix(
        actual,
        predicted,
    )

    print_confusion_matrix(matrix)

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    print_calibration(
        actual,
        predicted,
        confidence,
    )

    # --------------------------------------------------------
    # Label distribution
    # --------------------------------------------------------

    counts = Counter(actual)

    print()
    print("=" * 90)
    print("EVALUATION SET DISTRIBUTION")
    print("=" * 90)

    for label in LABELS:
        print(
            f"{label:<22}: "
            f"{counts.get(label, 0)}"
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print(f"FINAL ACCURACY: {accuracy * 100:.2f}%")
    print("=" * 90)
    print()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()