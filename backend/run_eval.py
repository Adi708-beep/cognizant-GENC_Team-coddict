"""Measure classifier accuracy and confidence calibration on a held-out set.

    uv run python run_eval.py
"""

import json
import pathlib
import sys
import time

from app.models import CATEGORIES
from app.services.classifier import classify
from app.services.confidence import fuse, neighbour_agreement

BUCKETS = [(0.0, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 1.01)]


def confusion_matrix(pairs):
    matrix = {t: {p: 0 for p in CATEGORIES} for t in CATEGORIES}
    for truth, predicted in pairs:
        matrix[truth][predicted] += 1
    return matrix


def macro_f1(matrix):
    scores = []
    for category in CATEGORIES:
        true_positive = matrix[category][category]
        false_negative = sum(matrix[category].values()) - true_positive
        false_positive = sum(matrix[t][category] for t in CATEGORIES) - true_positive
        precision_den = true_positive + false_positive
        recall_den = true_positive + false_negative
        precision = true_positive / precision_den if precision_den else 0.0
        recall = true_positive / recall_den if recall_den else 0.0
        denominator = precision + recall
        scores.append(2 * precision * recall / denominator if denominator else 0.0)
    return sum(scores) / len(scores)


def main() -> int:
    path = pathlib.Path("data/eval_set.jsonl")
    if not path.exists():
        print("missing data/eval_set.jsonl")
        return 1

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        print("eval set is empty")
        return 1

    pairs, scored, started = [], [], time.time()
    for index, row in enumerate(rows, 1):
        analysis, neighbours = classify(row["text"])
        agreement = neighbour_agreement(analysis.category, neighbours)
        breakdown = fuse(analysis.self_confidence, agreement, 1.0)

        pairs.append((row["label"], analysis.category))
        scored.append((row["label"] == analysis.category, breakdown["final"]))

        flag = "ok  " if row["label"] == analysis.category else "MISS"
        print(f'{index:>3} {flag} conf={breakdown["final"]:.2f} '
              f'true={row["label"]:<18} pred={analysis.category:<18} '
              f'{row["text"][:55]}')

    correct = sum(1 for truth, predicted in pairs if truth == predicted)
    matrix = confusion_matrix(pairs)
    elapsed = time.time() - started

    print("\n" + "=" * 78)
    print(f"ACCURACY {correct}/{len(pairs)} = {correct / len(pairs):.1%}")
    print(f"MACRO F1 {macro_f1(matrix):.3f}")
    print(f"ELAPSED  {elapsed:.1f}s ({elapsed / len(pairs):.2f}s per record)")

    print("\nCONFUSION MATRIX (rows = truth, columns = predicted)")
    width = max(len(c) for c in CATEGORIES) + 2
    print(" " * width + "".join(c[:10].rjust(12) for c in CATEGORIES))
    for truth in CATEGORIES:
        print(truth.ljust(width)
              + "".join(str(matrix[truth][p]).rjust(12) for p in CATEGORIES))

    print("\nCALIBRATION (is a confident prediction actually more often right?)")
    print(f'{"confidence band":<20}{"n":>5}{"accuracy":>12}')
    for low, high in BUCKETS:
        band = [ok for ok, conf in scored if low <= conf < high]
        accuracy = f"{sum(band) / len(band):.1%}" if band else "-"
        print(f"{f'{low:.1f} - {min(high, 1.0):.1f}':<20}{len(band):>5}{accuracy:>12}")

    return 0


if __name__ == "__main__":
    sys.exit(main())