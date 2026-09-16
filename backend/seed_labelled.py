"""Load data/labelled_seed.jsonl into the retrieval namespace.

    uv run python seed_labelled.py
    uv run python seed_labelled.py --file data/labelled_seed.jsonl
"""

import argparse
import json
import pathlib
import sys

from app.config import settings
from app.models import CATEGORIES
from app.services.vector_store import add_texts

VALID = set(CATEGORIES)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/labelled_seed.jsonl")
    args = parser.parse_args()

    path = pathlib.Path(args.file)
    if not path.exists():
        print(f"missing file: {path}")
        return 1

    ids, texts, metadatas, counts = [], [], [], {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        label = row["label"]
        if label not in VALID:
            print(f"line {number}: bad label {label!r}")
            return 1
        ids.append(f"seed-{number}")
        texts.append(row["text"])
        metadatas.append({"label": label, "origin": "seed"})
        counts[label] = counts.get(label, 0) + 1

    if not texts:
        print("nothing to seed")
        return 1

    add_texts(settings.labelled_namespace, ids, texts, metadatas)

    print(f"seeded {len(texts)} examples into namespace "
          f"'{settings.labelled_namespace}'")
    for label in sorted(counts):
        print(f"  {label:<20} {counts[label]}")

    if min(counts.values()) < 8:
        print("WARNING: at least one category has fewer than 8 examples. "
              "Retrieval will be biased toward the larger categories.")

    return 0


if __name__ == "__main__":
    sys.exit(main())