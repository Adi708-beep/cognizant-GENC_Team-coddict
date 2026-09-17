"""Seed the labelled feedback examples into the vector store.

Reads:
    backend/data/labelled_seed.jsonl

Each JSONL line should contain:
    {
        "id": "...",
        "text": "...",
        "label": "Excellent|Good|Need Improvements|Poor"
    }

The examples are stored in the "labelled" namespace and are later
retrieved to ground classification.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.services.vector_store import add_labelled


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
SEED_FILE = BASE_DIR / "data" / "labelled_seed.jsonl"


# -------------------------------------------------------------------
# Load labelled examples
# -------------------------------------------------------------------

def load_seed_examples() -> list[dict]:
    """Load labelled examples from the JSONL seed file."""

    if not SEED_FILE.exists():
        raise FileNotFoundError(
            f"Seed file not found:\n{SEED_FILE}\n\n"
            "Create backend/data/labelled_seed.jsonl first."
        )

    examples = []

    with SEED_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            # Ignore empty lines
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            if not isinstance(item, dict):
                raise ValueError(
                    f"Line {line_number} must contain a JSON object."
                )

            if not item.get("text"):
                raise ValueError(
                    f"Line {line_number} is missing 'text'."
                )

            if not item.get("label"):
                raise ValueError(
                    f"Line {line_number} is missing 'label'."
                )

            examples.append(item)

    return examples


# -------------------------------------------------------------------
# Seed vector store
# -------------------------------------------------------------------

def seed_labelled() -> None:
    """Add all labelled examples to the labelled vector namespace."""

    examples = load_seed_examples()

    print(f"[seed] found {len(examples)} labelled examples")
    print("[seed] adding examples to labelled vector store...")

    added = 0

    for index, example in enumerate(examples, start=1):
        record_id = str(
            example.get("id")
            or example.get("record_id")
            or f"seed-{index}"
        )

        text = str(example["text"]).strip()
        label = str(example["label"]).strip()

        origin = str(
            example.get("origin", "seed")
        )

        add_labelled(
            record_id=record_id,
            text=text,
            label=label,
            origin=origin,
        )

        added += 1

        print(
            f"[seed] {added}/{len(examples)} "
            f"-> {label} -> {record_id}"
        )

    print()
    print("[seed] ========================================")
    print(f"[seed] successfully seeded {added} examples")
    print("[seed] namespace: labelled")
    print("[seed] ========================================")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == "__main__":
    seed_labelled()