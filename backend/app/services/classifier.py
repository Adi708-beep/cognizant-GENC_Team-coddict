"""TEMPORARY stand-in classifier, owned by Person C in the real build.

This lets Person D (RAG/eval) test run_eval.py end to end before the real
LangChain classification chain exists. It uses retrieved neighbours' labels
by majority vote instead of calling an LLM. Delete this file and let
services/classifier.py be replaced once Person C's real chain is ready.
"""

from app.models import Classification
from app.services.vector_store import retrieve_labelled


def classify(text: str, k: int | None = None) -> tuple[Classification, list[dict]]:
    neighbours = retrieve_labelled(text, k)

    if not neighbours:
        return (
            Classification(
                category="Need Improvements",
                self_confidence=0.3,
                rationale="No labelled precedent was retrieved; low-confidence default.",
                evidence=[],
                aspect="Other",
                severity="Low",
            ),
            neighbours,
        )

    # majority vote among retrieved neighbours, weighted by similarity
    scores: dict[str, float] = {}
    for neighbour in neighbours:
        scores[neighbour["label"]] = scores.get(neighbour["label"], 0.0) + neighbour["score"]

    predicted = max(scores, key=scores.get)
    total = sum(scores.values())
    confidence = round(scores[predicted] / total, 3) if total else 0.5

    return (
        Classification(
            category=predicted,
            self_confidence=confidence,
            rationale=f"Placeholder classifier: majority vote over {len(neighbours)} "
                      f"retrieved neighbours (predicted {predicted}).",
            evidence=[],
            aspect="Other",
            severity="Low",
        ),
        neighbours,
    )