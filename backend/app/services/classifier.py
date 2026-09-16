"""
RAG-grounded classification.

retrieve 5 labelled neighbours -> put them in the prompt as precedent ->
ChatPromptTemplate | chat model (with fallbacks) | PydanticOutputParser.
"""

from functools import lru_cache

from app.config import settings
from app.models import Classification
from app.services.llm import build_json_chain
from app.services.vector_store import retrieve_labelled


CLASSIFY_SYSTEM = """You are a customer-satisfaction analyst. You read one piece
of customer feedback and assign exactly one category.

CATEGORIES
- "Excellent": strong satisfaction, delight, or advocacy. Superlatives, intent to
return, intent to recommend, or praise of a specific person.

- "Good": generally satisfied with a minor reservation. The customer still
delivers a positive overall verdict.

- "Need Improvements": a specific, concrete gap is named, in a constructive or
neutral tone. The complaint IS the verdict, but the customer is not angry.

- "Poor": frustration, disappointment, or churn intent. Anger, refund or
escalation demands, "never again", "worst".

THE BOUNDARY THAT MATTERS

Good and Need Improvements both contain a complaint. Decide like this: if the
customer still gives a positive overall verdict, it is "Good". If the complaint
is the verdict, it is "Need Improvements".

"Lovely meal, the wait was a bit long." -> Good

"The wait was 40 minutes and nobody told us." -> Need Improvements

RULES

- Judge intent, not keywords. "The wait was a bit long" is not "Poor".
- A polite complaint is still a complaint. Politeness does not upgrade the category.
- Feedback assembled from a form may read as "Food quality was rated 2 out of 5."
  Treat a low rating with no text as a real signal, not as missing data.
- If the text is empty, unreadable, or contains no opinion, return
  "Need Improvements" with self_confidence at or below 0.3 and say so in the
  rationale.
- Never invent detail that is not in the text.
- evidence must be verbatim substrings of the feedback, 2 to 12 words each.
- self_confidence is your own certainty. Be honest: use values below 0.6 when the
  text is short, ambiguous, or garbled.
"""


@lru_cache(maxsize=1)
def get_chain():
    """Built once. The model, its fallbacks and the parser all live inside."""

    return build_json_chain(
        CLASSIFY_SYSTEM,
        Classification,
    )


def _exemplar_block(
    neighbours: list[dict],
) -> str:

    if not neighbours:
        return (
            "No previously labelled examples were retrieved. "
            "Use the rubric alone."
        )

    lines = [
        "These are the most similar responses this team has already labelled "
        "and verified. Use them as precedent, but the rubric wins if they "
        "conflict.\n"
    ]

    for neighbour in neighbours:
        snippet = neighbour["text"].replace("\n", " ")[:220]

        lines.append(
            f'- (similarity {neighbour["score"]:.2f}) '
            f'"{snippet}" -> {neighbour["label"]}'
        )

    return "\n".join(lines)


def build_user_prompt(
    text: str,
    neighbours: list[dict],
) -> str:

    return (
        f"{_exemplar_block(neighbours)}\n\n"
        f"Now classify this feedback.\n\n"
        f'FEEDBACK:\n"""{text.strip()[:4000]}"""'
    )


def _guard(
    result: Classification,
    text: str,
) -> Classification:
    """
    Pydantic already enforced the types and the 0-1 range.

    The one thing it cannot check is whether the model quoted the
    customer or invented a quote.
    """

    lowered = text.lower()

    kept = [
        span
        for span in result.evidence[:3]
        if span.lower().strip('"').strip() in lowered
    ]

    result.evidence = kept or result.evidence[:1]

    result.rationale = (
        result.rationale.strip()[:400]
        or "No rationale returned."
    )

    return result


def classify(
    text: str,
    k: int | None = None,
) -> tuple[Classification, list[dict]]:
    """
    Classify one response.

    Returns:
        (classification, neighbours_used)

    Blocking on purpose - routes call it through asyncio.to_thread
    with a concurrency cap, which is what keeps a 40-record upload
    under the provider's rate limit.
    """

    neighbours = retrieve_labelled(
        text,
        k or settings.retrieval_k,
    )

    try:
        result = get_chain().invoke(
            {
                "input": build_user_prompt(
                    text,
                    neighbours,
                )
            }
        )

        return _guard(result, text), neighbours

    except Exception as exc:
        print(
            f"[classify] every provider failed: {exc}"
        )

        return (
            Classification(
                category="Need Improvements",
                self_confidence=0.0,
                rationale=(
                    "Automatic classification was unavailable "
                    "for this response; it has been sent for "
                    "human review."
                ),
                evidence=[],
                aspect="Other",
                severity="Low",
            ),
            neighbours,
        )