"""Aggregate analysis: from many classified records to a few decisions.

Deliberately NOT retrieval-augmented. Every fact the model is shown comes from
the records themselves, and the counts on the response are computed in Python,
not asked for from the model.
"""

from collections import Counter
from functools import lru_cache

from app.models import BusinessInsights, Issue
from app.services.llm import build_json_chain

MAX_QUOTES_PER_BUCKET = 6

INSIGHTS_SYSTEM = """You are a customer-experience consultant briefing the owner
of a business. You are given a digest of classified customer feedback: how many
responses fell into each satisfaction category, how they break down by aspect,
and a sample of the actual customer wording.

Your job is to tell the owner what to DO, not to restate the numbers.

RULES
- Every problem you name must be supported by the quoted feedback in the digest.
  If only one customer mentioned something, it is not a recurring problem.
- Order problems by how much damage they are doing: frequency first, then
  severity. The worst one goes first.
- Each recommendation must be a concrete action an owner can assign to someone
  this week. "Improve service" is useless. "Add a second person to the counter
  between 7pm and 9pm" is useful.
- priorities is the short list you would read out in a meeting: at most three,
  in the order you would fix them.
- praise is what the business should protect, not invent. Only list things the
  feedback actually praises.
- Do not mention percentages that are not in the digest. Do not invent numbers."""


@lru_cache(maxsize=1)
def get_chain():
    return build_json_chain(INSIGHTS_SYSTEM, BusinessInsights)


def build_digest(records: list[dict]) -> str:
    """A compact, factual briefing. Keeps the prompt small enough for free tiers."""
    categories = Counter(r.get("category", "Unknown") for r in records)
    aspects = Counter(r.get("aspect", "Other") for r in records)
    negative = Counter(
        r.get("aspect", "Other") for r in records
        if r.get("category") in ("Poor", "Need Improvements")
    )

    lines = [f"TOTAL RESPONSES: {len(records)}", "", "BY CATEGORY:"]
    for name, count in categories.most_common():
        lines.append(f"  {name}: {count}")

    lines.append("")
    lines.append("BY ASPECT (all responses):")
    for name, count in aspects.most_common():
        lines.append(f"  {name}: {count}")

    lines.append("")
    lines.append("BY ASPECT (negative and constructive responses only):")
    for name, count in negative.most_common():
        lines.append(f"  {name}: {count}")

    for bucket in ("Poor", "Need Improvements", "Excellent", "Good"):
        quotes = [r.get("raw_text", "")[:220] for r in records
                  if r.get("category") == bucket][:MAX_QUOTES_PER_BUCKET]
        if not quotes:
            continue
        lines.append("")
        lines.append(f"SAMPLE {bucket.upper()} FEEDBACK:")
        lines.extend(f'  - "{quote}"' for quote in quotes)

    return "\n".join(lines)


def generate(records: list[dict]) -> BusinessInsights:
    """Blocking. Routes call it through asyncio.to_thread."""
    if not records:
        return BusinessInsights(
            headline="No feedback has been analysed yet.",
            praise=[], problems=[], priorities=[],
        )

    try:
        return get_chain().invoke({"input": build_digest(records)})
    except Exception as exc:  # noqa: BLE001
        print(f"[insights] generation failed: {exc}")
        negative = Counter(
            r.get("aspect", "Other") for r in records
            if r.get("category") in ("Poor", "Need Improvements")
        )
        problems = [
            Issue(title=f"Recurring complaints about {aspect.lower()}",
                  aspect=aspect if aspect in ("Food", "Service", "Price",
                                                "Cleanliness", "Waiting Time",
                                                "Product", "Support") else "Other",
                  severity="Medium",
                  recommendation="Review these responses manually; the AI summary "
                                 "was unavailable.",
                  supporting_quotes=[])
            for aspect, _ in negative.most_common(3)
        ]
        return BusinessInsights(
            headline="AI summary unavailable; showing counted themes only.",
            praise=[], problems=problems, priorities=[],
        )