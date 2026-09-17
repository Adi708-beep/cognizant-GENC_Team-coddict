"""Feedback classifier for Feedy.

Pipeline:

feedback
    ↓
retrieve labelled examples
    ↓
build grounded prompt
    ↓
TinyLlama / configured LLM
    ↓
JSON classification
    ↓
category + confidence + rationale
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from app.services.llm import generate
from app.services.vector_store import retrieve_labelled


# ============================================================
# Classification schema
# ============================================================

class Classification(BaseModel):
    """Structured Feedy classification result."""

    category: str = Field(
        description=(
            "One of: Excellent, Good, "
            "Need Improvements, Poor"
        )
    )

    aspect: str = Field(
        default="service",
        description="Main feedback aspect.",
    )

    severity: str = Field(
        default="medium",
        description="Severity of the issue.",
    )

    rationale: str = Field(
        default="",
        description="Short explanation based on the feedback.",
    )

    self_confidence: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Model confidence from 0 to 1.",
    )


# ============================================================
# System prompt
# ============================================================

CLASSIFY_SYSTEM = """
You are Feedy, an AI system that classifies customer feedback.

You MUST classify the feedback into exactly one of these categories:

1. Excellent
2. Good
3. Need Improvements
4. Poor

Definitions:

Excellent:
Very positive feedback. The customer is clearly satisfied.

Good:
Generally positive or acceptable feedback with no major problem.

Need Improvements:
Mixed or moderately negative feedback. There is a problem,
but the experience is not completely bad.

Poor:
Strongly negative feedback or a clearly unacceptable experience.

Use the provided labelled examples as precedent.

Return ONLY valid JSON.

The JSON must contain exactly these fields:

{
  "category": "Excellent | Good | Need Improvements | Poor",
  "aspect": "short aspect",
  "severity": "low | medium | high",
  "rationale": "short explanation",
  "self_confidence": 0.0
}

self_confidence must be between 0 and 1.

Do not add markdown.
Do not add ```json.
Do not explain your answer outside the JSON.
"""


# ============================================================
# Prompt builder
# ============================================================

def build_prompt(
    feedback: str,
    examples: list[dict],
) -> str:
    """Build the grounded classification prompt."""

    prompt_parts = [
        CLASSIFY_SYSTEM,
        "",
        "SIMILAR PREVIOUSLY LABELLED EXAMPLES:",
        "",
    ]

    if examples:

        for index, example in enumerate(
            examples,
            start=1,
        ):

            prompt_parts.append(
                f"Example {index}:"
            )

            prompt_parts.append(
                f"Feedback: {example.get('text', '')}"
            )

            prompt_parts.append(
                f"Label: {example.get('label', 'Good')}"
            )

            prompt_parts.append(
                f"Similarity: {example.get('score', 0)}"
            )

            prompt_parts.append("")

    else:

        prompt_parts.append(
            "No previous examples were retrieved."
        )

        prompt_parts.append("")

    prompt_parts.extend(
        [
            "NEW CUSTOMER FEEDBACK:",
            feedback,
            "",
            "Classify this feedback now.",
        ]
    )

    return "\n".join(prompt_parts)


# ============================================================
# JSON extraction
# ============================================================

def extract_json(text: str) -> dict[str, Any]:
    """Extract JSON from model output."""

    text = text.strip()

    # Direct JSON
    try:
        value = json.loads(text)

        if isinstance(value, dict):
            return value

    except json.JSONDecodeError:
        pass

    # Remove markdown fences
    cleaned = re.sub(
        r"```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    cleaned = cleaned.replace(
        "```",
        "",
    ).strip()

    try:
        value = json.loads(cleaned)

        if isinstance(value, dict):
            return value

    except json.JSONDecodeError:
        pass

    # Find first JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end > start:

        candidate = cleaned[
            start : end + 1
        ]

        try:
            value = json.loads(candidate)

            if isinstance(value, dict):
                return value

        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Model did not return valid JSON:\n{text}"
    )


# ============================================================
# Normalize category
# ============================================================

def normalize_category(value: Any) -> str:
    """Normalize the model's category."""

    text = str(value).strip()

    normalized = text.lower()

    if normalized == "excellent":
        return "Excellent"

    if normalized == "good":
        return "Good"

    if normalized in {
        "need improvement",
        "needs improvement",
        "need improvements",
        "needs improvements",
    }:
        return "Need Improvements"

    if normalized == "poor":
        return "Poor"

    # Fallback
    return "Good"


# ============================================================
# Normalize confidence
# ============================================================

def normalize_confidence(value: Any) -> float:

    try:
        score = float(value)

    except (TypeError, ValueError):
        return 0.70

    # Handle percentages
    if score > 1:
        score = score / 100

    return max(
        0.0,
        min(1.0, score),
    )


# ============================================================
# Main classifier
# ============================================================

def classify(
    feedback: str,
    k: int | None = None,
) -> Classification:
    """Classify one feedback item using RAG + LLM."""

    feedback = str(feedback).strip()

    if not feedback:
        return Classification(
            category="Good",
            aspect="unknown",
            severity="low",
            rationale="No feedback text was provided.",
            self_confidence=0.0,
        )

    # --------------------------------------------------------
    # Retrieve similar labelled examples
    # --------------------------------------------------------

    try:

        examples = retrieve_labelled(
            feedback,
            k=k,
        )

        print(
            f"[classifier] retrieved "
            f"{len(examples)} labelled examples"
        )

    except Exception as exc:

        print(
            f"[classifier] retrieval failed: {exc}"
        )

        examples = []

    # --------------------------------------------------------
    # Build prompt
    # --------------------------------------------------------

    prompt = build_prompt(
        feedback,
        examples,
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    raw_output = generate(
        prompt,
        max_new_tokens=400,
    )

    print(
        "[classifier] model response:",
        raw_output[:500],
    )

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        data = extract_json(
            raw_output
        )

    except Exception as exc:

        print(
            f"[classifier] JSON parsing failed: {exc}"
        )

        return Classification(
            category="Good",
            aspect="unknown",
            severity="medium",
            rationale=(
                "The model response could not be parsed "
                "into the expected classification format."
            ),
            self_confidence=0.30,
        )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    category = normalize_category(
        data.get("category", "Good")
    )

    aspect = str(
        data.get(
            "aspect",
            "service",
        )
    ).strip()

    severity = str(
        data.get(
            "severity",
            "medium",
        )
    ).strip().lower()

    if severity not in {
        "low",
        "medium",
        "high",
    }:
        severity = "medium"

    rationale = str(
        data.get(
            "rationale",
            "Classification based on customer feedback.",
        )
    ).strip()

    confidence = normalize_confidence(
        data.get(
            "self_confidence",
            0.70,
        )
    )

    # --------------------------------------------------------
    # Return Pydantic result
    # --------------------------------------------------------

    return Classification(
        category=category,
        aspect=aspect,
        severity=severity,
        rationale=rationale,
        self_confidence=confidence,
    )


# ============================================================
# Simple manual test
# ============================================================

if __name__ == "__main__":

    result = classify(
        "The staff were very helpful and friendly."
    )

    print()
    print("RESULT")
    print(result.model_dump_json(indent=2))