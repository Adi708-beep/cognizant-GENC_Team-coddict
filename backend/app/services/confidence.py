from app.config import settings


# Ordinal position of each category on the satisfaction scale.
ORDER = {
    "Poor": 0,
    "Need Improvements": 1,
    "Good": 2,
    "Excellent": 3,
}

MAX_DISTANCE = 3

W_SELF = 0.50
W_NEIGHBOURS = 0.30
W_EXTRACTION = 0.20


def neighbour_agreement(
    predicted: str,
    neighbours: list[dict],
) -> float:
    """
    How much do the retrieved labelled examples support this prediction?

    Similarity-weighted, and ordinal-aware:
    disagreeing with a neighbour labelled "Good" when you predicted
    "Excellent" is a small penalty; disagreeing with one labelled
    "Poor" is a large one.
    """

    if not neighbours:
        return 0.5

    total_weight = 0.0
    total_score = 0.0

    for neighbour in neighbours:

        weight = max(
            0.0,
            float(neighbour.get("score", 0.0)),
        )

        if weight <= 0:
            continue

        distance = abs(
            ORDER.get(predicted, 1)
            - ORDER.get(
                neighbour.get("label", "Good"),
                1,
            )
        )

        total_score += weight * (
            1.0 - distance / MAX_DISTANCE
        )

        total_weight += weight

    return (
        round(
            total_score / total_weight,
            3,
        )
        if total_weight
        else 0.5
    )


def fuse(
    self_confidence: float,
    agreement: float,
    extraction_quality: float,
) -> dict:

    quality = max(
        0.0,
        min(1.0, extraction_quality),
    )

    final = (
        W_SELF * self_confidence
        + W_NEIGHBOURS * agreement
        + W_EXTRACTION * quality
    )

    final = round(
        max(0.0, min(1.0, final)),
        3,
    )

    return {
        "llm_self": round(self_confidence, 3),
        "neighbour_agreement": round(agreement, 3),
        "extraction_quality": round(quality, 3),
        "final": final,
    }


def route(
    category: str,
    final_confidence: float,
) -> tuple[bool, bool]:
    """
    Returns:
        (needs_review, alert)
    """

    needs_review = (
        final_confidence < settings.review_threshold
    )

    alert = (
        category == "Poor"
        and final_confidence >= settings.alert_threshold
    )

    return needs_review, alert