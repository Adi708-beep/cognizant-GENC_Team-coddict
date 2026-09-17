"""
AI schema understanding.

Different businesses ship different columns. This module decides what each
column MEANS - once per distinct header layout - and then normalises every row
into one internal shape that the rest of the pipeline can rely on.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.models import ColumnMapping, SchemaMap
from app.services.llm import build_json_chain
from app.services.vector_store import embed_texts


# What Feedy understands.
# Add a line here and the whole layer learns a new field.

FIELD_HINTS: dict[str, str] = {
    "customer_name": (
        "the name or identifier of the person giving feedback"
    ),
    "date": (
        "the date of the visit, order or feedback"
    ),
    "overall_rating": (
        "an overall satisfaction score for the whole experience"
    ),
    "food_quality": (
        "quality, taste, freshness or portion of food or meals"
    ),
    "service": (
        "staff behaviour, helpfulness, attitude or service experience"
    ),
    "price_value": (
        "price, cost, billing or value for money"
    ),
    "cleanliness": (
        "hygiene, cleanliness or condition of the premises"
    ),
    "waiting_time": (
        "waiting, delay, speed of service, check-in or delivery time"
    ),
    "product_quality": (
        "quality or condition of a physical product delivered"
    ),
    "support": (
        "customer support, complaint handling or after-sales help"
    ),
    "comments": (
        "free-text opinion, remarks, suggestions or what went wrong"
    ),
}


SIMILARITY_FLOOR = 0.35


MAP_SYSTEM = """You map the columns of a customer-feedback table onto Feedy's
internal schema so that feedback from any business can be analysed the same way.

THE CANONICAL FIELDS

""" + "\n".join(
    f"- {name}: {hint}"
    for name, hint in FIELD_HINTS.items()
) + """
- unmapped: the column means something Feedy has no field for

RULES

- Return exactly one mapping for every column you are given, in the same order.
- Judge meaning, not wording. "Meal Experience", "Food Rating" and "Quality of
  Food" are all food_quality. "Remarks", "What went wrong?" and "Customer
  Opinion" are all comments.
- Use the sample values. A column of numbers 1-5 is a rating; a column of
  sentences is text; a column of names or order numbers is an identifier.
- Never force a fit. An "Order ID" or "Table Number" column is unmapped, and
  that is the correct answer - unmapped data is preserved, not discarded.
- Only one column may map to comments if several are free text; map the longest
  free-text column to comments and leave the others unmapped.
- confidence is your own certainty from 0.0 to 1.0.
- reason is six words or fewer.
"""


def signature(columns: list[str]) -> str:
    key = "|".join(
        sorted(
            str(c).strip().lower()
            for c in columns
        )
    )

    return hashlib.sha1(
        key.encode()
    ).hexdigest()


# ------------------------------------------------------------ embedding pass


def shortlist(
    columns: list[str],
    top_n: int = 3,
) -> dict[str, list[tuple[str, float]]]:
    """
    Nearest canonical fields for each column,
    by local embedding similarity.
    """

    names = list(FIELD_HINTS)

    field_text = [
        f"{name.replace('_', ' ')}: {hint}"
        for name, hint in FIELD_HINTS.items()
    ]

    field_vectors = np.array(
        embed_texts(field_text),
        dtype="float32",
    )

    column_vectors = np.array(
        embed_texts(
            [str(c) for c in columns]
        ),
        dtype="float32",
    )

    out: dict[str, list[tuple[str, float]]] = {}

    for column, vector in zip(
        columns,
        column_vectors,
    ):

        scores = field_vectors @ vector

        order = np.argsort(-scores)[:top_n]

        out[str(column)] = [
            (
                names[i],
                round(float(scores[i]), 3),
            )
            for i in order
        ]

    return out


def _heuristic_map(
    columns: list[str],
    hints: dict[str, list[tuple[str, float]]],
) -> SchemaMap:
    """
    Used only when the model call fails.
    Embeddings alone, with a floor.
    """

    mappings = []

    for column in columns:

        best_field, best_score = hints[
            str(column)
        ][0]

        mapped = (
            best_field
            if best_score >= SIMILARITY_FLOOR
            else "unmapped"
        )

        mappings.append(
            ColumnMapping(
                source_column=str(column),
                canonical_field=mapped,
                kind=(
                    "text"
                    if mapped == "comments"
                    else "other"
                ),
                confidence=round(
                    float(best_score),
                    3,
                ),
                reason="embedding fallback",
            )
        )

    return SchemaMap(
        mappings=mappings
    )


# ------------------------------------------------------------ mapping call


def _build_prompt(
    columns: list[str],
    samples: list[dict[str, Any]],
    hints: dict[str, list[tuple[str, float]]],
) -> str:

    lines = [
        "COLUMNS, with up to three sample values and the nearest canonical "
        "fields by embedding similarity:\n"
    ]

    for column in columns:

        values = []

        for row in samples[:3]:

            value = row.get(column)

            if (
                value is None
                or str(value).strip().lower()
                in ("", "nan")
            ):
                continue

            values.append(
                str(value)[:60]
            )

        candidates = ", ".join(
            f"{field} ({score})"
            for field, score
            in hints[str(column)]
        )

        lines.append(
            f'- "{column}"'
        )

        lines.append(
            f" samples: "
            f"{values if values else 'none'}"
        )

        lines.append(
            f" nearest: {candidates}"
        )

    lines.append(
        "\nMap every column listed above."
    )

    return "\n".join(lines)


def infer_schema(
    columns: list[str],
    samples: list[dict[str, Any]],
) -> SchemaMap:
    """
    Blocking.

    Call it from a thread - routes use asyncio.to_thread.
    """

    columns = [
        str(c)
        for c in columns
    ]

    hints = shortlist(columns)

    try:

        chain = build_json_chain(
            MAP_SYSTEM,
            SchemaMap,
        )

        result = chain.invoke(
            {
                "input": _build_prompt(
                    columns,
                    samples,
                    hints,
                )
            }
        )

    except Exception as exc:
        print(
            f"[schema] model mapping failed "
            f"({exc}); using embedding fallback"
        )

        return _heuristic_map(
            columns,
            hints,
        )

    # The model may skip or invent a column.
    # Reconcile against the real headers.

    by_name = {
        m.source_column.strip().lower(): m
        for m in result.mappings
    }

    reconciled = []

    for column in columns:

        mapping = by_name.get(
            column.strip().lower()
        )

        if mapping is None:

            best_field, best_score = hints[
                column
            ][0]

            mapping = ColumnMapping(
                source_column=column,
                canonical_field=(
                    best_field
                    if best_score >= SIMILARITY_FLOOR
                    else "unmapped"
                ),
                kind="other",
                confidence=round(
                    float(best_score),
                    3,
                ),
                reason="model omitted column",
            )

        mapping.source_column = column

        reconciled.append(
            mapping
        )

    return SchemaMap(
        mappings=reconciled
    )


async def get_schema_map(
    db,
    columns: list[str],
    samples: list[dict[str, Any]],
) -> tuple[SchemaMap, bool]:
    """
    Cached by header signature.

    Returns:
        (map, was_cached)
    """

    key = signature(columns)

    cached = await db.schema_maps.find_one(
        {"_id": key}
    )

    if cached:
        return (
            SchemaMap.model_validate(
                cached["map"]
            ),
            True,
        )

    schema_map = await asyncio.to_thread(
        infer_schema,
        columns,
        samples,
    )

    await db.schema_maps.insert_one(
        {
            "_id": key,
            "columns": [
                str(c)
                for c in columns
            ],
            "map": schema_map.model_dump(),
            "created_at": datetime.now(
                timezone.utc
            ),
        }
    )

    return schema_map, False


# ------------------------------------------------------------ normalization


def _is_blank(value: Any) -> bool:

    if value is None:
        return True

    text = str(value).strip()

    return (
        text == ""
        or text.lower()
        in {
            "nan",
            "none",
            "null",
            "-",
            "n/a",
        }
    )


def _plain(value: Any) -> Any:
    """
    numpy / pandas scalars are not JSON-serialisable;
    Mongo will refuse them.
    """

    if hasattr(value, "item"):

        try:
            return value.item()

        except Exception:
            return str(value)

    return value


def _rating_phrase(
    field: str,
    value: Any,
) -> str:

    label = field.replace(
        "_",
        " ",
    )

    try:
        number = float(value)

    except (TypeError, ValueError):
        return (
            f"{label.capitalize()}: {value}."
        )

    if number <= 5:
        scale = 5

    elif number <= 10:
        scale = 10

    else:
        scale = 100

    return (
        f"{label.capitalize()} was rated "
        f"{number:g} out of {scale}."
    )


def normalise_rows(
    frame: pd.DataFrame,
    schema_map: SchemaMap,
) -> list[dict[str, Any]]:
    """
    One row in, one analysable response out.

    Ratings become sentences so that the classifier,
    which reads language, can actually use them.

    Unmapped columns are kept in `extra`
    and never dropped.
    """

    lookup = {
        m.source_column: m
        for m in schema_map.mappings
    }

    out: list[dict[str, Any]] = []

    for row in frame.to_dict(
        orient="records"
    ):

        normalised: dict[str, Any] = {}
        extra: dict[str, Any] = {}

        ratings: list[str] = []
        comments: list[str] = []

        for column, raw in row.items():

            if _is_blank(raw):
                continue

            value = _plain(raw)

            mapping = lookup.get(
                str(column)
            )

            field = (
                mapping.canonical_field
                if mapping
                else "unmapped"
            )

            kind = (
                mapping.kind
                if mapping
                else "other"
            )

            if field == "unmapped":

                extra[str(column)] = value

                if (
                    kind == "text"
                    and len(
                        str(value).split()
                    ) >= 4
                ):
                    comments.append(
                        f"{column}: {value}"
                    )

                continue

            if (
                field == "comments"
                and "comments" in normalised
            ):
                normalised["comments"] = (
                    f"{normalised['comments']} | {value}"
                )
            else:
                normalised[field] = value

            if field in (
                "customer_name",
                "date",
            ):
                continue

            if (
                field == "comments"
                or kind == "text"
            ):
                comments.append(
                    str(value).strip()
                )
            else:
                ratings.append(
                    _rating_phrase(
                        field,
                        value,
                    )
                )

        text = " ".join(
            ratings + comments
        ).strip()

        if not text:
            continue

        out.append(
            {
                "text": text,
                "normalised": normalised,
                "extra": extra,
            }
        )

    return out