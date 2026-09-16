from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


Category = Literal[
    "Excellent",
    "Good",
    "Need Improvements",
    "Poor",
]

Aspect = Literal[
    "Food",
    "Service",
    "Price",
    "Cleanliness",
    "Waiting Time",
    "Product",
    "Support",
    "Other",
]

Severity = Literal[
    "Low",
    "Medium",
    "High",
]

CanonicalField = Literal[
    "customer_name",
    "date",
    "overall_rating",
    "food_quality",
    "service",
    "price_value",
    "cleanliness",
    "waiting_time",
    "product_quality",
    "support",
    "comments",
    "unmapped",
]

ColumnKind = Literal[
    "rating",
    "text",
    "identifier",
    "date",
    "other",
]


CATEGORIES: list[str] = [
    "Excellent",
    "Good",
    "Need Improvements",
    "Poor",
]

ASPECTS: list[str] = [
    "Food",
    "Service",
    "Price",
    "Cleanliness",
    "Waiting Time",
    "Product",
    "Support",
    "Other",
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ------------------------------------------------------- parsed by LangChain

class Classification(BaseModel):
    """The contract. services/classifier.py parses model output into this."""

    category: Category = Field(
        description="Exactly one of: Excellent, Good, Need Improvements, Poor."
    )

    self_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Your own certainty from 0.0 to 1.0. Be honest: below 0.6 "
            "when the text is short, ambiguous or garbled."
        ),
    )

    rationale: str = Field(
        description="One plain-language sentence referencing the customer's own words."
    )

    evidence: list[str] = Field(
        default_factory=list,
        description=(
            "One to three verbatim spans copied from the feedback, "
            "2 to 12 words each."
        ),
    )

    aspect: Aspect = Field(
        default="Other",
        description="The single aspect the feedback is mostly about.",
    )

    severity: Severity = Field(
        default="Low",
        description="Business impact of the issue: Low, Medium or High.",
    )


class ColumnMapping(BaseModel):
    source_column: str = Field(
        description="The column header exactly as it appeared."
    )

    canonical_field: CanonicalField = Field(
        description="The Feedy field this column means, or 'unmapped' if none fits."
    )

    kind: ColumnKind = Field(
        description="rating, text, identifier, date or other."
    )

    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How sure you are about this mapping.",
    )

    reason: str = Field(
        default="",
        description="Six words or fewer.",
    )


class SchemaMap(BaseModel):
    mappings: list[ColumnMapping] = Field(
        default_factory=list,
        description="One entry for every column given, in the same order.",
    )


class Issue(BaseModel):
    title: str = Field(
        description="The recurring problem, in under ten words."
    )

    aspect: Aspect = Field(
        default="Other"
    )

    severity: Severity = Field(
        default="Medium"
    )

    recommendation: str = Field(
        description="One concrete action management can take."
    )

    supporting_quotes: list[str] = Field(
        default_factory=list,
        description="Up to two short verbatim customer phrases showing the problem.",
    )


class BusinessInsights(BaseModel):
    headline: str = Field(
        description="One sentence summarising what customers are saying."
    )

    praise: list[str] = Field(
        default_factory=list,
        description="Up to four things customers consistently like.",
    )

    problems: list[Issue] = Field(
        default_factory=list,
        description="Up to five recurring problems, worst first.",
    )

    priorities: list[str] = Field(
        default_factory=list,
        description="Up to three things management should fix first, in order.",
    )


# ------------------------------------------------------- stored / returned

class Neighbour(BaseModel):
    text: str
    label: Category
    score: float


class ConfidenceBreakdown(BaseModel):
    llm_self: float
    neighbour_agreement: float
    extraction_quality: float
    final: float


class FeedbackRecord(BaseModel):
    id: Optional[str] = None

    company_id: str = "demo-co"

    raw_text: str
    cleaned_text: str
    text_hash: str

    source_file: str
    source_type: str  # pdf | image | spreadsheet
    extraction_method: str  # digital | ocr | vision | spreadsheet
    extraction_confidence: float

    normalised: dict[str, Any] = Field(
        default_factory=dict
    )

    extra: dict[str, Any] = Field(
        default_factory=dict
    )

    category: Category
    confidence: float

    confidence_breakdown: ConfidenceBreakdown

    rationale: str

    evidence: list[str] = Field(
        default_factory=list
    )

    aspect: Aspect = "Other"

    severity: Severity = "Low"

    neighbours: list[Neighbour] = Field(
        default_factory=list
    )

    needs_review: bool = False
    alert: bool = False

    reviewed: bool = False

    reviewed_category: Optional[Category] = None

    created_at: datetime = Field(
        default_factory=_now
    )


class UploadResponse(BaseModel):
    message: str

    source_file: str
    source_type: str

    extraction_method: str
    extraction_confidence: float

    records: int

    schema_map: list[ColumnMapping] = Field(
        default_factory=list
    )

    schema_cached: bool = False

    results: list[FeedbackRecord] = Field(
        default_factory=list
    )


class CategoryCount(BaseModel):
    category: Category
    count: int


class AnalyticsSummary(BaseModel):
    total: int

    by_category: list[CategoryCount]

    mean_confidence: float

    confidence_buckets: dict[str, int]

    needs_review: int

    alerts: int

    top_aspects: list[str]

    extraction_mix: dict[str, int]


class InsightsResponse(BaseModel):
    generated_from: int

    headline: str

    praise: list[str] = Field(
        default_factory=list
    )

    problems: list[Issue] = Field(
        default_factory=list
    )

    priorities: list[str] = Field(
        default_factory=list
    )

    aspect_counts: dict[str, int] = Field(
        default_factory=dict
    )

    negative_aspect_counts: dict[str, int] = Field(
        default_factory=dict
    )


class ReviewRequest(BaseModel):
    corrected_category: Category

    note: str = ""


class ReviewResponse(BaseModel):
    message: str

    record_id: str

    corrected_category: Category

    added_to_retrieval: bool


class ChatRequest(BaseModel):
    question: str


class ChatAnswer(BaseModel):
    """Parsed from the model in services/rag_chat.py."""

    answer: str = Field(
        description="The answer, grounded only in the supplied context."
    )

    sources: list[str] = Field(
        default_factory=list,
        description="Names of the documents you used.",
    )


class ChatResponse(BaseModel):
    answer: str

    sources: list[str] = Field(
        default_factory=list
    )


class KnowledgeUploadResponse(BaseModel):
    message: str

    document_name: str

    chunks: int