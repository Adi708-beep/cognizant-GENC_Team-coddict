import asyncio
import uuid
from datetime import datetime, timezone

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Query,
    UploadFile,
)

from app.config import settings
from app.database import get_database
from app.models import (
    ColumnMapping,
    FeedbackRecord,
    UploadResponse,
)
from app.services import cleaning
from app.services import extraction as ex
from app.services import schema_mapper
from app.services.classifier import classify
from app.services.confidence import (
    fuse,
    neighbour_agreement,
    route,
)


router = APIRouter()


PDF_EXT = (
    ".pdf",
)

IMAGE_EXT = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".tif",
    ".tiff",
)

TABLE_EXT = (
    ".csv",
    ".xlsx",
    ".xls",
)


MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


def _suffix(name: str) -> str:
    return (
        "."
        + name.rsplit(".", 1)[-1].lower()
        if "." in name
        else ""
    )


async def _analyse_one(
    item: dict,
    extraction_confidence: float,
    semaphore: asyncio.Semaphore,
) -> dict:

    async with semaphore:

        analysis, neighbours = await asyncio.to_thread(
            classify,
            item["text"],
        )

        agreement = neighbour_agreement(
            analysis.category,
            neighbours,
        )

        breakdown = fuse(
            analysis.self_confidence,
            agreement,
            extraction_confidence,
        )

        needs_review, alert = route(
            analysis.category,
            breakdown["final"],
        )

        return {
            "item": item,
            "analysis": analysis,
            "neighbours": neighbours,
            "breakdown": breakdown,
            "needs_review": needs_review,
            "alert": alert,
        }


@router.post(
    "/upload",
    response_model=UploadResponse,
)
async def upload_feedback(
    file: UploadFile = File(...),
):

    content = await file.read()

    if not content:
        raise HTTPException(
            400,
            "Empty file.",
        )

    filename = file.filename or "upload"

    suffix = _suffix(filename)

    db = get_database()

    schema_rows: list[ColumnMapping] = []

    schema_cached = False

    # ---------------------------------------------------------
    # 1. extract, and for tables understand the schema
    # ---------------------------------------------------------

    if suffix in PDF_EXT:

        result = ex.extract_pdf(
            content
        )

        source_type = "pdf"

        extraction_method = result.method

        extraction_confidence = result.confidence

        items = [
            {
                "text": text,
                "normalised": {},
                "extra": {},
            }
            for text in cleaning.split_responses(
                result.text
            )
        ]

    elif suffix in IMAGE_EXT:

        result = ex.extract_image(
            content,
            MIME.get(
                suffix,
                "image/png",
            ),
        )

        source_type = "image"

        extraction_method = result.method

        extraction_confidence = result.confidence

        items = [
            {
                "text": text,
                "normalised": {},
                "extra": {},
            }
            for text in cleaning.split_responses(
                result.text
            )
        ]

    elif suffix in TABLE_EXT:

        try:

            frame = ex.read_table(
                content,
                filename,
            )

        except Exception as exc:

            raise HTTPException(
                400,
                f"Could not read that table: {exc}",
            ) from exc

        if frame.empty:
            raise HTTPException(
                422,
                "That spreadsheet has no rows.",
            )

        columns = [
            str(c)
            for c in frame.columns
        ]

        samples = frame.head(
            3
        ).to_dict(
            orient="records"
        )

        schema_map, schema_cached = (
            await schema_mapper.get_schema_map(
                db,
                columns,
                samples,
            )
        )

        schema_rows = schema_map.mappings

        items = schema_mapper.normalise_rows(
            frame,
            schema_map,
        )

        source_type = "spreadsheet"

        extraction_method = "spreadsheet"

        extraction_confidence = 1.0

    else:

        raise HTTPException(
            400,
            "Unsupported file type. Use PDF, PNG, JPG, CSV or XLSX.",
        )

    # ---------------------------------------------------------
    # de-duplicate whatever the front of the pipeline produced
    # ---------------------------------------------------------

    seen: set[str] = set()

    unique_items: list[dict] = []

    for item in items:

        key = cleaning.text_hash(
            item["text"]
        )

        if key in seen:
            continue

        seen.add(key)

        unique_items.append(
            item
        )

    if not unique_items:

        raise HTTPException(
            422,
            "No readable feedback text was found in this file. "
            "If it is a scan, try a higher-resolution copy.",
        )

    unique_items = unique_items[
        : settings.max_records_per_upload
    ]

    # ---------------------------------------------------------
    # 2. classify, concurrently but capped
    # ---------------------------------------------------------

    semaphore = asyncio.Semaphore(
        settings.llm_concurrency
    )

    outcomes = await asyncio.gather(
        *[
            _analyse_one(
                item,
                extraction_confidence,
                semaphore,
            )
            for item in unique_items
        ],
        return_exceptions=True,
    )

    # ---------------------------------------------------------
    # 3. persist
    # ---------------------------------------------------------

    now = datetime.now(
        timezone.utc
    )

    documents = []

    records: list[FeedbackRecord] = []

    for outcome in outcomes:

        if isinstance(
            outcome,
            Exception,
        ):

            print(
                f"[upload] record failed: {outcome}"
            )

            continue

        item = outcome["item"]

        analysis = outcome["analysis"]

        document = {
            "_id": uuid.uuid4().hex,

            "company_id": (
                settings.default_company_id
            ),

            "raw_text": item["text"],

            "cleaned_text": cleaning.normalise(
                item["text"]
            ),

            "text_hash": cleaning.text_hash(
                item["text"]
            ),

            "source_file": filename,

            "source_type": source_type,

            "extraction_method": extraction_method,

            "extraction_confidence": (
                extraction_confidence
            ),

            "normalised": item.get(
                "normalised",
                {},
            ),

            "extra": item.get(
                "extra",
                {},
            ),

            "category": analysis.category,

            "confidence": (
                outcome["breakdown"]["final"]
            ),

            "confidence_breakdown": (
                outcome["breakdown"]
            ),

            "rationale": analysis.rationale,

            "evidence": analysis.evidence,

            "aspect": analysis.aspect,

            "severity": analysis.severity,

            "neighbours": outcome[
                "neighbours"
            ],

            "needs_review": outcome[
                "needs_review"
            ],

            "alert": outcome[
                "alert"
            ],

            "reviewed": False,

            "reviewed_category": None,

            "created_at": now,
        }

        documents.append(
            document
        )

        records.append(
            FeedbackRecord(
                id=document["_id"],
                **{
                    k: v
                    for k, v in document.items()
                    if k != "_id"
                },
            )
        )

    if not documents:

        raise HTTPException(
            502,
            "Every record failed to classify. "
            "Check the provider keys reported by /api/health.",
        )

    await db.feedback.insert_many(
        documents
    )

    await db.uploads.insert_one(
        {
            "_id": uuid.uuid4().hex,
            "file": filename,
            "type": "feedback",
            "records": len(documents),
            "method": extraction_method,
            "created_at": now,
        }
    )

    return UploadResponse(
        message=(
            f"Analysed {len(documents)} "
            f"response(s) from {filename}"
        ),
        source_file=filename,
        source_type=source_type,
        extraction_method=extraction_method,
        extraction_confidence=extraction_confidence,
        records=len(documents),
        schema_map=schema_rows,
        schema_cached=schema_cached,
        results=records,
    )


@router.get(
    "/records"
)
async def list_records(
    category: str | None = Query(None),
    needs_review: bool | None = Query(None),
    alerts_only: bool = Query(False),
    limit: int = Query(
        100,
        le=500,
    ),
):

    query: dict = {}

    if category:
        query["category"] = category

    if needs_review is not None:
        query["needs_review"] = needs_review

    if alerts_only:
        query["alert"] = True

    cursor = (
        get_database()
        .feedback
        .find(query)
        .sort(
            "created_at",
            -1,
        )
        .limit(limit)
    )

    out = []

    async for document in cursor:

        document["id"] = str(
            document.pop("_id")
        )

        out.append(
            document
        )

    return out


@router.get(
    "/records/{record_id}"
)
async def get_record(
    record_id: str,
):

    document = await get_database().feedback.find_one(
        {
            "_id": record_id
        }
    )

    if not document:

        raise HTTPException(
            404,
            "Record not found.",
        )

    document["id"] = str(
        document.pop("_id")
    )

    return document