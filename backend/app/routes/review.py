from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.database import get_database
from app.models import ReviewRequest, ReviewResponse
from app.services.vector_store import add_labelled

router = APIRouter()


@router.post("/{record_id}", response_model=ReviewResponse)
async def submit_review(record_id: str, payload: ReviewRequest):
    db = get_database()
    document = await db.feedback.find_one({"_id": record_id})
    if not document:
        raise HTTPException(404, "Record not found.")

    note = payload.note.strip()

    await db.feedback.update_one(
        {"_id": record_id},
        {"$set": {
            "reviewed": True,
            "reviewed_category": payload.corrected_category,
            "category": payload.corrected_category,
            "needs_review": False,
            "confidence": 1.0,
            "rationale": f"Verified by a human reviewer. {note}".strip(),
            "reviewed_at": datetime.now(timezone.utc),
        }},
    )

    await db.reviews.insert_one({
        "record_id": record_id,
        "from_category": document.get("category"),
        "to_category": payload.corrected_category,
        "note": note,
        "created_at": datetime.now(timezone.utc),
    })

    # The point of the whole feature: the correction becomes retrieval precedent.
    added = True
    try:
        add_labelled(f"human-{record_id}", document["raw_text"],
                     payload.corrected_category, origin="human")
    except Exception as exc:  # noqa: BLE001
        print(f"[review] could not write back to the vector store: {exc}")
        added = False

    return ReviewResponse(message="Review saved", record_id=record_id,
                           corrected_category=payload.corrected_category,
                           added_to_retrieval=added)


@router.get("/history")
async def review_history(limit: int = 50):
    cursor = get_database().reviews.find().sort("created_at", -1).limit(limit)
    return [{**item, "_id": str(item["_id"])} async for item in cursor]