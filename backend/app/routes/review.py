from fastapi import APIRouter

router = APIRouter()


@router.get("/test")
async def review_test():
    return {
        "status": "ok",
        "route": "review",
    }