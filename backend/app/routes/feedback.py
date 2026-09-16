from fastapi import APIRouter

router = APIRouter()


@router.get("/test")
async def feedback_test():
    return {
        "status": "ok",
        "route": "feedback",
    }