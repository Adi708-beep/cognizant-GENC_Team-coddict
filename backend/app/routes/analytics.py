from fastapi import APIRouter

router = APIRouter()


@router.get("/test")
async def analytics_test():
    return {
        "status": "ok",
        "route": "analytics",
    }