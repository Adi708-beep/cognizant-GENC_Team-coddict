from fastapi import APIRouter

router = APIRouter()


@router.get("/test")
async def knowledge_test():
    return {
        "status": "ok",
        "route": "knowledge",
    }