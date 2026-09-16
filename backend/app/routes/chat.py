from fastapi import APIRouter

router = APIRouter()


@router.get("/test")
async def chat_test():
    return {
        "status": "ok",
        "route": "chat",
    }