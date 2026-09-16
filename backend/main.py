from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import close_mongo_connection, connect_to_mongo
from app.routes import analytics, chat, feedback, insights, knowledge, review
from app.services.llm import describe_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_to_mongo()

    yield

    await close_mongo_connection()


app = FastAPI(
    title="Feedy API",
    version="2.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(
    feedback.router,
    prefix="/api/feedback",
    tags=["Feedback"],
)

app.include_router(
    review.router,
    prefix="/api/review",
    tags=["Review"],
)

app.include_router(
    analytics.router,
    prefix="/api/analytics",
    tags=["Analytics"],
)

app.include_router(
    insights.router,
    prefix="/api/insights",
    tags=["Insights"],
)

app.include_router(
    knowledge.router,
    prefix="/api/knowledge",
    tags=["Knowledge"],
)

app.include_router(
    chat.router,
    prefix="/api/chat",
    tags=["Chat"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "Feedy API",
        "orchestration": "langchain",
        "models": describe_models(),
        "embeddings": settings.embedding_model,
        "vector_backend": (
            "pinecone"
            if settings.pinecone_api_key
            else "local"
        ),
        "vision_ocr": settings.enable_vision_ocr,
    }