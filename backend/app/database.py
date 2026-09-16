from motor.motor_asyncio import AsyncIOMotorClient

from app.config import settings


class _DB:
    client: AsyncIOMotorClient | None = None
    db = None


_state = _DB()


async def connect_to_mongo():
    _state.client = AsyncIOMotorClient(
        settings.mongodb_uri,
        serverSelectionTimeoutMS=5000,
    )

    _state.db = _state.client[settings.mongodb_db_name]

    await _state.db.command("ping")

    await _state.db.feedback.create_index("created_at")
    await _state.db.feedback.create_index("category")
    await _state.db.feedback.create_index("needs_review")
    await _state.db.feedback.create_index("company_id")
    await _state.db.feedback.create_index("text_hash")

    await _state.db.schema_maps.create_index("created_at")

    print("[db] connected to", settings.mongodb_db_name)


async def close_mongo_connection():
    if _state.client:
        _state.client.close()

    print("[db] closed")


def get_database():
    if _state.db is None:
        raise RuntimeError("Mongo is not connected yet")

    return _state.db