from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================
# PROJECT ROOT
# ============================================================

# config.py is:
#
# backend/app/config.py
#
# parents[0] = app
# parents[1] = backend

BACKEND_DIR = Path(__file__).resolve().parents[1]

ENV_FILE = BACKEND_DIR / ".env"


print("[config] backend directory:", BACKEND_DIR)
print("[config] env file:", ENV_FILE)
print("[config] env file exists:", ENV_FILE.exists())


# ============================================================
# SETTINGS
# ============================================================

class Settings(BaseSettings):

    # --------------------------------------------------------
    # Pydantic settings configuration
    # --------------------------------------------------------

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------

    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "feedy"
    default_company_id: str = "demo-co"

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    llm_provider_order: str = (
        "huggingface,groq,gemini"
    )

    max_new_tokens: int = 700

    use_local_llm: bool = False

    hf_token: str = ""

    hf_repo_id: str = (
        "Qwen/Qwen2.5-7B-Instruct"
    )

    hf_local_model: str = (
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

    groq_api_key: str = ""

    groq_model: str = (
        "llama-3.3-70b-versatile"
    )

    gemini_api_key: str = ""

    gemini_model: str = (
        "gemini-2.5-flash"
    )

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    embedding_dim: int = 384

    # --------------------------------------------------------
    # Pinecone
    # --------------------------------------------------------

    pinecone_api_key: str = ""

    pinecone_index_name: str = "feedy-index"

    pinecone_cloud: str = "aws"

    pinecone_region: str = "us-east-1"

    labelled_namespace: str = "labelled"

    knowledge_namespace: str = "knowledge"

    local_vector_path: str = (
        "data/local_vectors.npz"
    )

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    tesseract_cmd: str = ""

    ocr_dpi: int = 300

    enable_vision_ocr: bool = False

    # --------------------------------------------------------
    # Scoring
    # --------------------------------------------------------

    review_threshold: float = 0.60

    alert_threshold: float = 0.80

    # --------------------------------------------------------
    # Limits
    # --------------------------------------------------------

    max_records_per_upload: int = 60

    llm_concurrency: int = 4

    retrieval_k: int = 5

    # --------------------------------------------------------
    # CORS
    # --------------------------------------------------------

    cors_origins: str = (
        "http://localhost:5173"
    )

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    @property
    def provider_order(self) -> list[str]:

        return [
            provider.strip().lower()
            for provider
            in self.llm_provider_order.split(",")
            if provider.strip()
        ]

    @property
    def cors_origins_list(self) -> list[str]:

        return [
            origin.strip()
            for origin
            in self.cors_origins.split(",")
            if origin.strip()
        ]


# ============================================================
# GLOBAL SETTINGS INSTANCE
# ============================================================

settings = Settings()


# ============================================================
# DEBUG INFORMATION
# ============================================================

print()
print("[config] ========================================")
print("[config] Feedy configuration loaded")
print("[config] ========================================")

print(
    "[config] USE_LOCAL_LLM =",
    settings.use_local_llm,
)

print(
    "[config] HF_LOCAL_MODEL =",
    settings.hf_local_model,
)

print(
    "[config] HF_TOKEN loaded =",
    bool(settings.hf_token),
)

print(
    "[config] EMBEDDING_MODEL =",
    settings.embedding_model,
)

print(
    "[config] PINECONE configured =",
    bool(settings.pinecone_api_key),
)

print("[config] ========================================")