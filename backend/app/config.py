from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "feedy"
    default_company_id: str = "demo-co"

    llm_provider_order: str = "huggingface,groq,gemini"
    max_new_tokens: int = 700

    use_local_llm: bool = False

    hf_token: str = ""
    hf_repo_id: str = "Qwen/Qwen2.5-7B-Instruct"
    hf_local_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    pinecone_api_key: str = ""
    pinecone_index_name: str = "feedy-index"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    labelled_namespace: str = "labelled"
    knowledge_namespace: str = "knowledge"

    local_vector_path: str = "data/local_vectors.npz"

    tesseract_cmd: str = ""
    ocr_dpi: int = 300
    enable_vision_ocr: bool = False

    review_threshold: float = 0.60
    alert_threshold: float = 0.80

    max_records_per_upload: int = 60
    llm_concurrency: int = 4
    retrieval_k: int = 5

    cors_origins: str = "http://localhost:5173"

    @property
    def provider_order(self) -> list[str]:
        return [
            p.strip().lower()
            for p in self.llm_provider_order.split(",")
            if p.strip()
        ]

    @property
    def cors_origins_list(self) -> list[str]:
        return [
            o.strip()
            for o in self.cors_origins.split(",")
            if o.strip()
        ]


settings = Settings()