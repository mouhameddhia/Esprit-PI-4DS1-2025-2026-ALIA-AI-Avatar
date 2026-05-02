"""Application configuration for the Knowledge Retrieval Agent."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven runtime settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_dir: str = Field(default="data", description="Directory containing source PDFs.")
    vector_store_dir: str = Field(default="vector_store", description="Directory for persisted retrieval indexes.")

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model for dense retrieval.",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder reranker model.",
    )

    llm_base_url: str = Field(default="http://localhost:11434", description="Ollama endpoint.")
    llm_model: str = Field(default="llama3:8b", description="Ollama model name.")
    llm_judge_model: str = Field(default="smollm2:135m", description="Ollama model name for LLM-as-judge evaluation.")
    llm_num_ctx: int = Field(default=8192, ge=1024, description="Context window for the local LLM.")
    llm_max_output_tokens: int = Field(default=512, ge=64, description="Reserved output budget in tokens.")
    fast_answer_mode: bool = Field(default=False, description="Use extractive fast answers instead of LLM generation.")

    prompt_name: str = Field(default="pharma_answer_v1", description="Registered prompt template to use.")
    max_doc_age_days: int = Field(default=3650, ge=1, description="Soft recency horizon for metadata scoring.")
    ingestion_max_workers: int = Field(default=4, ge=1, le=32, description="Parallel workers for ingestion file loading.")
    embedding_batch_size: int = Field(default=64, ge=1, le=2048, description="Base embedding batch size.")
    embedding_min_batch_size: int = Field(default=8, ge=1, le=512, description="Minimum embedding batch size under low memory.")
    embedding_max_tokens: int = Field(default=256, ge=32, le=2048, description="Maximum tokens per chunk before embedding.")
    min_chunk_chars: int = Field(default=40, ge=1, le=5000, description="Minimum chunk length accepted for embedding.")
    max_non_printable_ratio: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Reject chunks with high non-printable character ratio.",
    )

    chunk_size: int = Field(default=900, ge=200)
    chunk_overlap: int = Field(default=120, ge=0)

    dense_top_k: int = Field(default=12, ge=1)
    sparse_top_k: int = Field(default=12, ge=1)
    hybrid_top_k: int = Field(default=10, ge=1)
    rerank_top_k: int = Field(default=6, ge=1)

    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    beta: float = Field(default=0.4, ge=0.0, le=1.0)
    gamma: float = Field(default=0.1, ge=0.0, le=1.0)

    min_top_score: float = Field(default=0.35, ge=0.0, le=1.0)
    min_mean_score: float = Field(default=0.25, ge=0.0, le=1.0)
    min_source_count: int = Field(default=1, ge=1)
    fallback_enabled: bool = Field(default=True)

    request_timeout_seconds: int = Field(default=90, ge=5)


def get_settings() -> Settings:
    """Return a new settings instance loaded from environment variables."""

    return Settings()
