from functools import cached_property

from pydantic_settings import BaseSettings, SettingsConfigDict

NVIDIA_API_CATALOG_URL = "https://integrate.api.nvidia.com/v1"


class Settings(BaseSettings):
    """Runtime configuration. Every value can be overridden with an environment variable."""

    app_env: str = "local"
    log_level: str = "INFO"
    database_url: str = "sqlite:///./nvidia_lab.db"
    redis_url: str = "redis://localhost:6379/0"
    # Empty QDRANT_URL runs Qdrant in-process (in memory) - used by tests and quick demos.
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    backend_cors_origins: str = "http://localhost:3000"
    # Create tables on startup. Production deployments set this to false and run `alembic upgrade head`.
    auto_create_tables: bool = True

    # NVIDIA inference. Works with the hosted NVIDIA API catalog (build.nvidia.com) or a self-hosted
    # NIM exposing the OpenAI-compatible API. Live calls are enabled only when an API key is set or a
    # self-hosted base URL is configured.
    nvidia_nim_base_url: str = ""
    nvidia_nim_api_key: str = ""
    # Defaults verified against the NVIDIA API catalog (2026-11). Catalog models are retired over time:
    # check https://build.nvidia.com or GET {base_url}/models and override via environment variables.
    nvidia_chat_model: str = "nvidia/nemotron-3.5-bolt-e4b"
    nvidia_embed_model: str = "nvidia/nemotron-3.5-embed-1b"
    # Empty = use the local reranker. Set to a reranking NIM endpoint (hosted or self-hosted) to enable.
    nvidia_rerank_url: str = ""
    nvidia_rerank_model: str = ""
    nvidia_timeout_seconds: float = 60.0

    # RAG limits
    rag_max_upload_mb: int = 15
    rag_chunk_chars: int = 900
    rag_chunk_overlap: int = 150
    rag_top_k: int = 5

    # Live benchmark guard rails (protect cost and shared endpoints)
    bench_max_requests: int = 64
    bench_max_concurrency: int = 16

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @cached_property
    def cors_origins(self) -> list[str]:
        return [x.strip() for x in self.backend_cors_origins.split(",") if x.strip()]

    @property
    def llm_base_url(self) -> str:
        return (self.nvidia_nim_base_url or NVIDIA_API_CATALOG_URL).rstrip("/")

    @property
    def llm_enabled(self) -> bool:
        self_hosted = bool(self.nvidia_nim_base_url) and not self.nvidia_nim_base_url.startswith(
            NVIDIA_API_CATALOG_URL
        )
        return bool(self.nvidia_nim_api_key) or self_hosted

    @property
    def rerank_enabled(self) -> bool:
        # The hosted reranker needs a key; self-hosted rerankers are configured via NVIDIA_RERANK_URL.
        return bool(self.nvidia_rerank_url) and (
            bool(self.nvidia_nim_api_key) or "api.nvidia.com" not in self.nvidia_rerank_url
        )


settings = Settings()
