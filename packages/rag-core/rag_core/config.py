"""Agentic Graph RAG configuration via Pydantic Settings."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, PydanticBaseSettingsSource
from typing import Any


def _find_env_file() -> Path | None:
    """Find .env file starting from current directory and going up."""
    cwd = Path.cwd()
    for path in [cwd] + list(cwd.parents):
        env_path = path / ".env"
        if env_path.exists():
            return env_path
    return None


def _load_dotenv() -> dict[str, str]:
    """Load .env file manually and return as dict."""
    env_file = _find_env_file()
    if not env_file:
        return {}

    env_vars = {}
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # Remove inline comments
                if " #" in value:
                    value = value.split(" #")[0].strip()
                # Remove quotes
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                env_vars[key] = value
    return env_vars


# Load .env file into environment at module import time
_env_vars = _load_dotenv()
for _key, _value in _env_vars.items():
    if _key not in os.environ:
        os.environ[_key] = _value


class Neo4jSettings(BaseSettings):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "neo4j"

    model_config = {"env_prefix": "NEO4J_"}


class EmbeddingSettings(BaseSettings):
    provider: str = "openai"  # "openai" or "local"
    model: str = "text-embedding-3-small"  # OpenAI model or local HuggingFace model name
    dimensions: int = 1536  # 1536 for text-embedding-3-small, 1024 for multilingual-e5-large

    model_config = {"env_prefix": "EMBEDDING_"}


class OpenAISettings(BaseSettings):
    api_key: str = ""
    base_url: str = ""  # LiteLLM proxy: e.g. "http://localhost:4000/v1"
    llm_model: str = "gpt-4o"
    llm_model_mini: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    max_context_tokens: int = 100000  # Reserve ~28K for response (DeepSeek 128K limit)

    model_config = {"env_prefix": "OPENAI_"}


class IndexingSettings(BaseSettings):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    skeleton_beta: float = 0.25
    knn_k: int = 10
    pagerank_damping: float = 0.85

    model_config = {"env_prefix": "INDEXING_"}


class RetrievalSettings(BaseSettings):
    top_k_vector: int = 10
    top_k_final: int = 10
    vector_threshold: float = 0.5
    max_hops: int = 3
    ppr_alpha: float = 0.15

    model_config = {"env_prefix": "RETRIEVAL_"}


class AgentSettings(BaseSettings):
    max_retries: int = 2
    relevance_threshold: float = 2.0

    model_config = {"env_prefix": "AGENT_"}


class Settings(BaseSettings):
    neo4j: Neo4jSettings = Neo4jSettings()
    openai: OpenAISettings = OpenAISettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    indexing: IndexingSettings = IndexingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    agent: AgentSettings = AgentSettings()

    log_level: str = "INFO"

    model_config = {"extra": "ignore"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Create and cache settings instance loading from environment.

    Provides backward compatibility for legacy OPENAI_EMBEDDING_* variables.
    """
    settings = Settings()

    # Backward compatibility: migrate old OPENAI_EMBEDDING_* to new EMBEDDING_*
    # Only override if new vars are not set and old vars exist
    old_model = os.getenv("OPENAI_EMBEDDING_MODEL")
    old_dims = os.getenv("OPENAI_EMBEDDING_DIMENSIONS")

    if old_model and "EMBEDDING_MODEL" not in os.environ:
        settings.embedding.model = old_model
    if old_dims and "EMBEDDING_DIMENSIONS" not in os.environ:
        try:
            settings.embedding.dimensions = int(old_dims)
        except ValueError:
            pass

    return settings


def make_openai_client(settings: Settings | None = None):
    """Create OpenAI client with optional LiteLLM proxy support.

    If OPENAI_BASE_URL is set, uses it as base_url (e.g. LiteLLM proxy).
    If api_key is empty and base_url is set, uses "none" as placeholder.
    Raises ValueError if neither api_key nor base_url is configured.
    """
    from openai import OpenAI

    cfg = settings or get_settings()
    if not cfg.openai.api_key and not cfg.openai.base_url:
        raise ValueError(
            "OPENAI_API_KEY or OPENAI_BASE_URL must be set. "
            "Set OPENAI_API_KEY for direct OpenAI access, or "
            "OPENAI_BASE_URL for a LiteLLM proxy."
        )
    kwargs: dict[str, str] = {}
    if cfg.openai.api_key:
        kwargs["api_key"] = cfg.openai.api_key
    elif cfg.openai.base_url:
        kwargs["api_key"] = "none"  # LiteLLM proxy doesn't need real key
    if cfg.openai.base_url:
        kwargs["base_url"] = cfg.openai.base_url
    return OpenAI(**kwargs)


# Local embedding model cache (lazy-loaded)
_local_embedding_model = None


def get_local_embedding_model(model_name: str | None = None):
    """Get or load the local sentence-transformers embedding model.

    Uses LRU-style caching via global variable.
    """
    global _local_embedding_model  # noqa: PLW0603
    if _local_embedding_model is None:
        from sentence_transformers import SentenceTransformer
        cfg = get_settings()
        model = model_name or cfg.embedding.model
        _local_embedding_model = SentenceTransformer(model)
    return _local_embedding_model
