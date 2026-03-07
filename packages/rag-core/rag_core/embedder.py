"""Batch embedding for chunks via OpenAI or local embedding models.

Supports:
- OpenAI embeddings (text-embedding-3-small, etc.)
- Local embeddings via sentence-transformers (e.g., intfloat/multilingual-e5-large)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rag_core.config import get_settings, get_local_embedding_model, make_openai_client
from rag_core.models import Chunk

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)


def _embed_texts_openai(texts: list[str], client: OpenAI, model: str) -> list[list[float]]:
    """Embed texts using OpenAI API."""
    response = client.embeddings.create(
        model=model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def _embed_texts_local(texts: list[str], model_name: str, is_query: bool = False) -> list[list[float]]:
    """Embed texts using local sentence-transformers model.

    For E5 models, adds appropriate prefix:
    - Queries: "query: " prefix
    - Passages: "passage: " prefix
    """
    model = get_local_embedding_model(model_name)

    # Check if this is an E5 model (requires prefix)
    is_e5 = "e5" in model_name.lower()

    if is_e5:
        prefix = "query: " if is_query else "passage: "
        texts = [prefix + t for t in texts]

    embeddings = model.encode(texts, convert_to_numpy=True)
    return [emb.tolist() for emb in embeddings]


def embed_texts(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """Embed a list of texts using configured provider.

    Args:
        texts: List of text strings to embed.
        is_query: Whether these are query texts (affects E5 model prefix).

    Returns:
        List of embedding vectors.
    """
    if not texts:
        return []

    cfg = get_settings()
    provider = cfg.embedding.provider.lower()

    if provider == "local":
        return _embed_texts_local(texts, cfg.embedding.model, is_query=is_query)
    elif provider == "openai":
        client = make_openai_client(cfg)
        return _embed_texts_openai(texts, client, cfg.embedding.model)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


def embed_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Batch embed chunks using configured embedding provider.

    Uses enriched_content (context + content) if available.
    Sets chunk.embedding for each chunk.
    """
    if not chunks:
        return chunks

    cfg = get_settings()
    texts = [chunk.enriched_content for chunk in chunks]

    try:
        embeddings = embed_texts(texts, is_query=False)

        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]

        logger.info("Embedded %d chunks (%s via %s)",
                    len(chunks), cfg.embedding.model, cfg.embedding.provider)

    except Exception as e:
        logger.error("Failed to embed chunks: %s", e)
        raise

    return chunks
