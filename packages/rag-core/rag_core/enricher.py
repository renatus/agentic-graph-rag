"""Contextual enrichment for chunks via LLM.

From RAG 2.0 — generates per-chunk context explaining its role
within the document using OpenAI chat completions.

Uses embedding-based centroid selection to find representative chunks
for document summarization, ensuring the summary captures the most
important content rather than random excerpts.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from rag_core.config import get_settings, make_openai_client
from rag_core.models import Chunk

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

# Minimum chunks required to use centroid-based selection
MIN_CHUNKS_FOR_CENTROID = 5
# Number of representative chunks to select for summarization
REPRESENTATIVE_CHUNK_COUNT = 5


def enrich_chunks(
    chunks: list[Chunk], document_summary: str = "",
) -> list[Chunk]:
    """Enrich chunks with contextual information via LLM.

    If no document_summary provided, generates one using embedding-based
    representative chunk selection. For each chunk, calls OpenAI to generate
    1-2 sentence context explaining its role within the document.

    Args:
        chunks: List of chunks to enrich. Must have embeddings if document_summary
                is not provided or too long.
        document_summary: Pre-computed document summary. If >2000 chars or empty,
                          will be regenerated using representative chunks.

    Returns:
        List of chunks with .context field populated.
    """
    if not chunks:
        return chunks

    cfg = get_settings()
    client = make_openai_client(cfg)

    # Generate summary if not provided or too long
    if not document_summary or len(document_summary) > 2000:
        document_summary = _generate_summary_from_chunks(chunks, client, cfg.openai.llm_model)
        logger.info("Generated document summary: %s", document_summary[:100])

    enriched: list[Chunk] = []
    for i, chunk in enumerate(chunks):
        try:
            context = _generate_context(
                chunk.content, document_summary, client, cfg.openai.llm_model,
            )
            chunk.context = context
            logger.debug("Enriched chunk %d/%d: %s", i + 1, len(chunks), context[:50])

            if i < len(chunks) - 1:
                time.sleep(0.1)

        except Exception as e:
            logger.warning("Failed to enrich chunk %d: %s", i, e)

        enriched.append(chunk)

    logger.info("Enriched %d chunks", len(enriched))
    return enriched


def find_representative_chunks(chunks: list[Chunk], top_k: int = REPRESENTATIVE_CHUNK_COUNT) -> list[Chunk]:
    """Find chunks most representative of the document using centroid similarity.

    Computes the centroid (mean embedding) of all chunks and selects the chunks
    closest to this centroid. This identifies content that best represents the
    document's core themes and topics.

    Args:
        chunks: List of chunks with embeddings populated.
        top_k: Number of representative chunks to return.

    Returns:
        List of top_k chunks sorted by similarity to centroid (most similar first).

    Raises:
        ValueError: If chunks don't have embeddings.
    """
    if not chunks:
        return []

    # Check for embeddings
    chunks_with_embeddings = [c for c in chunks if c.embedding is not None]
    if len(chunks_with_embeddings) < MIN_CHUNKS_FOR_CENTROID:
        logger.warning(
            "Only %d chunks have embeddings (need %d), using first %d chunks",
            len(chunks_with_embeddings), MIN_CHUNKS_FOR_CENTROID, top_k
        )
        return chunks[:top_k]

    embeddings = np.array([c.embedding for c in chunks_with_embeddings])

    # Compute centroid (mean of all embeddings)
    centroid = embeddings.mean(axis=0)

    # Compute cosine similarity to centroid for each chunk
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(centroid)
    # Avoid division by zero
    norms = np.maximum(norms, 1e-10)
    similarities = np.dot(embeddings, centroid) / norms

    # Get indices of top-k most similar chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]  # Descending order

    representative = [chunks_with_embeddings[i] for i in top_indices]
    logger.debug(
        "Selected %d representative chunks with similarities: %s",
        len(representative),
        [f"{similarities[i]:.3f}" for i in top_indices]
    )

    return representative


def _generate_summary_from_chunks(
    chunks: list[Chunk], client: OpenAI, model: str,
) -> str:
    """Generate document summary using embedding-based representative selection.

    If chunks have embeddings, uses centroid similarity to find the most
    representative chunks. Otherwise falls back to first few chunks.

    Args:
        chunks: List of chunks (preferably with embeddings).
        client: OpenAI client for LLM calls.
        model: Model name for completions.

    Returns:
        2-3 sentence document summary.
    """
    # Try to use representative chunks if embeddings available
    chunks_with_embeddings = [c for c in chunks if c.embedding is not None]

    if len(chunks_with_embeddings) >= MIN_CHUNKS_FOR_CENTROID:
        selected_chunks = find_representative_chunks(chunks, REPRESENTATIVE_CHUNK_COUNT)
        source_desc = "representative"
    else:
        # Fallback to first few chunks
        selected_chunks = chunks[:REPRESENTATIVE_CHUNK_COUNT]
        source_desc = "first"
        logger.info("Using %s %d chunks for summary (no embeddings)", source_desc, len(selected_chunks))

    combined = "\n\n".join(c.content for c in selected_chunks)

    prompt = (
        f"Here are {source_desc} sections from a document:\n\n"
        f"{combined[:3000]}\n\n"
        f"Write 2-3 sentences summarizing what this document is about. "
        f"Focus on the main purpose, key topics, and document type."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content or "Unknown document"
    except Exception as e:
        logger.warning("Failed to generate summary: %s", e)
        return "Document"


def _generate_context(
    chunk_content: str, document_summary: str, client: OpenAI, model: str,
) -> str:
    """Generate 1-2 sentence context for a chunk."""
    prompt = (
        f"Here's the document: {document_summary}\n\n"
        f"Here's a chunk from the document:\n\n"
        f"{chunk_content[:500]}\n\n"
        f"Write 1-2 sentences explaining the context of this chunk "
        f"within the document. Be specific and concise."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning("Failed to generate context: %s", e)
        return ""
