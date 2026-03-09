"""LLM answer generation from retrieved chunks.

From RAG 2.0 — generates answers using OpenAI chat completions
with context from retrieved search results.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import tiktoken

from rag_core.config import get_settings, make_openai_client
from rag_core.models import QAResult, SearchResult

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

_ENUM_RE = None
_TOKEN_ENCODER = None


def _is_enumeration_query(query: str) -> bool:
    """Detect enumeration/global queries that need comprehensive listing."""
    global _ENUM_RE  # noqa: PLW0603
    if _ENUM_RE is None:
        import re
        _ENUM_RE = re.compile(
            r'\b('
            r'все\b|всех\b|всё\b|перечисл|опиши все|резюмируй все|обзор\b'
            r'|list all|describe all|summarize all|overview|every\b'
            r'|все компоненты|все методы|все слои|все решения|семь\b|seven\b'
            r'|all components|all layers|all methods|all decisions'
            r')\b',
            re.IGNORECASE,
        )
    return bool(_ENUM_RE.search(query))


def _get_encoder(model: str = "deepseek-chat") -> tiktoken.Encoding:
    """Get or create tiktoken encoder for a model (cached).

    DeepSeek uses cl100k_base encoding (same as GPT-4).
    """
    global _TOKEN_ENCODER  # noqa: PLW0603
    if _TOKEN_ENCODER is None:
        # DeepSeek and most modern models use cl100k_base encoding
        _TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
    return _TOKEN_ENCODER


def count_tokens(text: str, model: str = "deepseek-chat") -> int:
    """Count tokens in text using tiktoken for accurate measurement.

    Args:
        text: Text to count tokens for.
        model: Model name (default: deepseek-chat). Uses cl100k_base encoding.

    Returns:
        Exact token count as the model would see it.
    """
    if not text:
        return 0
    encoder = _get_encoder(model)
    return len(encoder.encode(text))


def generate_answer(
    query: str, results: list[SearchResult], openai_client: OpenAI | None = None,
) -> QAResult:
    """Generate answer from query and retrieved chunks using LLM."""
    cfg = get_settings()
    if openai_client is None:
        openai_client = make_openai_client(cfg)

    if not results:
        logger.warning("No results provided for answer generation")
        return QAResult(
            answer="I don't have enough context to answer this question.",
            sources=[],
            confidence=0.0,
            query=query,
        )

    # Build context with token limit
    max_context_tokens = cfg.openai.max_context_tokens
    # Reserve tokens for system prompt, user prompt template, query
    prompt_overhead = 500 + count_tokens(query)
    available_tokens = max_context_tokens - prompt_overhead

    # Sort by score descending to prioritize most relevant chunks
    sorted_results = sorted(results, key=lambda r: r.score, reverse=True)

    context_chunks = []
    used_results = []
    total_tokens = 0
    truncated_count = 0

    for i, result in enumerate(sorted_results, start=1):
        chunk_text = f"[Chunk {i}]\n{result.chunk.enriched_content}"
        chunk_tokens = count_tokens(chunk_text)

        if total_tokens + chunk_tokens <= available_tokens:
            context_chunks.append(chunk_text)
            used_results.append(result)
            total_tokens += chunk_tokens
        else:
            truncated_count += 1

    if truncated_count > 0:
        logger.warning(
            "Context truncated: %d chunks used, %d chunks dropped "
            "(estimated %d tokens used, limit %d)",
            len(context_chunks), truncated_count, total_tokens, available_tokens
        )

    context = "\n\n".join(context_chunks)

    # Detect enumeration/global queries for specialized prompt
    is_enumeration = _is_enumeration_query(query)

    if is_enumeration:
        system_prompt = (
            "You are an expert Q&A assistant specialized in comprehensive enumeration. "
            "Your task is to extract and list EVERY distinct item, component, decision, method, "
            "or concept mentioned across ALL provided context chunks.\n\n"
            "INSTRUCTIONS:\n"
            "1. Scan ALL chunks systematically — do not stop at the first few\n"
            "2. Create a NUMBERED LIST of every distinct item found\n"
            "3. For each item, provide a brief description (1-2 sentences)\n"
            "4. Combine information from multiple chunks about the same item\n"
            "5. Do NOT say 'the document does not list' — extract items even if "
            "they are discussed narratively rather than listed explicitly\n"
            "6. Answer in the same language as the query"
        )
    else:
        system_prompt = (
            "You are a knowledgeable Q&A assistant. Synthesize information from ALL provided "
            "context chunks to give a comprehensive answer. Combine facts from different chunks "
            "when needed. If some details are missing, answer with what IS available rather than "
            "refusing. Cite chunk numbers used."
        )

    user_prompt = f"Query: {query}\n\nContext:\n{context}\n\nPlease provide an answer based on the above context."

    logger.info("Generating answer for query: %s", query)
    try:
        response = openai_client.chat.completions.create(
            model=cfg.openai.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=cfg.openai.llm_temperature,
        )

        answer_text = response.choices[0].message.content or ""
        prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0
        logger.info("Generated answer: %s", answer_text[:100])

        avg_score = sum(r.score for r in used_results) / len(used_results) if used_results else 0.0
        confidence = min(1.0, max(0.1, avg_score))

        return QAResult(
            answer=answer_text,
            sources=used_results,
            confidence=confidence,
            query=query,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    except Exception as e:
        logger.error("Error generating answer: %s", e)
        return QAResult(
            answer=f"Error generating answer: {e}",
            sources=used_results,
            confidence=0.0,
            query=query,
        )
