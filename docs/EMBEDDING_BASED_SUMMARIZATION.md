# Embedding-Based Document Summarization

## Overview

This document describes the hybrid approach used for generating document summaries during the chunk enrichment phase. The approach uses **embedding-based centroid selection** to identify the most representative chunks of a document, ensuring summaries capture the document's core content rather than random excerpts.

## Problem Statement

During document ingestion, the system generates contextual information for each chunk to improve retrieval quality. This requires a document summary. The naive approach of using the entire document text as "summary" causes:

1. **Token limit errors** — Large documents (100K+ chars) exceed LLM context limits
2. **Random sampling bias** — Simple truncation to first N chars may miss important content
3. **Quality degradation** — Non-representative excerpts produce poor chunk context

## Solution: Centroid-Based Representative Selection

### Algorithm

```
┌─────────────────────────────────────────────────────────────┐
│  1. Embed all chunks (already done during ingestion)        │
│       ↓                                                     │
│  2. Compute centroid = mean(all embeddings)                 │
│       ↓                                                     │
│  3. Calculate cosine similarity: chunk ↔ centroid           │
│       ↓                                                     │
│  4. Select top-K chunks closest to centroid                 │
│       ↓                                                     │
│  5. LLM generates summary from selected chunks              │
└─────────────────────────────────────────────────────────────┘
```

### Why Centroid Selection Works

The centroid (mean embedding) represents the "average" semantic content of the document. Chunks closest to this centroid:

- **Contain recurring themes** — Topics mentioned throughout the document
- **Avoid outliers** — Unique/introductory content is deprioritized
- **Provide coverage** — Representative of the document as a whole

### Mathematical Foundation

For a document with N chunks having embeddings e₁, e₂, ..., eₙ:

```
centroid = (1/N) × Σ eᵢ

similarity(chunkᵢ, centroid) = (eᵢ · centroid) / (||eᵢ|| × ||centroid||)
```

Chunks with highest cosine similarity are selected.

## Implementation

### Pipeline Order

**Before (broken):**
```
Create chunks → Enrich (no embeddings) → Embed → Store
```

**After (fixed):**
```
Create chunks → Embed → Enrich (uses embeddings) → Store
```

### Key Functions

#### `find_representative_chunks(chunks, top_k=5)`
- Input: Chunks with embeddings
- Output: Top-K chunks closest to centroid
- Uses numpy for efficient vector operations

#### `_generate_summary_from_chunks(chunks, client, model)`
- Calls `find_representative_chunks()` if embeddings available
- Falls back to first-K chunks if no embeddings
- Generates 2-3 sentence summary via LLM

### Configuration

| Constant | Value | Purpose |
|----------|-------|---------|
| `MIN_CHUNKS_FOR_CENTROID` | 5 | Minimum chunks to use centroid selection |
| `REPRESENTATIVE_CHUNK_COUNT` | 5 | Number of chunks to include in summary |

## Pros and Cons

### Advantages

| Pro | Description |
|-----|-------------|
| **No extra API cost** | Uses embeddings already computed for vector storage |
| **Semantic selection** | Chooses chunks based on meaning, not position |
| **Document coverage** | Captures themes from entire document |
| **Deterministic** | Same input produces same output |
| **Language agnostic** | Works for any language the embedding model supports |

### Limitations

| Con | Description | Mitigation |
|-----|-------------|------------|
| **"Representative" ≠ "Important"** | Centroid chunks may miss critical unique info | Use hybrid with structure-aware extraction |
| **Requires embeddings first** | Pipeline order must embed before enrich | Documented in pipeline |
| **Small document edge case** | <5 chunks falls back to simple selection | Acceptable for short documents |
| **Homogeneous content bias** | Very uniform documents may select similar chunks | Diversity selection could be added |

## Comparison with Alternatives

| Approach | Cost | Quality | Complexity |
|----------|------|---------|------------|
| **Centroid Selection (this)** | $0 extra | Medium-High | Low |
| Random sampling (beginning/middle/end) | $0 extra | Low | Very Low |
| Map-Reduce hierarchical | High (100+ calls) | High | High |
| KeyBERT topic extraction | $0 extra | Medium | Medium |
| Structure-aware extraction | $0 extra | High (for structured docs) | Medium |

## Usage Example

```python
from rag_core.enricher import enrich_chunks, find_representative_chunks
from rag_core.embedder import embed_chunks
from rag_core.models import Chunk

# Create chunks from document
chunks = [Chunk(content=...), ...]

# Embed first (required for centroid selection)
chunks = embed_chunks(chunks)

# Find representative chunks
representative = find_representative_chunks(chunks, top_k=5)

# Enrich with automatic summary generation
chunks = enrich_chunks(chunks)  # Uses embeddings for smart summary
```

## Future Improvements

1. **Diversity-aware selection** — Ensure selected chunks cover different topics
2. **Structure detection** — Weight headings/sections higher
3. **Adaptive K** — Select more chunks for longer documents
4. **Caching** — Store summaries to avoid regeneration

## References

- Implementation: `packages/rag-core/rag_core/enricher.py`
- Pipeline: `ui/streamlit_app.py` (ingestion flow)
- Related: `docs/PORTFOLIO_ANALYSIS.md` (overall architecture)
