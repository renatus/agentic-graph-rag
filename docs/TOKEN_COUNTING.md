# Token Counting Strategy

## Overview

This document describes the token counting strategy used in the RAG system for context window management and preventing token limit errors.

## The Problem

Different LLM providers use different tokenizers:

| Model Family | Tokenizer | Vocabulary Size |
|--------------|-----------|-----------------|
| GPT-4 / GPT-4o | tiktoken `cl100k_base` | ~100,277 |
| DeepSeek-V2/V3/R1 | Custom Byte-level BPE | ~128,000 |
| LLaMA | SentencePiece BPE | 32,000-128,000 |

Using the wrong tokenizer for token counting leads to:
- **Underestimation** — Context may exceed limits, causing API errors
- **Overestimation** — Wasted context capacity, less relevant content included

## Our Solution: deepseek-tokenizer

We use the [`deepseek-tokenizer`](https://pypi.org/project/deepseek-tokenizer/) package which provides the exact same tokenization as DeepSeek models.

### Why deepseek-tokenizer

1. **Exact accuracy** — Same tokenizer as DeepSeek-V3/R1 models
2. **Lightweight** — Only depends on `tokenizers`, not the heavy `transformers` library
3. **Fast** — No need to download large model files
4. **Official vocabulary** — Uses the 128K vocabulary from DeepSeek

### Implementation

```python
from deepseek_tokenizer import ds_token

def count_tokens(text: str) -> int:
    """Count tokens using DeepSeek's official tokenizer."""
    if not text:
        return 0
    return len(ds_token.encode(text))
```

### Usage Example

```python
from rag_core.generator import count_tokens

# English text
count_tokens("Hello world")  # → 2

# Russian text
count_tokens("Привет мир")   # → 3

# Mixed content
count_tokens("Hello, мир!")  # → 5

# Long text
count_tokens("A" * 1000)     # → 125
```

## DeepSeek Tokenizer Details

DeepSeek uses a custom **Byte-level BPE tokenizer** with:

| Characteristic | Value |
|----------------|-------|
| Vocabulary size | ~128,000 tokens |
| Algorithm | Byte-level BPE |
| Number handling | Individual digits (like LLaMA) |
| Special tokens | DeepSeek-specific (`<｜end▁of▁sentence｜>`, etc.) |

### How It Differs from GPT-4

| Aspect | GPT-4 (cl100k_base) | DeepSeek |
|--------|---------------------|----------|
| Vocabulary size | ~100K | ~128K |
| Number handling | Multi-digit chunks | Individual digits |
| Special tokens | OpenAI-specific | DeepSeek-specific |
| Russian efficiency | Moderate | Slightly better |

**Important:** Using tiktoken's `cl100k_base` for DeepSeek would underestimate token counts by 10-20%, potentially causing context overflow errors.

## Installation

The package is included as a dependency in `rag-core`:

```toml
# packages/rag-core/pyproject.toml
dependencies = [
    # ...
    "deepseek-tokenizer>=0.1.0",
]
```

Manual installation:

```bash
pip install deepseek-tokenizer
```

## Accuracy Comparison

| Approach | Accuracy | Speed | Dependencies |
|----------|----------|-------|--------------|
| deepseek-tokenizer | **Exact** | Fast | tokenizers |
| tiktoken (cl100k_base) | ~85% accurate | Fast | tiktoken |
| tiktoken + correction factor | ~95% accurate | Fast | tiktoken |
| HuggingFace AutoTokenizer | Exact | Slow | transformers, model files |

## Related Files

- Implementation: `packages/rag-core/rag_core/generator.py`
- Configuration: `packages/rag-core/pyproject.toml`
- Context management: `docs/EMBEDDING_BASED_SUMMARIZATION.md`

## References

- [deepseek-tokenizer on PyPI](https://pypi.org/project/deepseek-tokenizer/)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Byte-level BPE Paper](https://arxiv.org/abs/1909.03304)
