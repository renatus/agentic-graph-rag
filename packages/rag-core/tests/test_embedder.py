"""Tests for rag_core.embedder."""

from unittest.mock import MagicMock, patch

import pytest
from rag_core.embedder import embed_chunks, embed_texts
from rag_core.models import Chunk


class TestEmbedTexts:
    def test_empty_list(self):
        assert embed_texts([]) == []

    @patch("rag_core.embedder.make_openai_client")
    @patch("rag_core.embedder.get_settings")
    def test_openai_provider(self, mock_settings, mock_make_client):
        cfg = MagicMock()
        cfg.embedding.provider = "openai"
        cfg.embedding.model = "text-embedding-3-small"
        mock_settings.return_value = cfg

        # Mock embedding response
        emb1 = MagicMock()
        emb1.embedding = [0.1, 0.2, 0.3]
        emb2 = MagicMock()
        emb2.embedding = [0.4, 0.5, 0.6]

        mock_response = MagicMock()
        mock_response.data = [emb1, emb2]

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_make_client.return_value = mock_client

        result = embed_texts(["hello", "world"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

    @patch("rag_core.embedder.get_local_embedding_model")
    @patch("rag_core.embedder.get_settings")
    def test_local_provider_with_e5_model(self, mock_settings, mock_get_model):
        cfg = MagicMock()
        cfg.embedding.provider = "local"
        cfg.embedding.model = "intfloat/multilingual-e5-large"
        mock_settings.return_value = cfg

        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_get_model.return_value = mock_model

        result = embed_texts(["hello", "world"], is_query=True)

        # Check that E5 prefix was added
        call_args = mock_model.encode.call_args
        texts = call_args[0][0]
        assert texts[0] == "query: hello"
        assert texts[1] == "query: world"


class TestEmbedChunks:
    def test_empty_list(self):
        assert embed_chunks([]) == []

    @patch("rag_core.embedder.embed_texts")
    @patch("rag_core.embedder.get_settings")
    def test_embeds_chunks(self, mock_settings, mock_embed_texts):
        cfg = MagicMock()
        cfg.embedding.model = "text-embedding-3-small"
        cfg.embedding.provider = "openai"
        mock_settings.return_value = cfg

        mock_embed_texts.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        chunks = [Chunk(content="hello"), Chunk(content="world")]
        result = embed_chunks(chunks)

        assert len(result) == 2
        assert result[0].embedding == [0.1, 0.2, 0.3]
        assert result[1].embedding == [0.4, 0.5, 0.6]

    @patch("rag_core.embedder.embed_texts")
    @patch("rag_core.embedder.get_settings")
    def test_uses_enriched_content(self, mock_settings, mock_embed_texts):
        cfg = MagicMock()
        cfg.embedding.model = "text-embedding-3-small"
        cfg.embedding.provider = "openai"
        mock_settings.return_value = cfg

        mock_embed_texts.return_value = [[0.1]]

        chunk = Chunk(content="text", context="prefix")
        embed_chunks([chunk])

        call_args = mock_embed_texts.call_args
        texts = call_args[0][0]
        assert texts == ["prefix\n\ntext"]  # enriched_content

    @patch("rag_core.embedder.embed_texts")
    @patch("rag_core.embedder.get_settings")
    def test_raises_on_error(self, mock_settings, mock_embed_texts):
        cfg = MagicMock()
        cfg.embedding.model = "text-embedding-3-small"
        cfg.embedding.provider = "openai"
        mock_settings.return_value = cfg

        mock_embed_texts.side_effect = Exception("Embedding failed")

        with pytest.raises(Exception, match="Embedding failed"):
            embed_chunks([Chunk(content="test")])
