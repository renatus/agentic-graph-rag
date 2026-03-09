"""Neo4j Vector Index store for RAG.

From RAG 2.0 — CRUD operations on Neo4j vector index with cosine similarity search.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from rag_core.config import get_settings
from rag_core.models import Chunk, SearchResult

if TYPE_CHECKING:
    from neo4j import Driver

logger = logging.getLogger(__name__)

INDEX_NAME = "rag_chunks_index"
NODE_LABEL = "RagChunk"
EMBEDDING_PROPERTY = "embedding"


class VectorStore:
    """Neo4j-backed vector store with cosine similarity search."""

    def __init__(self, driver: Driver | None = None) -> None:
        cfg = get_settings()
        if driver is None:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                cfg.neo4j.uri,
                auth=(cfg.neo4j.user, cfg.neo4j.password),
            )
        else:
            self._driver = driver

    def close(self) -> None:
        self._driver.close()

    def init_index(self) -> None:
        """Create vector index in Neo4j if it doesn't exist."""
        cfg = get_settings()
        with self._driver.session() as session:
            session.run(
                f"""
                CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS
                FOR (n:{NODE_LABEL})
                ON (n.{EMBEDDING_PROPERTY})
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: $dimensions,
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
                """,
                dimensions=cfg.embedding.dimensions,
            )
        logger.info("Vector index '%s' initialized", INDEX_NAME)

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Store chunks as Neo4j nodes with embeddings. Returns count added."""
        if not chunks:
            return 0

        with self._driver.session() as session:
            for chunk in chunks:
                chunk_id = chunk.id or hashlib.md5(chunk.content.encode()).hexdigest()
                session.run(
                    f"""
                    MERGE (c:{NODE_LABEL} {{id: $id}})
                    SET c.content = $content,
                        c.context = $context,
                        c.enriched_content = $enriched_content,
                        c.{EMBEDDING_PROPERTY} = $embedding,
                        c.metadata = $metadata_json
                    """,
                    id=chunk_id,
                    content=chunk.content,
                    context=chunk.context,
                    enriched_content=chunk.enriched_content,
                    embedding=chunk.embedding,
                    metadata_json=str(chunk.metadata),
                )

        logger.info("Added %d chunks to vector store", len(chunks))
        return len(chunks)

    def search(
        self, query_embedding: list[float], top_k: int | None = None,
    ) -> list[SearchResult]:
        """Search vector index by cosine similarity."""
        if top_k is None:
            top_k = get_settings().retrieval.top_k_vector

        with self._driver.session() as session:
            result = session.run(
                f"""
                CALL db.index.vector.queryNodes(
                    '{INDEX_NAME}', $top_k, $embedding
                )
                YIELD node, score
                RETURN node.id AS id,
                       node.content AS content,
                       node.context AS context,
                       score
                ORDER BY score DESC
                """,
                top_k=top_k,
                embedding=query_embedding,
            )

            results = []
            for i, record in enumerate(result):
                chunk = Chunk(
                    id=record["id"] or "",
                    content=record["content"] or "",
                    context=record["context"] or "",
                )
                results.append(
                    SearchResult(chunk=chunk, score=record["score"], rank=i + 1)
                )

        return results

    def delete_all(self) -> int:
        """Delete all RagChunk nodes. Returns count deleted."""
        with self._driver.session() as session:
            result = session.run(
                f"""
                MATCH (c:{NODE_LABEL})
                WITH c, count(c) AS total
                DETACH DELETE c
                RETURN total
                """
            )
            record = result.single()
            count = record["total"] if record else 0

        logger.info("Deleted %d chunks from vector store", count)
        return count

    def count(self) -> int:
        """Return total number of chunks."""
        with self._driver.session() as session:
            result = session.run(
                f"MATCH (c:{NODE_LABEL}) RETURN count(c) AS total"
            )
            record = result.single()
            return record["total"] if record else 0

    def list_documents(self) -> list[dict]:
        """List unique documents with metadata.

        Returns list of dicts with:
        - source: document source/filename
        - chunk_count: number of chunks
        - total_chars: total characters
        - uploaded_at: upload timestamp (if available)
        - file_path: original file path (if available)
        - sections: list of section titles (for unknown sources)

        Sorted by uploaded_at (newest first) by default.
        """
        import ast

        with self._driver.session() as session:
            result = session.run(
                f"""
                MATCH (c:{NODE_LABEL})
                RETURN c.metadata AS meta, size(c.content) AS content_size,
                       substring(c.content, 0, 100) AS content_preview
                """
            )

            # Aggregate by source in Python
            docs_by_source: dict[str, dict] = {}
            for record in result:
                meta_str = record["meta"] or "{}"
                try:
                    meta = ast.literal_eval(meta_str) if meta_str else {}
                except (ValueError, SyntaxError):
                    meta = {}

                source = meta.get("source", "unknown")
                uploaded_at = meta.get("uploaded_at", "")
                file_path = meta.get("file_path", "")
                section_title = meta.get("section_title", "")

                if source not in docs_by_source:
                    docs_by_source[source] = {
                        "source": source,
                        "chunk_count": 0,
                        "total_chars": 0,
                        "uploaded_at": uploaded_at,
                        "file_path": file_path,
                        "sections": {},
                    }
                docs_by_source[source]["chunk_count"] += 1
                docs_by_source[source]["total_chars"] += record["content_size"] or 0

                # Track sections for unknown sources
                if source == "unknown" and section_title:
                    if section_title not in docs_by_source[source]["sections"]:
                        docs_by_source[source]["sections"][section_title] = {
                            "chunk_count": 0,
                            "total_chars": 0,
                        }
                    docs_by_source[source]["sections"][section_title]["chunk_count"] += 1
                    docs_by_source[source]["sections"][section_title]["total_chars"] += record["content_size"] or 0

            # Convert sections dict to sorted list
            for doc in docs_by_source.values():
                if doc["sections"]:
                    doc["sections"] = sorted(
                        doc["sections"].items(),
                        key=lambda x: x[1]["chunk_count"],
                        reverse=True
                    )
                else:
                    doc["sections"] = []

            # Sort by uploaded_at (newest first), fallback to source name
            def sort_key(doc):
                uploaded = doc.get("uploaded_at", "")
                return (uploaded is None or uploaded == "", uploaded, doc.get("source", ""))

            return sorted(docs_by_source.values(), key=sort_key, reverse=True)

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a specific source. Returns count deleted."""
        with self._driver.session() as session:
            result = session.run(
                f"""
                MATCH (c:{NODE_LABEL})
                WHERE c.metadata CONTAINS $source
                WITH c, count(c) AS total
                DETACH DELETE c
                RETURN total
                """,
                source=f"'source': '{source}'",
            )
            record = result.single()
            count = record["total"] if record else 0

        logger.info("Deleted %d chunks from source '%s'", count, source)
        return count

    def get_chunks_by_source(self, source: str) -> list[dict]:
        """Get all chunks for a specific source.

        Returns list of dicts with: id, content, context, metadata
        """
        import ast

        with self._driver.session() as session:
            if source == "unknown":
                # Get chunks without source metadata
                result = session.run(
                    f"""
                    MATCH (c:{NODE_LABEL})
                    WHERE NOT c.metadata CONTAINS 'source'
                       OR c.metadata CONTAINS "'source': 'unknown'"
                    RETURN c.id AS id, c.content AS content,
                           c.context AS context, c.metadata AS metadata
                    """
                )
            else:
                result = session.run(
                    f"""
                    MATCH (c:{NODE_LABEL})
                    WHERE c.metadata CONTAINS $source
                    RETURN c.id AS id, c.content AS content,
                           c.context AS context, c.metadata AS metadata
                    """,
                    source=f"'source': '{source}'",
                )

            chunks = []
            for record in result:
                meta_str = record["metadata"] or "{}"
                try:
                    metadata = ast.literal_eval(meta_str) if meta_str else {}
                except (ValueError, SyntaxError):
                    metadata = {}

                chunks.append({
                    "id": record["id"],
                    "content": record["content"],
                    "context": record["context"],
                    "metadata": metadata,
                })

            return chunks

    def update_chunk_embeddings(self, chunks: list[dict]) -> int:
        """Update embeddings for existing chunks. Returns count updated."""
        with self._driver.session() as session:
            count = 0
            for chunk in chunks:
                if chunk.get("embedding"):
                    session.run(
                        f"""
                        MATCH (c:{NODE_LABEL} {{id: $id}})
                        SET c.embedding = $embedding
                        """,
                        id=chunk["id"],
                        embedding=chunk["embedding"],
                    )
                    count += 1

        logger.info("Updated embeddings for %d chunks", count)
        return count

    def update_chunk_metadata(self, chunk_id: str, metadata: dict) -> bool:
        """Update metadata for a chunk. Returns True if successful."""
        with self._driver.session() as session:
            result = session.run(
                f"""
                MATCH (c:{NODE_LABEL} {{id: $id}})
                SET c.metadata = $metadata_str
                RETURN c.id AS id
                """,
                id=chunk_id,
                metadata_str=str(metadata),
            )
            record = result.single()
            return record is not None

    def get_chunks_by_section(self, section_title: str) -> list[dict]:
        """Get all chunks for a specific section title (for legacy documents).

        Returns list of dicts with: id, content, context, metadata
        """
        import ast

        with self._driver.session() as session:
            result = session.run(
                f"""
                MATCH (c:{NODE_LABEL})
                WHERE c.metadata CONTAINS $section_pattern
                RETURN c.id AS id, c.content AS content,
                       c.context AS context, c.metadata AS metadata
                """,
                section_pattern=f"'section_title': '{section_title}'",
            )

            chunks = []
            for record in result:
                meta_str = record["metadata"] or "{}"
                try:
                    metadata = ast.literal_eval(meta_str) if meta_str else {}
                except (ValueError, SyntaxError):
                    metadata = {}

                # Verify the section title matches exactly
                if metadata.get("section_title") == section_title:
                    chunks.append({
                        "id": record["id"],
                        "content": record["content"],
                        "context": record["context"],
                        "metadata": metadata,
                    })

            return chunks
