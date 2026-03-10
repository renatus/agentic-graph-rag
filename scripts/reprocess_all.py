#!/usr/bin/env python3
"""Reprocess all uploaded files to generate logs and update metadata.

This script:
1. Finds all files in data/ directory
2. Deletes old chunks for each source
3. Re-ingests with proper logging and metadata

Usage:
    python scripts/reprocess_all.py [--skip-enrichment] [--skip-graph] [--dry-run]
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "pymangle"))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("reprocess_all")


def find_all_files(data_dir: str) -> list[Path]:
    """Find all supported files in data directory."""
    extensions = {".pdf", ".txt", ".md", ".docx", ".html", ".htm"}
    files = []

    for path in Path(data_dir).rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            files.append(path)

    return sorted(files)


def reprocess_file(
    file_path: Path,
    *,
    store,
    driver,
    client,
    skip_enrichment: bool = False,
    skip_graph: bool = False,
    dry_run: bool = False,
) -> dict:
    """Reprocess a single file. Returns stats dict."""

    from rag_core.chunker import chunk_text
    from rag_core.config import get_settings
    from rag_core.embedder import embed_chunks
    from rag_core.enricher import enrich_chunks
    from rag_core.ingestion_logger import IngestionLogger
    from rag_core.loader import load_file

    # Get source name from filename
    source_name = file_path.stem

    stats = {
        "file": str(file_path),
        "source": source_name,
        "status": "pending",
        "chunks_created": 0,
        "chunks_deleted": 0,
        "log_path": None,
        "error": None,
    }

    if dry_run:
        stats["status"] = "dry_run"
        logger.info("[DRY RUN] Would process: %s", file_path)
        return stats

    try:
        # Initialize ingestion logger
        ingestion_log = IngestionLogger(
            source_identifier=source_name,
            source_path=str(file_path),
        )
        stats["log_path"] = str(ingestion_log.get_log_path())

        # Delete old chunks for this source
        ingestion_log.start_step("deleting_old")
        deleted = store.delete_by_source(source_name)
        ingestion_log.end_step(output=f"{deleted} old chunks deleted")
        stats["chunks_deleted"] = deleted

        # Load file
        ingestion_log.start_step("loading")
        text = load_file(str(file_path), use_gpu=False)
        ingestion_log.end_step(output=f"{len(text)} characters loaded")

        if not text.strip():
            stats["status"] = "empty"
            ingestion_log.log_summary()
            return stats

        # Chunk
        ingestion_log.start_step("chunking")
        cfg = get_settings()
        chunks = chunk_text(text, cfg.indexing.chunk_size, cfg.indexing.chunk_overlap)

        # Add metadata to each chunk
        doc_metadata = {
            "source": source_name,
            "uploaded_at": datetime.now().isoformat(),
            "file_path": str(file_path),
        }
        for chunk in chunks:
            chunk.metadata.update(doc_metadata)

        ingestion_log.end_step(output=f"{len(chunks)} chunks created")
        stats["chunks_created"] = len(chunks)

        # Embed
        ingestion_log.start_step("embedding")
        chunks = embed_chunks(chunks)
        ingestion_log.end_step(output=f"{len(chunks)} chunks embedded")

        # Enrich
        ingestion_log.start_step("enrichment")
        if not skip_enrichment:
            chunks = enrich_chunks(chunks, text)
            ingestion_log.end_step(output=f"{len(chunks)} chunks enriched")
        else:
            ingestion_log.end_step(output="skipped")

        # Store
        ingestion_log.start_step("storing")
        stored = store.add_chunks(chunks)
        ingestion_log.end_step(output=f"{stored} chunks stored")

        # Build graph
        if not skip_graph:
            from agentic_graph_rag.indexing.dual_node import (
                build_dual_graph,
                embed_phrase_nodes,
                init_phrase_index,
            )
            from agentic_graph_rag.indexing.skeleton import build_skeleton_index

            ingestion_log.start_step("graph_building")
            embeddings = [c.embedding for c in chunks if c.embedding]
            entities, relationships, skeletal, peripheral = build_skeleton_index(
                chunks, embeddings, openai_client=client,
            )

            if entities:
                phrase_nodes, passage_nodes, link_count = build_dual_graph(
                    entities, chunks, driver, relationships=relationships,
                )
                embed_phrase_nodes(phrase_nodes, driver, client)
                init_phrase_index(driver)
                ingestion_log.end_step(
                    output=f"{len(entities)} entities, {len(relationships)} relationships"
                )
            else:
                ingestion_log.end_step(output="no entities extracted")
        else:
            ingestion_log.start_step("graph_building")
            ingestion_log.end_step(output="skipped")

        # Finalize log
        ingestion_log.log_chunks(chunks)
        ingestion_log.log_summary()

        stats["status"] = "success"
        logger.info("Processed: %s (%d chunks)", file_path, len(chunks))

    except Exception as e:
        stats["status"] = "error"
        stats["error"] = str(e)
        logger.error("Failed to process %s: %s", file_path, e)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Reprocess all uploaded files")
    parser.add_argument("--skip-enrichment", action="store_true", help="Skip LLM enrichment")
    parser.add_argument("--skip-graph", action="store_true", help="Skip knowledge graph building")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without processing")
    parser.add_argument("--limit", type=int, help="Limit number of files to process")
    args = parser.parse_args()

    # Find data directory
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"

    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    # Find all files
    files = find_all_files(data_dir)
    logger.info("Found %d files in %s", len(files), data_dir)

    if args.limit:
        files = files[:args.limit]
        logger.info("Limited to %d files", len(files))

    if not files:
        logger.info("No files to process")
        return

    # Initialize connections
    from neo4j import GraphDatabase
    from rag_core.config import get_settings, make_openai_client
    from rag_core.vector_store import VectorStore

    cfg = get_settings()
    driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))
    store = VectorStore(driver=driver)
    client = make_openai_client(cfg)

    try:
        # Process each file
        results = []
        for i, file_path in enumerate(files, 1):
            logger.info("[%d/%d] Processing: %s", i, len(files), file_path)
            result = reprocess_file(
                file_path,
                store=store,
                driver=driver,
                client=client,
                skip_enrichment=args.skip_enrichment,
                skip_graph=args.skip_graph,
                dry_run=args.dry_run,
            )
            results.append(result)

        # Summary
        success = sum(1 for r in results if r["status"] == "success")
        errors = sum(1 for r in results if r["status"] == "error")
        empty = sum(1 for r in results if r["status"] == "empty")
        total_chunks = sum(r["chunks_created"] for r in results)

        logger.info("=" * 50)
        logger.info("SUMMARY:")
        logger.info("  Total files: %d", len(files))
        logger.info("  Success: %d", success)
        logger.info("  Errors: %d", errors)
        logger.info("  Empty: %d", empty)
        logger.info("  Total chunks created: %d", total_chunks)

        # Show errors
        if errors:
            logger.info("")
            logger.info("ERRORS:")
            for r in results:
                if r["status"] == "error":
                    logger.info("  - %s: %s", r["file"], r["error"])

    finally:
        driver.close()


if __name__ == "__main__":
    main()
