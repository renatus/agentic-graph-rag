"""Ingestion logging for document processing.

Creates detailed log files for each uploaded or reprocessed document,
tracking all processing steps, timings, and chunk details.

Log files are stored next to the processed files with .log extension.
Example: document.pdf -> document.pdf.log
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_core.models import Chunk

logger = logging.getLogger(__name__)


class IngestionLogger:
    """Logger for tracking document ingestion processing.

    Creates a detailed log file for each document being processed,
    recording all steps, timings, and chunk information.

    Log file format: {source_file}.log
    Location: Same directory as the source file

    Example log file contents:
        === DOCUMENT PROCESSING LOG ===
        File: report.pdf
        Source: /path/to/report.pdf
        Started: 2024-03-09T12:00:00

        --- STEP: loading ---
        Started: 2024-03-09T12:00:00.000
        Ended: 2024-03-09T12:00:01.234
        Duration: 1.234s
        Output: 50000 characters loaded

        --- STEP: chunking ---
        ...

        === CHUNKS ===
        [Chunk 1]
        Content: ...
        Metadata: {...}

        [Chunk 2]
        ...
    """

    def __init__(self, source_identifier: str, source_path: str | Path | None = None):
        """Initialize the ingestion logger.

        Args:
            source_identifier: Filename or identifier for the document being processed.
            source_path: Optional path to the source file. If provided, log is stored
                        next to it. Otherwise, log is stored in current directory.
        """
        self.source_identifier = source_identifier
        self.start_time = datetime.now()
        self.current_step: str | None = None
        self.step_start_time: float | None = None
        self.steps: list[dict] = []
        self.chunks: list[Chunk] = []

        # Determine log file path
        if source_path:
            source = Path(source_path)
            self.log_path = source.parent / f"{source.name}.log"
        else:
            # Store in current directory with sanitized name
            safe_name = self._sanitize_filename(source_identifier)
            self.log_path = Path(f"{safe_name}.log")

        # Initialize log file
        self._write_header()

        logger.info(
            "Ingestion logger initialized for: %s (log: %s)",
            source_identifier, self.log_path
        )

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe filesystem use."""
        safe = "".join(c if c.isalnum() or c in ".-_" else "_" for c in filename)
        safe = safe.strip("_.")
        return safe or "unknown_file"

    def _write_header(self) -> None:
        """Write the log file header."""
        header = f"""{'=' * 60}
DOCUMENT PROCESSING LOG
{'=' * 60}
File: {self.source_identifier}
Started: {self.start_time.isoformat()}
Log file: {self.log_path}

"""
        self.log_path.write_text(header, encoding="utf-8")

    def start_step(self, step_name: str) -> None:
        """Start tracking a processing step.

        Args:
            step_name: Name of the step (e.g., 'loading', 'chunking', 'embedding').
        """
        self.current_step = step_name
        self.step_start_time = time.time()

        entry = f"\n--- STEP: {step_name} ---\n"
        entry += f"Started: {datetime.now().isoformat()}\n"

        self._append(entry)
        logger.debug("Started step: %s", step_name)

    def end_step(
        self,
        output: str | dict | None = None,
        error: str | None = None,
    ) -> float:
        """End the current processing step.

        Args:
            output: Output/result of the step (string or dict).
            error: Error message if step failed.

        Returns:
            Duration of the step in seconds.
        """
        if not self.current_step or self.step_start_time is None:
            logger.warning("end_step called without active step")
            return 0.0

        duration = time.time() - self.step_start_time

        step_info = {
            "step": self.current_step,
            "start_time": datetime.fromtimestamp(self.step_start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": round(duration, 3),
            "output": output,
            "error": error,
        }
        self.steps.append(step_info)

        entry = f"Ended: {datetime.now().isoformat()}\n"
        entry += f"Duration: {duration:.3f}s\n"

        if error:
            entry += f"Status: FAILED\n"
            entry += f"Error: {error}\n"
        else:
            entry += f"Status: SUCCESS\n"

        if output:
            if isinstance(output, dict):
                entry += f"Output:\n{json.dumps(output, indent=2, ensure_ascii=False)}\n"
            else:
                entry += f"Output: {output}\n"

        self._append(entry)

        logger.debug(
            "Completed step: %s (%.3fs)",
            self.current_step, duration
        )

        self.current_step = None
        self.step_start_time = None

        return duration

    def log_chunks(self, chunks: list[Chunk]) -> None:
        """Log all chunks with their content and metadata.

        Args:
            chunks: List of Chunk objects to log.
        """
        self.chunks = chunks

        entry = f"\n{'=' * 60}\n"
        entry += f"CHUNKS ({len(chunks)} total)\n"
        entry += f"{'=' * 60}\n\n"

        for i, chunk in enumerate(chunks, 1):
            entry += f"[Chunk {i}]\n"
            entry += f"ID: {chunk.id}\n"

            # Content (truncated if very long)
            content = chunk.content
            if len(content) > 500:
                content = content[:500] + "...[truncated]"
            entry += f"Content:\n{content}\n"

            # Context (if enriched)
            if chunk.context:
                ctx = chunk.context
                if len(ctx) > 200:
                    ctx = ctx[:200] + "..."
                entry += f"Context: {ctx}\n"

            # Metadata
            if chunk.metadata:
                entry += f"Metadata: {json.dumps(chunk.metadata, indent=2, ensure_ascii=False)}\n"

            # Embedding info (just dimensions, not the actual vector)
            if chunk.embedding is not None:
                entry += f"Embedding: {len(chunk.embedding)} dimensions\n"

            entry += "\n"

        self._append(entry)
        logger.info("Logged %d chunks", len(chunks))

    def log_summary(self) -> dict:
        """Write and return a summary of the ingestion process.

        Returns:
            Summary dict with timing and chunk statistics.
        """
        total_duration = (datetime.now() - self.start_time).total_seconds()

        step_summary = {}
        for step in self.steps:
            step_summary[step["step"]] = {
                "duration": step["duration_seconds"],
                "status": "failed" if step.get("error") else "success",
            }

        summary = {
            "filename": self.source_identifier,
            "started": self.start_time.isoformat(),
            "completed": datetime.now().isoformat(),
            "total_duration_seconds": round(total_duration, 3),
            "total_chunks": len(self.chunks),
            "steps": step_summary,
        }

        entry = f"\n{'=' * 60}\n"
        entry += f"SUMMARY\n"
        entry += f"{'=' * 60}\n"
        entry += f"Completed: {datetime.now().isoformat()}\n"
        entry += f"Total Duration: {total_duration:.3f}s\n"
        entry += f"Total Chunks: {len(self.chunks)}\n"
        entry += f"\nStep Timings:\n"
        for step_name, step_info in step_summary.items():
            entry += f"  - {step_name}: {step_info['duration']:.3f}s ({step_info['status']})\n"

        self._append(entry)
        logger.info(
            "Ingestion complete for %s: %.3fs, %d chunks",
            self.source_identifier, total_duration, len(self.chunks)
        )

        return summary

    def _append(self, content: str) -> None:
        """Append content to the log file."""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(content)

    def get_log_path(self) -> Path:
        """Get the path to the log file."""
        return self.log_path


def find_log_for_file(file_path: str | Path) -> Path | None:
    """Find the log file associated with a document.

    Args:
        file_path: Path to the document file.

    Returns:
        Path to the log file if it exists, None otherwise.
    """
    log_path = Path(str(file_path) + ".log")
    return log_path if log_path.exists() else None


def list_all_ingestion_logs(data_dir: str | Path = "data") -> list[Path]:
    """List all ingestion log files in the data directory structure.

    Args:
        data_dir: Root data directory to search for logs.

    Returns:
        List of log file paths, sorted by modification time (newest first).
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return []

    # Search recursively for all .log files
    logs = list(data_path.rglob("*.log"))
    return sorted(logs, key=lambda p: p.stat().st_mtime, reverse=True)
