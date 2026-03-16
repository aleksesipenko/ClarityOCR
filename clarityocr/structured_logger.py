"""
Structured JSON logger for ClarityOCR Enterprise (Phase 1.2)

Provides structured logging with job_id, file_id, stage, duration, and other
contextual fields for centralized log aggregation and monitoring.
"""

import json
import logging
import time
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime


class StructuredLogger:
    """
    Structured JSON logger that emits one JSON object per line.

    Fields:
    - timestamp: ISO 8601 timestamp
    - level: log level (INFO, WARNING, ERROR, etc.)
    - message: human-readable message
    - job_id: optional job identifier
    - file_id: optional file identifier
    - stage: optional processing stage (upload/ocr/merge/polish/convert/done)
    - duration_ms: optional operation duration in milliseconds
    - extra: optional dict of additional context
    """

    def __init__(self, name: str = "clarityocr", log_file: Optional[str] = None):
        """
        Initialize structured logger.

        Args:
            name: logger name
            log_file: optional path to log file (defaults to ~/.clarityocr/logs/structured.log)
        """
        self.name = name
        self.logger = logging.getLogger(f"structured.{name}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        # Setup file handler
        if log_file is None:
            log_dir = Path.home() / ".clarityocr" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = str(log_dir / "structured.log")

        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        # No formatter - we'll emit raw JSON
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(handler)

    def _emit(
        self,
        level: str,
        message: str,
        job_id: Optional[str] = None,
        file_id: Optional[str] = None,
        stage: Optional[str] = None,
        duration_ms: Optional[int] = None,
        **extra: Any
    ):
        """Emit a structured log entry as JSON."""
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "logger": self.name,
            "message": message,
        }

        if job_id:
            record["job_id"] = job_id
        if file_id:
            record["file_id"] = file_id
        if stage:
            record["stage"] = stage
        if duration_ms is not None:
            record["duration_ms"] = duration_ms
        if extra:
            record["extra"] = extra

        # Emit as single-line JSON
        json_line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))

        # Route to appropriate log level
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        self.logger.log(level_map.get(level, logging.INFO), json_line)

    def debug(
        self,
        message: str,
        job_id: Optional[str] = None,
        file_id: Optional[str] = None,
        stage: Optional[str] = None,
        duration_ms: Optional[int] = None,
        **extra: Any
    ):
        """Log debug message."""
        self._emit("DEBUG", message, job_id, file_id, stage, duration_ms, **extra)

    def info(
        self,
        message: str,
        job_id: Optional[str] = None,
        file_id: Optional[str] = None,
        stage: Optional[str] = None,
        duration_ms: Optional[int] = None,
        **extra: Any
    ):
        """Log info message."""
        self._emit("INFO", message, job_id, file_id, stage, duration_ms, **extra)

    def warning(
        self,
        message: str,
        job_id: Optional[str] = None,
        file_id: Optional[str] = None,
        stage: Optional[str] = None,
        duration_ms: Optional[int] = None,
        **extra: Any
    ):
        """Log warning message."""
        self._emit("WARNING", message, job_id, file_id, stage, duration_ms, **extra)

    def error(
        self,
        message: str,
        job_id: Optional[str] = None,
        file_id: Optional[str] = None,
        stage: Optional[str] = None,
        duration_ms: Optional[int] = None,
        **extra: Any
    ):
        """Log error message."""
        self._emit("ERROR", message, job_id, file_id, stage, duration_ms, **extra)


# Global singleton for convenience
_global_logger: Optional[StructuredLogger] = None


def get_logger(name: str = "clarityocr") -> StructuredLogger:
    """Get or create global structured logger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = StructuredLogger(name)
    return _global_logger
