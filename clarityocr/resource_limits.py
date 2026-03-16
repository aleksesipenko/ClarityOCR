"""
Resource limits for ClarityOCR pipeline.
Phase 4.4: Configurable limits to prevent resource exhaustion.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False


@dataclass
class ResourceLimits:
    """Resource limit configuration."""
    max_pages_per_file: int = int(os.getenv("LIMIT_MAX_PAGES", "500"))
    max_file_size_mb: int = int(os.getenv("LIMIT_MAX_FILE_SIZE_MB", "512"))
    max_processing_time_sec: int = int(os.getenv("LIMIT_MAX_PROCESSING_TIME", "3600"))


class ResourceLimitError(Exception):
    """Raised when a resource limit is exceeded."""
    pass


def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes."""
    if not os.path.exists(file_path):
        return 0.0
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)


def count_pdf_pages(file_path: str) -> Optional[int]:
    """
    Count pages in a PDF file.
    Returns None if unable to determine.
    """
    if not PYPDF2_AVAILABLE:
        return None

    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return len(reader.pages)
    except Exception:
        # If PDF is corrupted or unreadable, return None
        # The actual processing will fail later with appropriate error
        return None


def check_file_limits(file_path: str, limits: ResourceLimits) -> None:
    """
    Check if file meets resource limits.
    Raises ResourceLimitError if any limit is exceeded.
    """
    # Check file size
    file_size_mb = get_file_size_mb(file_path)
    if file_size_mb > limits.max_file_size_mb:
        raise ResourceLimitError(
            f"File size {file_size_mb:.1f}MB exceeds limit of {limits.max_file_size_mb}MB"
        )

    # Check page count for PDFs
    if Path(file_path).suffix.lower() == '.pdf':
        page_count = count_pdf_pages(file_path)
        if page_count is not None and page_count > limits.max_pages_per_file:
            raise ResourceLimitError(
                f"PDF has {page_count} pages, exceeds limit of {limits.max_pages_per_file} pages"
            )
