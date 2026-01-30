#!/usr/bin/env python3
"""
ClarityOCR - High-quality PDF to Markdown conversion

Features:
- GPU-accelerated OCR using marker-pdf (Surya models)
- LLM post-processing for error correction (LM Studio compatible)
- Real-time Web UI with progress tracking
- Optimized for NVIDIA RTX GPUs (16GB VRAM)
- Apple Silicon (MPS) support with fallback mode
"""

__version__ = "1.0.0"
__author__ = "ClarityOCR Team"

from typing import Optional
from pathlib import Path


def convert_pdf(pdf_path: str, output_path: Optional[str] = None, **kwargs) -> str:
    """
    Convert PDF to plain text (fallback mode).

    This mode uses pypdfium2 for fast, reliable text extraction
    without the heavy marker-pdf dependency.

    Args:
        pdf_path: Path to PDF file
        output_path: Optional output path (defaults to same name with .txt extension)
        **kwargs: Additional options (currently ignored)

    Returns:
        str: Path to generated text file
    """
    from .simple_converter import extract_text

    pdf = Path(pdf_path)

    # Determine output path
    if output_path:
        out = Path(output_path)
    else:
        out = pdf.with_suffix(".txt")

    # Run fallback converter
    result = extract_text(pdf, out)

    return result


def start_server(host: str = "127.0.0.1", port: int = 8008) -> None:
    """Start ClarityOCR Web UI server (fallback mode)."""
    from .server import run_server

    run_server(host=host, port=port)


def start_server_simple(host: str = "127.0.0.1", port: int = 8009) -> None:
    """Start simple server for fallback mode."""
    from .simple_converter import simple_server

    simple_server(host=host, port=port)


# Convenience exports
__all__ = [
    "__version__",
    "__author__",
    "convert_pdf",
    "start_server",
    "start_server_simple",
]
