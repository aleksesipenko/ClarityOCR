"""
ClarityOCR - High-quality PDF to Markdown conversion

Features:
- GPU-accelerated OCR using marker-pdf (Surya models)
- LLM post-processing for error correction (LM Studio compatible)
- Real-time Web UI with progress tracking
- Optimized for NVIDIA RTX GPUs (16GB VRAM)
"""

__version__ = "1.0.0"
__author__ = "ClarityOCR Team"

from typing import Optional


def convert_pdf(pdf_path: str, output_path: Optional[str] = None, **kwargs) -> str:
    """Convert a single PDF to Markdown.

    This is a convenience wrapper around the converter module.
    For batch processing or more control, use the CLI or server.

    Args:
        pdf_path: Path to the PDF file
        output_path: Optional output path (defaults to same name with .md extension)
        **kwargs: Additional options passed to the converter

    Returns:
        str: Path to the generated Markdown file
    """
    from pathlib import Path
    from .converter import main as converter_main
    import sys

    # Build arguments for the converter
    pdf = Path(pdf_path)
    out = Path(output_path) if output_path else pdf.with_suffix(".md")

    # Run converter with these specific files
    old_argv = sys.argv
    sys.argv = ["clarityocr-convert", "--pdf", str(pdf), "--output-dir", str(out.parent)]
    try:
        converter_main()
    finally:
        sys.argv = old_argv

    return str(out)


def start_server(host: str = "127.0.0.1", port: int = 8008) -> None:
    """Start the ClarityOCR Web UI server.

    Args:
        host: Host to bind to (default: 127.0.0.1)
        port: Port to listen on (default: 8008)
    """
    from .server import run_server

    run_server(host=host, port=port)


# Convenience exports
__all__ = [
    "__version__",
    "__author__",
    "convert_pdf",
    "start_server",
]
