#!/usr/bin/env python3
"""
Fallback PDF to Text Converter (no marker-pdf dependency)

This is a simplified PDF extractor that works without marker-pdf.
It extracts text directly from PDF using PyMuPDF or pypdfium2.

Fallback priority:
1. PyMuPDF (fast, pure Python)
2. pypdfium2 (already as dependency)
"""

import os
from pathlib import Path
from typing import Optional, List
import warnings

warnings.filterwarnings("ignore")


def extract_with_pymupdf(pdf_path: Path) -> Optional[str]:
    """Extract text using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text = []
        for page in doc:
            text.append(page.get_text())
        return "\n\n".join(text)
    except ImportError:
        return None
    except Exception as e:
        print(f"[fallback] PyMuPDF failed: {e}")
        return None


def extract_with_pypdfium2(pdf_path: Path) -> Optional[str]:
    """Extract text using pypdfium2."""
    try:
        import pypdfium2
        pdf = pypdfium2.Pdf(pdf_path)
        n_pages = len(pdf)

        text_parts = []
        for page_index in range(n_pages):
            page = pdf[page_index]
            text_page = page.get_textpage()
            text_parts.append(text_page.get_text().strip())
            # Free memory
            text_page = None

        return "\n\n".join(text_parts)
    except Exception as e:
        print(f"[fallback] pypdfium2 failed: {e}")
        return None


def fallback_convert(pdf_path: str, output_path: Optional[str] = None) -> str:
    """Convert PDF to plain text (no markdown formatting)."""
    pdf = Path(pdf_path)

    # Try PyMuPDF first (faster, pure Python)
    print("[fallback] Attempting PyMuPDF...")
    text = extract_with_pymupdf(pdf)

    # If PyMuPDF fails, try pypdfium2
    if not text:
        print("[fallback] PyMuPDF not available, trying pypdfium2...")
        text = extract_with_pypdfium2(pdf)

    # If both fail, error out
    if not text:
        raise RuntimeError("Both PyMuPDF and pypdfium2 failed to extract text")

    # Determine output path
    if output_path:
        out = Path(output_path)
    else:
        out = pdf.with_suffix(".txt")

    # Write text
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")

    return str(out)


def simple_markdown_convert(pdf_path: str, output_path: Optional[str] = None) -> str:
    """Convert PDF to simple Markdown (no complex formatting)."""
    pdf = Path(pdf_path)
    text = fallback_convert(pdf_path, None)

    # Simple markdown formatting
    lines = text.split("\n\n")
    md_lines = []
    for line in lines:
        if line.strip():
            md_lines.append(line)
        else:
            md_lines.append("")  # Preserve paragraph breaks

    md_text = "\n\n".join(md_lines)

    # Determine output path
    if output_path:
        out = Path(output_path)
    else:
        out = pdf.with_suffix(".md")

    # Write markdown
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md_text, encoding="utf-8")

    print(f"[fallback] Simple markdown saved to: {out}")

    return str(out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fallback PDF converter (no marker-pdf)")
    parser.add_argument("pdf", help="PDF file to convert")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--markdown", "-m", action="store_true", help="Output as markdown")

    args = parser.parse_args()

    if args.markdown:
        simple_markdown_convert(args.pdf, args.output)
    else:
        fallback_convert(args.pdf, args.output)
