#!/usr/bin/env python3
"""
Simple PDF to Text Converter (Fallback Mode)

Uses pypdfium2 for reliable PDF text extraction on all platforms.
Works without marker-pdf dependency.

Performance: Fast and low-memory, useful for local debugging.
"""

import sys
from pathlib import Path
from typing import Optional
import warnings

warnings.filterwarnings("ignore")


def _open_pdf(pdf_path: str):
    import pypdfium2

    if hasattr(pypdfium2, "PdfDocument"):
        return pypdfium2.PdfDocument(pdf_path)
    if hasattr(pypdfium2, "Pdf"):  # backward compatibility
        return pypdfium2.Pdf(pdf_path)
    raise RuntimeError("Unsupported pypdfium2 API: no PdfDocument/Pdf found")


def _page_text(page) -> str:
    text_page = page.get_textpage()
    if hasattr(text_page, "get_text_range"):
        text = text_page.get_text_range()
    elif hasattr(text_page, "get_text"):
        text = text_page.get_text()
    else:
        text = ""
    return (text or "").strip()


def extract_text(pdf_path: str, output_path: Optional[str] = None) -> str:
    """
    Extract text from PDF using pypdfium2.

    Args:
        pdf_path: Path to PDF file
        output_path: Optional output path (defaults to same name with .md extension)

    Returns:
        str: Path to output file
    """
    pdf = Path(pdf_path)

    if not pdf.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Determine output path
    if output_path:
        out = Path(output_path)
    else:
        out = pdf.with_suffix(".txt")

    try:
        pdf_doc = _open_pdf(pdf_path)
        n_pages = len(pdf_doc)

        text_parts = []
        for page_index in range(n_pages):
            page = pdf_doc[page_index]
            text_parts.append(_page_text(page))

        text = "\n\n".join(text_parts)

        # Write output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")

        print(f"OK: extracted {n_pages} pages from {pdf.name}")
        print(f"Output: {out}")

        return str(out)

    except Exception as e:
        print(f"Error extracting text: {e}")
        raise


def convert_to_markdown(pdf_path: str, output_path: Optional[str] = None) -> str:
    """
    Convert PDF to simple Markdown.

    Args:
        pdf_path: Path to PDF file
        output_path: Optional output path

    Returns:
        str: Path to output file
    """
    pdf = Path(pdf_path)

    if not pdf.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Determine output path
    if output_path:
        out = Path(output_path)
    else:
        out = pdf.with_suffix(".md")

    try:
        pdf_doc = _open_pdf(pdf_path)
        n_pages = len(pdf_doc)

        md_lines = []
        for page_index in range(n_pages):
            page = pdf_doc[page_index]
            text_page = _page_text(page)

            if text_page:
                md_lines.append(text_page)
            else:
                md_lines.append("")  # Empty page

        md_text = "\n\n".join(md_lines)

        # Write output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md_text, encoding="utf-8")

        print(f"OK: converted {pdf.name} ({n_pages} pages) to Markdown")
        print(f"Output: {out}")

        return str(out)

    except Exception as e:
        print(f"Error converting to Markdown: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple PDF converter using pypdfium2 (no marker-pdf dependency)"
    )
    parser.add_argument("pdf", help="PDF file to convert")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--markdown", "-m", action="store_true", help="Output as Markdown")

    args = parser.parse_args()

    if args.markdown:
        convert_to_markdown(args.pdf, args.output)
    else:
        extract_text(args.pdf, args.output)
