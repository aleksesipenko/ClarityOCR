#!/usr/bin/env python3
"""
ClarityOCR - Command Line Interface

Usage:
    clarityocr convert [OPTIONS] [PDF_FILES...]
    clarityocr polish [OPTIONS] [MD_FILES...]
    clarityocr serve [OPTIONS]
    clarityocr scan [OPTIONS] DIRECTORY

Commands:
    convert     Convert PDF files to Markdown
    polish      Polish Markdown files with LLM (fix OCR errors)
    serve       Start the Web UI server
    scan        Scan a directory for PDFs and show status
"""

import argparse
import sys
from pathlib import Path


def cmd_convert(args):
    """Convert PDF files to Markdown."""
    from .converter import main as converter_main

    # Build arguments for the converter
    sys.argv = ["clarityocr-convert"]

    if args.input_dir:
        sys.argv.extend(["--input-dir", args.input_dir])
    if args.output_dir:
        sys.argv.extend(["--output-dir", args.output_dir])
    if args.max_pages:
        sys.argv.extend(["--max-pages", str(args.max_pages)])
    if args.no_auto_fallback:
        sys.argv.append("--no-auto-fallback")
    if args.scan:
        sys.argv.append("--scan")

    for pdf in args.pdfs:
        sys.argv.extend(["--pdf", pdf])

    return converter_main()


def cmd_polish(args):
    """Polish Markdown files with LLM."""
    from .polish import main as polish_main

    sys.argv = ["clarityocr-polish"]

    if args.base_url:
        sys.argv.extend(["--base-url", args.base_url])
    if args.model:
        sys.argv.extend(["--model", args.model])
    if args.temperature:
        sys.argv.extend(["--temperature", str(args.temperature)])
    if args.chunk_size:
        sys.argv.extend(["--chunk-size", str(args.chunk_size)])
    if args.max_tokens:
        sys.argv.extend(["--max-tokens", str(args.max_tokens)])
    if args.dry_run:
        sys.argv.append("--dry-run")
    if args.verbose:
        sys.argv.append("--verbose")
    if args.dir:
        sys.argv.extend(["--dir", args.dir])

    for f in args.files:
        sys.argv.extend(["--file", f])

    return polish_main()


def cmd_serve(args):
    """Start the Web UI server."""
    from .server import run_server

    run_server(host=args.host, port=args.port)
    return 0


def cmd_scan(args):
    """Scan directory for PDFs and show status."""
    from .converter import scan_pdfs
    from pathlib import Path

    input_dir = Path(args.directory)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "md"

    return scan_pdfs(input_dir, output_dir, args.max_pages)


def main():
    parser = argparse.ArgumentParser(
        prog="clarityocr",
        description="High-quality PDF to Markdown conversion with GPU-accelerated OCR and LLM post-processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    clarityocr convert --input-dir ./pdfs --output-dir ./markdown
    clarityocr polish --file document.md --base-url http://localhost:1234/v1
    clarityocr serve --port 8008
    clarityocr scan ./documents
        """,
    )
    parser.add_argument("--version", action="version", version="ClarityOCR 1.0.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert PDF files to Markdown")
    convert_parser.add_argument("--input-dir", "-i", help="Directory with PDF files")
    convert_parser.add_argument("--output-dir", "-o", help="Output directory for Markdown files")
    convert_parser.add_argument(
        "--max-pages", type=int, default=500, help="Skip PDFs with more pages"
    )
    convert_parser.add_argument(
        "--no-auto-fallback", action="store_true", help="Disable batch size auto-reduction"
    )
    convert_parser.add_argument("--scan", action="store_true", help="Scan only, no conversion")
    convert_parser.add_argument("pdfs", nargs="*", help="Specific PDF files to convert")
    convert_parser.set_defaults(func=cmd_convert)

    # Polish command
    polish_parser = subparsers.add_parser(
        "polish", help="Polish Markdown with LLM (fix OCR errors)"
    )
    polish_parser.add_argument("files", nargs="*", help="Markdown files to polish")
    polish_parser.add_argument("--dir", "-d", help="Directory with Markdown files")
    polish_parser.add_argument(
        "--base-url", default="http://localhost:1234/v1", help="LM Studio API URL"
    )
    polish_parser.add_argument("--model", default="local-model", help="Model identifier")
    polish_parser.add_argument(
        "--temperature", type=float, default=0.1, help="Generation temperature"
    )
    polish_parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size in tokens")
    polish_parser.add_argument("--max-tokens", type=int, default=4096, help="Max output tokens")
    polish_parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    polish_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    polish_parser.set_defaults(func=cmd_polish)

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start Web UI server")
    serve_parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)"
    )
    serve_parser.add_argument(
        "--port", "-p", type=int, default=8008, help="Port to listen (default: 8008)"
    )
    serve_parser.set_defaults(func=cmd_serve)

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan directory for PDFs")
    scan_parser.add_argument("directory", help="Directory to scan")
    scan_parser.add_argument("--output-dir", "-o", help="Output directory for status check")
    scan_parser.add_argument("--max-pages", type=int, default=500, help="Max pages threshold")
    scan_parser.set_defaults(func=cmd_scan)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
