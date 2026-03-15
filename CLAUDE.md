# CLAUDE.md — ClarityOCR

## Project Overview

ClarityOCR is a standalone tool for converting PDF documents to clean Markdown text using GPU-accelerated OCR (marker-pdf/Surya models) with optional LLM post-processing. It provides a CLI, REST API, and real-time Web UI.

## Repository Structure

```
clarityocr/                    # Main Python package
├── cli.py                     # CLI entry point (convert, polish, serve, scan)
├── converter.py               # Core PDF→Markdown conversion logic (marker-pdf wrapper)
├── device_utils.py            # GPU/MPS/CPU detection and memory utilities
├── fallback_converter.py      # PyMuPDF/pypdfium2 fallback extractor
├── polish.py                  # LLM-based OCR error correction (OpenAI-compatible API)
├── server.py                  # FastAPI web server with SSE progress streaming
├── simple_converter.py        # Minimal pypdfium2-only converter
└── web/static/                # Vanilla JS/HTML/CSS web UI assets
tests/                         # Pytest test suite
├── conftest.py                # Adds repo root to sys.path
├── test_markers.py            # Page marker normalization tests
├── test_mojibake.py           # Encoding detection/repair tests
└── test_polish.py             # LLM polish utility tests
```

## Quick Reference

| Task | Command |
|------|---------|
| Install (dev) | `pip install -e ".[dev]"` |
| Run tests | `pytest` |
| Lint | `ruff check .` |
| Type check | `mypy clarityocr/` |
| Start server | `clarityocr serve --port 8008` |
| Convert PDFs | `clarityocr convert FILE.pdf` |
| Docker build | `docker compose up --build` |

## Build & Dependencies

- **Python**: >=3.10, <3.13 (PyTorch CUDA constraint)
- **Build backend**: setuptools via `pyproject.toml`
- **Core deps**: marker-pdf, torch, fastapi, uvicorn, openai, pypdfium2, psutil
- **Dev deps** (optional `[dev]` extra): pytest, ruff, mypy

Install for development:
```bash
pip install -e ".[dev]"
```

## Testing

Tests use **pytest** with no special markers or fixtures beyond `conftest.py` (which adds the repo root to `sys.path`).

```bash
pytest              # Run all tests
pytest tests/       # Run specific directory
pytest -v           # Verbose output
```

Tests cover:
- Page marker normalization (`{N}` → `[p:N]`, 0-based → 1-based)
- Mojibake/encoding detection and repair (CP1251 vs Latin-1)
- LLM polish utilities (token estimation, table detection, text splitting)

## Linting & Style

Configured in `pyproject.toml`:

- **Ruff** — line length 100, target Python 3.10
  - Rules: `E`, `F`, `W`, `I` (isort), `UP` (pyupgrade), `B` (bugbear), `C4` (comprehensions)
  - `E501` (line-too-long) is ignored
- **MyPy** — `warn_return_any=true`, `ignore_missing_imports=true`

Code conventions observed in the codebase:
- Type hints on all public functions (see `device_utils.py` for a good example)
- Docstrings on public functions
- `Literal` types for constrained values (e.g., `DeviceType = Literal["cuda", "mps", "cpu"]`)
- Dataclasses for configuration objects (e.g., `PolishConfig`)

## Architecture Notes

**Entry points** (defined in `pyproject.toml`):
- `clarityocr` → `clarityocr.cli:main`
- `clarityocr-server` → `clarityocr.server:main`

**Key design patterns**:
- Subprocess-based async: conversion/polishing runs in a subprocess; main thread reads progress via SSE
- Graceful degradation: marker-pdf → PyMuPDF → pypdfium2 fallback chain
- VRAM watchdog: auto-reduces batch sizes on OOM, polls GPU stats via nvidia-smi
- Page markers: `[p:N]` format for stable citation references

**Device support**: NVIDIA CUDA, Apple Silicon MPS, CPU fallback — all abstracted in `device_utils.py`.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `API_USER` / `API_PASSWORD` | HTTP Basic Auth credentials for the API |
| `CORS_ORIGINS` | Comma-separated allowed origins (default: localhost:8008) |
| `CUDA_VISIBLE_DEVICES` | GPU selection |

## Docker

Uses `pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime` base image. Port 8008 is exposed. GPU passthrough is configured in `docker-compose.yml` with persistent volumes for model cache (~15GB) and logs.

```bash
docker compose up --build
```
