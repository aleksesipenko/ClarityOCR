# ClarityOCR

Server-first OCR pipeline with V2 API, async job orchestration, and VPS/Docker-ready web console.

## Status

- V1 API/UI: removed from active runtime.
- Apple Silicon/MPS support: removed.
- Runtime target: NVIDIA CUDA on server, with CPU fallback for local debugging/tests.

## Quick Start (Docker, Recommended)

```bash
docker compose up -d --build
```

Open `http://localhost:8008`.
VLLM OpenAI-compatible endpoint: `http://localhost:8000/v1`.

Local full-cycle smoke test:

```bash
set E2E_INPUT_FILE=sample_real.pdf
py -3 -u scripts\e2e_v2_local_vllm.py
```

## V2 Console Workflow

1. Upload files with `POST /api/v2/uploads`.
2. Submit job with `POST /api/v2/jobs`.
3. Track `GET /api/v2/jobs/{job_id}`, `/files`, `/events`.
4. Download outputs via `GET /api/v2/jobs/{job_id}/artifacts` and artifact download URLs.

## Requirements

- Python 3.10-3.12
- NVIDIA GPU + CUDA for production OCR performance
- CPU mode supported for local debug/testing

## Local Development (PC)

```bash
pip install -r requirements.txt
pip install -e .
clarityocr serve --host 127.0.0.1 --port 8008
```

Run tests:

```bash
py -3 -m pytest -q
```

## API

See `API_GUIDE.md` for the V2 contract.
