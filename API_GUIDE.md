# ClarityOCR API Guide (V2 Only)

ClarityOCR exposes a V2 asynchronous API for server-side OCR/merge pipelines.

If `API_USER` and `API_PASSWORD` are set, all `/api/v2/*` endpoints require HTTP Basic Auth.

## Health

- `GET /api/v2/health/live` -> `{"status":"alive"}`
- `GET /api/v2/health/ready` -> component readiness (`ocr_core`, `db`, `llm`)

## Upload

- `POST /api/v2/uploads` (`multipart/form-data`, field: `files`)
- Response includes:
  - `upload_id`
  - `inputs` (server-side absolute paths)
  - uploaded file metadata

## Job Submission

- `POST /api/v2/jobs`
- Idempotent by (`client_id`, `client_request_id`) + payload hash.
- Supported `mode` values:
  - `ocr_only`
  - `ocr_plus_metadata`
  - `merge_only`
  - `merge_then_ocr`
  - `ocr_plus_polish`

## Job Tracking

- `GET /api/v2/jobs?limit=20`
- `GET /api/v2/jobs/{job_id}`
- `GET /api/v2/jobs/{job_id}/files`
- `GET /api/v2/jobs/{job_id}/events`

## Artifacts

- `GET /api/v2/jobs/{job_id}/artifacts`
- `GET /api/v2/artifacts/{artifact_id}/download`

## Control

- `POST /api/v2/jobs/{job_id}/cancel`
- `POST /api/v2/jobs/{job_id}/retry-failed`

## Merge Shortcuts

- `POST /api/v2/merge` (forces `mode=merge_only`)
- `POST /api/v2/merge-and-ocr` (forces `mode=merge_then_ocr`)

## Output Formats

ClarityOCR produces the following artifact types per OCR job:

| Type | Extension | Description |
|------|-----------|-------------|
| `md` | `.md` | **Primary output: Markdown** — full OCR text, optionally polished by LLM grounding |
| `meta` | `.meta.json` | **JSON metadata** — title, language, page count, polish status, confidence source |
| `naming` | `.naming.json` | **JSON naming data** — slug, suggested filename, extracted title tokens |
| `manifest` | `batch_manifest.json` | **JSON job manifest** — per-file artifact list with SHA-256 checksums |
| `merge_report` | `merge_report.json` | **JSON merge report** — produced for merge jobs, lists merged inputs + ordering |
| `merged_pdf` | `merged.pdf` | Merged PDF — produced by `merge_only` / `merge_then_ocr` as intermediate artifact |

### Enterprise scope: confirmed output formats

- **Markdown (`.md`)** — always produced for OCR jobs; primary consumer format for Alfred and downstream pipelines
- **JSON (`.meta.json`, `.naming.json`, `batch_manifest.json`)** — always produced alongside Markdown; structured metadata for integration, indexing, and automation
- **DOCX — not produced.** Alfred handles document output independently. No DOCX output is planned.
- **PDF — only produced as an intermediate merge artifact**, not a final OCR output.

## Security Constraints

- Batch size limits
- URL SSRF protections (private/loopback blocked)
- ZIP safety checks (depth, entries, ratio, total unpacked size)
