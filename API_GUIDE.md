# ClarityOCR API Guide for AI Agents

ClarityOCR is designed to be easily integrated into AI workflows. This guide covers the versioned API endpoints available for agents.

## Authentication

If `API_USER` and `API_PASSWORD` environment variables are set, all endpoints require HTTP Basic Authentication.

## Endpoints

### 1. Scan Directory
`GET /api/scan?dir={path}&max_pages={n}`

Lists all supported files (PDF, ZIP, Images) in a directory.

### 2. Start OCR Job
`POST /api/job/start`

```json
{
  "files": ["/path/to/file1.pdf", "/path/to/archive.zip"],
  "output_dir": "/path/to/output",
  "preset": "balanced",
  "parallel": 2,
  "auto_fallback": true
}
```
**Presets:** `speed`, `balanced`, `quality`.

### 3. Job Status
`GET /api/job/status`

Returns progress of the current OCR job.

### 4. LLM Polish
`POST /api/llm/job/start`

```json
{
  "files": ["/path/to/file.md"],
  "base_url": "http://localhost:1234/v1"
}
```

### 5. Export Document
`POST /api/v1/export`

```json
{
  "file": "/path/to/file.md",
  "format": "docx",
  "reference_doc": "/path/to/style.docx",
  "options": {
    "font-size": "12pt"
  }
}
```

### 6. File Management
- `GET /api/v1/file/read?path={path}`: Read file content.
- `POST /api/v1/file/save`: Save file content (`path` and `content` in body).

## Integration Example (Python)

```python
import requests

API_URL = "http://your-vps:8008"
AUTH = ("admin", "yourpassword")

# Start a job
res = requests.post(f"{API_URL}/api/job/start", auth=AUTH, json={
    "files": ["/app/inputs/doc.pdf"],
    "preset": "quality"
})
print(res.json())
```
