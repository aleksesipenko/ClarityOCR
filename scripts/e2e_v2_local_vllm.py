#!/usr/bin/env python3
from __future__ import annotations

import json
import mimetypes
import os
import sys
import time
import uuid
from pathlib import Path
from urllib import error, request


API_BASE = os.getenv("CLARITY_API_BASE", "http://127.0.0.1:8008")
VLLM_BASE = os.getenv("VLLM_API_BASE", "http://127.0.0.1:8000")
PDF_PATH = Path(os.getenv("E2E_INPUT_FILE", "test.pdf")).resolve()
WAIT_VLLM_SEC = int(os.getenv("E2E_WAIT_VLLM_SEC", "1200"))
WAIT_JOB_SEC = int(os.getenv("E2E_WAIT_JOB_SEC", "2400"))


def http_json(method: str, url: str, payload: dict | None = None, headers: dict | None = None) -> dict:
    raw = None
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    if payload is not None:
        raw = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    req = request.Request(url=url, data=raw, method=method, headers=req_headers)
    with request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def build_multipart(file_path: Path, field_name: str = "files") -> tuple[bytes, str]:
    boundary = f"----clarity-e2e-{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    file_bytes = file_path.read_bytes()
    chunks = []
    chunks.append(f"--{boundary}\r\n".encode("utf-8"))
    chunks.append(
        (
            f'Content-Disposition: form-data; name="{field_name}"; filename="{file_path.name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
    )
    chunks.append(file_bytes)
    chunks.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(chunks)
    return body, f"multipart/form-data; boundary={boundary}"


def upload_file(file_path: Path) -> dict:
    body, content_type = build_multipart(file_path)
    req = request.Request(
        url=f"{API_BASE}/api/v2/uploads",
        data=body,
        method="POST",
        headers={"Content-Type": content_type, "Accept": "application/json"},
    )
    with request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_vllm() -> None:
    deadline = time.time() + WAIT_VLLM_SEC
    last_err = ""
    while time.time() < deadline:
        try:
            data = http_json("GET", f"{VLLM_BASE}/v1/models")
            if data.get("data"):
                print(f"[ok] vllm ready, models={len(data['data'])}")
                return
        except Exception as exc:
            last_err = str(exc)
        time.sleep(5)
    raise RuntimeError(f"vllm not ready in {WAIT_VLLM_SEC}s; last_error={last_err}")


def wait_for_job(job_id: str) -> dict:
    deadline = time.time() + WAIT_JOB_SEC
    last = {}
    while time.time() < deadline:
        last = http_json("GET", f"{API_BASE}/api/v2/jobs/{job_id}")
        status = last.get("status")
        if status in {"completed", "partial", "failed", "canceled"}:
            return last
        time.sleep(3)
    raise RuntimeError(f"job timeout after {WAIT_JOB_SEC}s, last={last}")


def main() -> int:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"input file not found: {PDF_PATH}")

    print(f"[info] api={API_BASE}")
    print(f"[info] vllm={VLLM_BASE}")
    print(f"[info] file={PDF_PATH}")

    print("[step] waiting vllm readiness")
    wait_for_vllm()

    print("[step] upload")
    uploaded = upload_file(PDF_PATH)
    inputs = uploaded.get("inputs", [])
    if not inputs:
        raise RuntimeError(f"upload returned no inputs: {uploaded}")

    payload = {
        "api_version": "v2",
        "meta_schema_version": "1.0",
        "client_id": "local-e2e",
        "client_request_id": f"e2e-{uuid.uuid4()}",
        "inputs": inputs,
        "mode": "ocr_plus_polish",
        "preset": "speed",
        "naming_policy": "deterministic_only",
        "polish": "on",
    }

    print("[step] start job")
    started = http_json("POST", f"{API_BASE}/api/v2/jobs", payload)
    job_id = started["job_id"]
    print(f"[info] job_id={job_id}")

    print("[step] wait job")
    final_job = wait_for_job(job_id)
    final_status = final_job.get("status")
    print(f"[info] final_status={final_status}")

    files = http_json("GET", f"{API_BASE}/api/v2/jobs/{job_id}/files").get("files", [])
    events = http_json("GET", f"{API_BASE}/api/v2/jobs/{job_id}/events?limit=500").get("events", [])
    artifacts = http_json("GET", f"{API_BASE}/api/v2/jobs/{job_id}/artifacts").get("artifacts", [])
    event_codes = [e.get("event_code") for e in events]

    polish_applied = "file_polish_applied" in event_codes
    polish_fallback = "file_polish_fallback" in event_codes

    downloaded = None
    if artifacts:
        first = artifacts[0]
        url = f"{API_BASE}{first['download_url']}"
        out_path = Path("output_v2") / f"{job_id}-artifact.bin"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with request.urlopen(url, timeout=300) as resp:
            out_path.write_bytes(resp.read())
        downloaded = str(out_path.resolve())

    summary = {
        "job_id": job_id,
        "final_status": final_status,
        "files_count": len(files),
        "events_count": len(events),
        "artifacts_count": len(artifacts),
        "polish_applied_event": polish_applied,
        "polish_fallback_event": polish_fallback,
        "downloaded_artifact": downloaded,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if final_status not in {"completed", "partial"}:
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except error.HTTPError as exc:
        print(f"[http-error] {exc.code} {exc.reason}", file=sys.stderr)
        body = exc.read().decode("utf-8", errors="replace")
        if body:
            print(body, file=sys.stderr)
        raise
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise
