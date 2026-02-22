import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import desc

from .db import Artifact, Job, JobEvent, JobFile, get_session
from .job_manager import (
    IdempotencyError,
    append_event,
    create_job,
    transition_file_status,
    transition_job_status,
)
from .security import SecurityValidationError, validate_callback_url, validate_inputs_security

router = APIRouter()

ALLOWED_MODES = {"ocr_only", "ocr_plus_metadata", "merge_only", "merge_then_ocr", "ocr_plus_polish"}
ALLOWED_PRESETS = {"speed", "balanced", "quality"}
ALLOWED_UPLOAD_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp", ".zip"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _upload_root() -> Path:
    configured = os.getenv("V2_UPLOAD_ROOT")
    if configured:
        root = Path(configured)
    else:
        root = Path.home() / ".clarityocr" / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_filename(filename: str) -> str:
    base = Path(filename).name.strip() or "file"
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return safe[:180] or "file"


def _parse_payload(payload: Optional[str]) -> Optional[Dict[str, Any]]:
    if not payload:
        return None
    try:
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return {"raw": payload}
    return {"raw": payload}


def _resolve_existing_path(path_text: str) -> Path:
    path = Path(path_text)
    candidates: List[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(Path.cwd() / path)
        candidates.append(Path.home() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


class JobStartV2Request(BaseModel):
    api_version: str = Field("v2")
    meta_schema_version: str = Field("1.0")
    client_id: str = Field(default="default")
    client_request_id: str = Field(...)
    inputs: List[str] = Field(..., min_length=1)
    mode: str = Field(
        "ocr_plus_metadata",
        description="ocr_only, ocr_plus_metadata, merge_only, merge_then_ocr, ocr_plus_polish",
    )
    preset: str = Field("balanced")
    naming_policy: str = Field(
        "on", description="on (deterministic + LLM if available) or deterministic_only or off"
    )
    polish: str = Field("off", description="off, auto, on")
    callback_url: Optional[str] = None
    callback_secret: Optional[str] = None


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str


class HealthReadyResponse(BaseModel):
    ocr_core: str
    db: str
    llm: str


class UploadResponse(BaseModel):
    upload_id: str
    inputs: List[str]
    files: List[Dict[str, Any]]


@router.post("/uploads", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_inputs(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded")
    if len(files) > 1000:
        raise HTTPException(status_code=400, detail="Batch size exceeds limit of 1000")

    max_file_size_mb = _env_int("V2_MAX_INPUT_FILE_SIZE_MB", 512)
    max_file_bytes = max_file_size_mb * 1024 * 1024

    upload_id = str(uuid.uuid4())
    batch_dir = _upload_root() / upload_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    saved_inputs: List[str] = []
    saved_meta: List[Dict[str, Any]] = []

    try:
        for index, upload in enumerate(files, start=1):
            original_name = upload.filename or f"file-{index}"
            safe_name = _safe_filename(original_name)
            suffix = Path(safe_name).suffix.lower()
            if suffix not in ALLOWED_UPLOAD_EXTS:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {original_name}")

            target = batch_dir / f"{index:04d}_{safe_name}"
            size = 0
            with target.open("wb") as out:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_file_bytes:
                        raise HTTPException(
                            status_code=400,
                            detail=f"File too large: {original_name} (limit {max_file_size_mb} MB)",
                        )
                    out.write(chunk)
            await upload.close()

            saved_inputs.append(str(target.resolve()))
            saved_meta.append(
                {
                    "name": original_name,
                    "server_path": str(target.resolve()),
                    "size_bytes": size,
                    "content_type": upload.content_type or "application/octet-stream",
                }
            )

        validate_inputs_security(saved_inputs)
        return UploadResponse(upload_id=upload_id, inputs=saved_inputs, files=saved_meta)
    except Exception:
        shutil.rmtree(batch_dir, ignore_errors=True)
        raise


@router.post("/jobs", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def start_job(request: JobStartV2Request):
    if len(request.inputs) > 1000:
        raise HTTPException(status_code=400, detail="Batch size exceeds limit of 1000")
    if request.mode not in ALLOWED_MODES:
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {request.mode}")
    if request.preset not in ALLOWED_PRESETS:
        raise HTTPException(status_code=400, detail=f"Unsupported preset: {request.preset}")

    try:
        validate_inputs_security(request.inputs)
        if request.callback_url:
            validate_callback_url(request.callback_url)
    except SecurityValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    payload_dict = request.model_dump()
    try:
        with get_session() as session:
            job = create_job(
                session=session,
                client_id=request.client_id,
                client_request_id=request.client_request_id,
                payload=payload_dict,
                files=request.inputs,
                mode=request.mode,
                preset=request.preset,
            )
            return JobResponse(job_id=job.job_id, status=job.status, message="Job accepted")
    except IdempotencyError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post("/merge", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def start_merge(request: JobStartV2Request):
    request.mode = "merge_only"
    return start_job(request)


@router.post("/merge-and-ocr", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def start_merge_and_ocr(request: JobStartV2Request):
    request.mode = "merge_then_ocr"
    return start_job(request)


@router.get("/jobs")
def list_jobs(
    limit: int = Query(20, ge=1, le=100),
    status_filter: Optional[str] = Query(default=None, alias="status"),
    mode: Optional[str] = Query(default=None),
):
    with get_session() as session:
        query = session.query(Job)
        if status_filter:
            query = query.filter(Job.status == status_filter)
        if mode:
            query = query.filter(Job.mode == mode)

        rows = query.order_by(desc(Job.created_at)).limit(limit).all()
        return {
            "jobs": [
                {
                    "job_id": j.job_id,
                    "status": j.status,
                    "mode": j.mode,
                    "preset": j.preset,
                    "client_id": j.client_id,
                    "client_request_id": j.client_request_id,
                    "created_at": j.created_at,
                    "accepted_at": j.accepted_at,
                }
                for j in rows
            ]
        }


@router.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    with get_session() as session:
        job = session.query(Job).filter_by(job_id=job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        config = _parse_payload(job.config)
        return {
            "job_id": job.job_id,
            "status": job.status,
            "mode": job.mode,
            "preset": job.preset,
            "client_id": job.client_id,
            "client_request_id": job.client_request_id,
            "created_at": job.created_at,
            "accepted_at": job.accepted_at,
            "config": config,
        }


@router.get("/jobs/{job_id}/files")
def get_job_files(job_id: str):
    with get_session() as session:
        files = session.query(JobFile).filter_by(job_id=job_id).all()
        return {
            "job_id": job_id,
            "files": [
                {
                    "id": f.id,
                    "input_path": f.input_path,
                    "status": f.status,
                    "attempt": f.attempt,
                    "max_attempts": f.max_attempts,
                    "duration_ms": f.duration_ms,
                    "error": f.last_error_message,
                }
                for f in files
            ],
        }


@router.get("/jobs/{job_id}/artifacts")
def get_job_artifacts(job_id: str):
    with get_session() as session:
        artifacts = session.query(Artifact).filter(Artifact.job_id == job_id).all()
        return {
            "job_id": job_id,
            "artifacts": [
                {
                    "id": a.id,
                    "type": a.type,
                    "path": a.path,
                    "sha256": a.sha256,
                    "download_url": f"/api/v2/artifacts/{a.id}/download",
                }
                for a in artifacts
            ],
        }


@router.get("/artifacts/{artifact_id}/download")
def download_artifact(artifact_id: str):
    with get_session() as session:
        artifact = session.query(Artifact).filter_by(id=artifact_id).first()
        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")

        resolved = _resolve_existing_path(artifact.path)
        if not resolved.exists() or not resolved.is_file():
            raise HTTPException(status_code=404, detail="Artifact file is missing")

        return FileResponse(str(resolved), filename=resolved.name, media_type="application/octet-stream")


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    with get_session() as session:
        job = session.query(Job).filter_by(job_id=job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        files = session.query(JobFile).filter(
            JobFile.job_id == job_id,
            JobFile.status.in_(["queued", "running", "failed_recoverable"]),
        ).all()
        for f in files:
            try:
                transition_file_status(session, f.id, "canceled")
                append_event(session, job_id, "file_canceled", file_id=f.id)
            except ValueError:
                continue

        if job.status in ["completed"]:
            return {"status": job.status, "job_id": job_id}

        try:
            transition_job_status(session, job_id, "canceled")
            append_event(session, job_id, "job_canceled")
        except ValueError:
            session.refresh(job)
            return {"status": job.status, "job_id": job_id}

        return {"status": "canceled", "job_id": job_id}


@router.post("/jobs/{job_id}/retry-failed")
def retry_failed_job(job_id: str):
    with get_session() as session:
        files = session.query(JobFile).filter(
            JobFile.job_id == job_id,
            JobFile.status.in_(["failed_recoverable", "failed_final", "canceled"]),
        ).all()

        requeued = 0
        for f in files:
            if f.attempt < f.max_attempts:
                transition_file_status(session, f.id, "queued")
                append_event(session, job_id, "file_requeued", file_id=f.id)
                requeued += 1

        if requeued > 0:
            try:
                transition_job_status(session, job_id, "running")
            except ValueError:
                pass

        return {"requeued_count": requeued, "job_id": job_id}


@router.get("/jobs/{job_id}/events")
def get_job_events(job_id: str, limit: int = 200, skip: int = 0):
    with get_session() as session:
        events = (
            session.query(JobEvent)
            .filter_by(job_id=job_id)
            .order_by(JobEvent.timestamp.asc())
            .offset(skip)
            .limit(limit)
            .all()
        )
        return {
            "job_id": job_id,
            "events": [
                {
                    "event_code": e.event_code,
                    "timestamp": e.timestamp,
                    "file_id": e.file_id,
                    "payload": _parse_payload(e.payload),
                }
                for e in events
            ],
        }


@router.get("/health/live", summary="Liveness Check")
def liveness():
    return {"status": "alive"}


@router.get("/health/ready", response_model=HealthReadyResponse, summary="Readiness Check")
def readiness():
    return HealthReadyResponse(ocr_core="ready", db="ready", llm="degraded")
