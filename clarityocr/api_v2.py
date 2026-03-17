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
from sqlalchemy import desc, text

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

    # ROBUSTNESS: Check disk space before writing
    min_free_mb = _env_int("V2_MIN_FREE_DISK_MB", 500)
    try:
        disk = shutil.disk_usage(str(batch_dir))
        if disk.free < min_free_mb * 1024 * 1024:
            shutil.rmtree(batch_dir, ignore_errors=True)
            raise HTTPException(
                status_code=507,
                detail=f"Insufficient disk space: {disk.free // (1024*1024)}MB free, need at least {min_free_mb}MB"
            )
    except OSError:
        pass  # disk_usage failed, proceed anyway

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
                    try:
                        out.write(chunk)
                    except OSError as disk_err:
                        raise HTTPException(
                            status_code=507,
                            detail=f"Disk write failed for {original_name}: {disk_err}"
                        )
            await upload.close()

            # ROBUSTNESS: Reject empty files
            if size == 0:
                target.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail=f"Empty file rejected: {original_name}")

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

    # ROBUSTNESS: Verify input files exist before creating job
    for input_path in request.inputs:
        if not input_path.startswith("http://") and not input_path.startswith("https://"):
            if not Path(input_path).exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"Input file not found on server: {Path(input_path).name}. Upload files first via /uploads."
                )

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
    """
    Get job status with progress and ETA (Phase 1.4).
    Returns: overall_progress_pct, eta_seconds, files_done, files_total.
    """
    with get_session() as session:
        job = session.query(Job).filter_by(job_id=job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        files = session.query(JobFile).filter_by(job_id=job_id).all()

        # Calculate progress
        files_total = len(files)
        files_done = sum(1 for f in files if f.status == "completed")

        # Calculate overall progress from pages
        total_pages = sum(f.pages_total or 0 for f in files if f.pages_total)
        done_pages = sum(f.pages_done or 0 for f in files if f.pages_done)

        if total_pages > 0:
            overall_progress_pct = int((done_pages / total_pages) * 100)
        elif files_total > 0:
            overall_progress_pct = int((files_done / files_total) * 100)
        else:
            overall_progress_pct = 0

        # Calculate ETA (linear extrapolation)
        eta_seconds = None
        if files_total > 0 and files_done > 0 and files_done < files_total:
            # Use average time per completed file
            completed_files = [f for f in files if f.status == "completed" and f.duration_ms]
            if completed_files:
                avg_duration_ms = sum(f.duration_ms for f in completed_files) / len(completed_files)
                remaining_files = files_total - files_done
                eta_seconds = int((remaining_files * avg_duration_ms) / 1000)

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
            # Phase 1.4: Progress/ETA
            "overall_progress_pct": overall_progress_pct,
            "eta_seconds": eta_seconds,
            "files_done": files_done,
            "files_total": files_total,
            # ROBUSTNESS: Include error summary so callers don't need a second request
            "errors": [f.last_error_message for f in files if f.last_error_message][:5],
            "error_count": sum(1 for f in files if f.last_error_message),
        }


@router.get("/jobs/{job_id}/files")
def get_job_files(job_id: str):
    """
    Get job files with stage, progress, and page tracking (Phase 1.4).
    Returns: stage, stage_progress_pct, pages_done, pages_total, duration_ms.
    """
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
                    # Phase 1.4: Stage and progress tracking
                    "stage": f.stage,
                    "stage_progress_pct": f.stage_progress_pct,
                    "pages_done": f.pages_done,
                    "pages_total": f.pages_total,
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

        # Path traversal protection: ensure artifact is under allowed roots
        resolved_abs = resolved.resolve()
        allowed_roots = [Path("output_v2").resolve(), _upload_root().resolve()]
        if not any(str(resolved_abs).startswith(str(root)) for root in allowed_roots):
            raise HTTPException(status_code=403, detail="Artifact path outside allowed directory")

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
        job = session.query(Job).filter_by(job_id=job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
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


@router.get("/version", summary="Version Info")
def version_info():
    """Return build version, canary routing status."""
    from .version import __version__
    from .canary import canary_status
    return {
        "version": __version__,
        "canary": canary_status(),
    }


@router.get("/health/live", summary="Liveness Check")
def liveness():
    return {"status": "alive"}


def _probe_ocr_core() -> str:
    """Probe OCR core (marker-pdf availability)."""
    try:
        import marker  # noqa
        return "ready"
    except ImportError:
        return "unavailable"
    except Exception:
        return "error"


def _probe_db() -> str:
    """Probe database connectivity."""
    try:
        with get_session() as session:
            session.execute(text("SELECT 1"))
            return "ready"
    except Exception:
        return "unavailable"


def _probe_llm() -> str:
    """Probe LLM service (optional polish)."""
    import os
    llm_base_url = os.getenv("V2_LLM_BASE_URL", "http://localhost:1234/v1")

    # Quick check if LLM endpoint is configured
    if not llm_base_url or llm_base_url == "http://localhost:1234/v1":
        # Default unconfigured state
        return "not_configured"

    try:
        import httpx
        response = httpx.get(f"{llm_base_url.rstrip('/')}/models", timeout=2.0)
        if response.status_code == 200:
            return "ready"
        return "degraded"
    except Exception:
        return "unavailable"


@router.get("/health/ready", response_model=HealthReadyResponse, summary="Readiness Check")
def readiness():
    """
    Enhanced readiness check with real component probes (Phase 1.3).
    Returns nested status for OCR core, database, and LLM service.
    """
    return HealthReadyResponse(
        ocr_core=_probe_ocr_core(),
        db=_probe_db(),
        llm=_probe_llm()
    )
