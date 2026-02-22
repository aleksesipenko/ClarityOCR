import os
import json
import time
import uuid
import threading
import logging
import traceback
import hashlib
import hmac
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

import httpx

# Avoid circular imports, keep specific imports
from .db import get_session, Job, JobFile, Worker, JobEvent
from .job_manager import (
    transition_file_status, 
    check_and_update_job_completion,
    update_worker_heartbeat,
    append_event,
    record_artifact
)
from .converter import main as run_converter # We will call converter as a python module.
from .naming import generate_naming # to be implemented
from .metadata import generate_metadata # to be implemented

logger = logging.getLogger("pipeline_v2")

# Constants from Implementation Plan v2.2.1
HEARTBEAT_INTERVAL_SEC = 5
LEASE_TIMEOUT_SEC = 30
CLEANUP_INTERVAL_SEC = 60
EVENT_RETENTION_HOURS = int(os.getenv("JOB_EVENTS_TTL_HOURS", "168"))
WORKER_RETENTION_HOURS = int(os.getenv("WORKERS_TTL_HOURS", "48"))
POLISH_TIMEOUT_SEC = int(os.getenv("V2_POLISH_TIMEOUT_SEC", "120"))
POLISH_FAILURE_THRESHOLD = int(os.getenv("V2_POLISH_BREAKER_THRESHOLD", "3"))
POLISH_COOLDOWN_SEC = int(os.getenv("V2_POLISH_BREAKER_COOLDOWN_SEC", "120"))
FINAL_JOB_STATUSES = {"completed", "partial", "failed", "canceled"}


def _file_sha256(filepath: str) -> Optional[str]:
    if not os.path.exists(filepath):
        return None
    h = hashlib.sha256()
    with open(filepath, "rb") as fp:
        for chunk in iter(lambda: fp.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(manifest_path: str, files: list[dict[str, Any]]) -> None:
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"schema_version": "1.0", "files": files}, f, indent=2)


def _fallback_markdown_from_pdf(input_pdf: str, md_path: str) -> None:
    """Generate markdown with lightweight converter when marker pipeline fails."""
    from .simple_converter import convert_to_markdown

    Path(md_path).parent.mkdir(parents=True, exist_ok=True)
    convert_to_markdown(input_pdf, md_path)


class LLMCircuitBreaker:
    """Simple process-local circuit breaker for optional LLM polish stage."""

    def __init__(self, failure_threshold: int, cooldown_sec: int):
        self.failure_threshold = max(1, failure_threshold)
        self.cooldown_sec = max(1, cooldown_sec)
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._opened_until = 0.0
        self._last_error = ""

    def state(self) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            return {
                "status": "open" if now < self._opened_until else "closed",
                "consecutive_failures": self._consecutive_failures,
                "opened_until_epoch_s": self._opened_until,
                "last_error": self._last_error,
            }

    def allow_request(self) -> bool:
        with self._lock:
            return time.time() >= self._opened_until

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._opened_until = 0.0
            self._last_error = ""

    def record_failure(self, reason: str) -> None:
        with self._lock:
            self._consecutive_failures += 1
            self._last_error = reason
            if self._consecutive_failures >= self.failure_threshold:
                self._opened_until = time.time() + self.cooldown_sec


_polish_breaker = LLMCircuitBreaker(
    failure_threshold=POLISH_FAILURE_THRESHOLD,
    cooldown_sec=POLISH_COOLDOWN_SEC,
)

class PipelineWorker(threading.Thread):
    def __init__(self, gpu_id: str = "cpu"):
        super().__init__(daemon=True)
        self.worker_id = str(uuid.uuid4())
        self.gpu_id = gpu_id
        self.stop_event = threading.Event()
        self.current_file_id = None
        self._last_cleanup_ts = 0.0
        
    def run(self):
        logger.info(f"Worker {self.worker_id} starting on GPU {self.gpu_id}")
        
        while not self.stop_event.is_set():
            # Heartbeat & recovery
            try:
                with get_session() as session:
                    update_worker_heartbeat(session, self.worker_id, self.gpu_id)
                    self._recover_stale_files(session)
                    self._cleanup_retention(session)
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error updating heartbeat: {e}")
                
            # Job processing
            try:
                file_id, job_id, input_path = None, None, None
                with get_session() as session:
                    file_to_process = self._acquire_next_file(session)
                    if file_to_process:
                        file_id = file_to_process.id
                        job_id = file_to_process.job_id
                        input_path = file_to_process.input_path
                    
                if file_id:
                    self.current_file_id = file_id
                    self._process_file(file_id, job_id, input_path)
                    self.current_file_id = None
                else:
                    self.stop_event.wait(HEARTBEAT_INTERVAL_SEC)
                    
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error processing: {e}")
                self.stop_event.wait(HEARTBEAT_INTERVAL_SEC)

    def _recover_stale_files(self, session):
        """Find files running on dead workers and requeue/fail them."""
        timeout_threshold = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=LEASE_TIMEOUT_SEC)

        healthy_workers = session.query(Worker).filter(Worker.last_heartbeat >= timeout_threshold).count()
        stale_workers = session.query(Worker).filter(Worker.last_heartbeat < timeout_threshold).all()
        if not stale_workers or healthy_workers > 0:
            return

        running_files = session.query(JobFile).filter(JobFile.status == "running").all()
        for f in running_files:
            try:
                next_attempt = f.attempt + 1
                if next_attempt >= f.max_attempts:
                    transition_file_status(
                        session,
                        f.id,
                        "failed_final",
                        attempt=next_attempt,
                        last_error_code="E_LEASE_TIMEOUT",
                        last_error_message="Recovered stale file exceeded max attempts",
                    )
                    append_event(
                        session,
                        f.job_id,
                        "file_failed_final",
                        file_id=f.id,
                        payload={"reason": "lease_timeout", "attempt": next_attempt},
                    )
                else:
                    transition_file_status(
                        session,
                        f.id,
                        "queued",
                        attempt=next_attempt,
                        last_error_code="E_LEASE_TIMEOUT",
                        last_error_message="Recovered stale file was requeued",
                    )
                    append_event(
                        session,
                        f.job_id,
                        "file_requeued_stale",
                        file_id=f.id,
                        payload={"attempt": next_attempt},
                    )
            except Exception as exc:
                logger.warning("Failed to recover stale file %s: %s", f.id, exc)

    def _cleanup_retention(self, session):
        now = time.time()
        if now - self._last_cleanup_ts < CLEANUP_INTERVAL_SEC:
            return

        self._last_cleanup_ts = now
        utc_now_naive = datetime.now(timezone.utc).replace(tzinfo=None)
        cutoff_events = utc_now_naive - timedelta(hours=EVENT_RETENTION_HOURS)
        cutoff_workers = utc_now_naive - timedelta(hours=WORKER_RETENTION_HOURS)

        deleted_events = (
            session.query(JobEvent)
            .filter(JobEvent.timestamp < cutoff_events)
            .delete(synchronize_session=False)
        )
        (
            session.query(Worker)
            .filter(Worker.last_heartbeat < cutoff_workers)
            .delete(synchronize_session=False)
        )
        session.commit()

        if deleted_events:
            logger.info("Worker %s cleaned up %s old events", self.worker_id, deleted_events)

    def _acquire_next_file(self, session):
        # We process files ordered by created_at (implied by job ID/creation time)
        # Find first queued file
        queued_file = session.query(JobFile).join(Job).filter(
            JobFile.status == "queued",
            Job.status.in_(["queued", "running"])
        ).first()
        
        if not queued_file:
            return None
            
        try:
            # Transition file state
            updated_file = transition_file_status(session, queued_file.id, "running")
            # Transition Job state if it was queued
            if updated_file.job.status == "queued":
                from .job_manager import transition_job_status
                transition_job_status(session, updated_file.job_id, "running")
                
            append_event(session, updated_file.job_id, "file_running", file_id=updated_file.id)
            return updated_file
        except ValueError:
            return None # Invalid transition, skip

    def _should_attempt_polish(self, mode: str, config: Dict[str, Any]) -> bool:
        polish_setting = str(config.get("polish", "off")).lower()
        if mode == "ocr_plus_polish":
            return True
        return polish_setting in {"on", "auto"}

    def _try_optional_polish(self, md_path: str, mode: str, config: Dict[str, Any], job_id: str, file_id: str) -> Dict[str, Any]:
        if not self._should_attempt_polish(mode, config):
            return {
                "polish_applied": False,
                "fallback_reason": "disabled",
                "circuit_breaker_status": _polish_breaker.state(),
            }

        if not _polish_breaker.allow_request():
            return {
                "polish_applied": False,
                "fallback_reason": "circuit_open",
                "circuit_breaker_status": _polish_breaker.state(),
            }

        import sys
        import subprocess

        base_url = config.get("llm_base_url", os.getenv("V2_LLM_BASE_URL", "http://localhost:1234/v1"))
        model = config.get("llm_model", os.getenv("V2_LLM_MODEL", "local-model"))
        timeout_sec = int(config.get("polish_timeout_sec", POLISH_TIMEOUT_SEC))

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "clarityocr.polish",
            "--file",
            md_path,
            "--base-url",
            str(base_url),
            "--model",
            str(model),
        ]

        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)

        try:
            result = subprocess.run(
                cmd,
                cwd=str(Path.home()),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_sec,
                env=env,
            )
            if result.returncode == 0:
                _polish_breaker.record_success()
                payload = {
                    "polish_applied": True,
                    "circuit_breaker_status": _polish_breaker.state(),
                }
                with get_session() as session:
                    append_event(session, job_id, "file_polish_applied", file_id=file_id, payload=payload)
                return payload

            _polish_breaker.record_failure("llm_error")
            payload = {
                "polish_applied": False,
                "fallback_reason": "llm_error",
                "circuit_breaker_status": _polish_breaker.state(),
            }
            with get_session() as session:
                append_event(session, job_id, "file_polish_fallback", file_id=file_id, payload=payload)
            return payload
        except subprocess.TimeoutExpired:
            _polish_breaker.record_failure("timeout")
            payload = {
                "polish_applied": False,
                "fallback_reason": "timeout",
                "circuit_breaker_status": _polish_breaker.state(),
            }
            with get_session() as session:
                append_event(session, job_id, "file_polish_fallback", file_id=file_id, payload=payload)
            return payload
        except Exception as exc:
            _polish_breaker.record_failure("internal_error")
            payload = {
                "polish_applied": False,
                "fallback_reason": str(exc),
                "circuit_breaker_status": _polish_breaker.state(),
            }
            with get_session() as session:
                append_event(session, job_id, "file_polish_fallback", file_id=file_id, payload=payload)
            return payload

    def _post_file_finalize(self, job_id: str) -> None:
        with get_session() as session:
            check_and_update_job_completion(session, job_id)
        self._maybe_send_job_webhook(job_id)

    def _maybe_send_job_webhook(self, job_id: str) -> None:
        with get_session() as session:
            job = session.query(Job).filter_by(job_id=job_id).first()
            if not job or job.status not in FINAL_JOB_STATUSES:
                return

            already_sent = (
                session.query(JobEvent)
                .filter(JobEvent.job_id == job_id, JobEvent.event_code == "job_webhook_sent")
                .first()
            )
            if already_sent:
                return

            config = json.loads(job.config or "{}")
            webhook_url = config.get("callback_url")
            if not webhook_url:
                return
            secret = config.get("callback_secret") or ""

            payload = {
                "job_id": job.job_id,
                "status": job.status,
                "mode": job.mode,
                "accepted_at": job.accepted_at.isoformat() if job.accepted_at else None,
            }

        if send_webhook(webhook_url, secret, payload):
            with get_session() as session:
                append_event(session, job_id, "job_webhook_sent", payload={"callback_url": webhook_url})
        else:
            with get_session() as session:
                append_event(session, job_id, "job_webhook_failed", payload={"callback_url": webhook_url})

    def _process_file(self, file_id: str, job_id: str, input_path: str):
        start_time = time.time()
        logger.info(f"Worker {self.worker_id} processing file {file_id}: {input_path}")
        
        try:
            # 1. Fetch Job context to know mode and config
            with get_session() as session:
                job = session.query(Job).filter_by(job_id=job_id).first()
                mode = job.mode
                preset = job.preset
                config = json.loads(job.config or "{}")
                
            if mode in {"ocr_only", "ocr_plus_metadata", "ocr_plus_polish"}:
                # Output dir format: output_root/job_id/file_id
                output_dir = os.path.join("output_v2", job_id, file_id)
                os.makedirs(output_dir, exist_ok=True)
                
                # Execute converter (this calls marker-pdf etc.)
                # In actual implementation, we call it via subprocess to isolate VRAM and crash safety
                import sys
                import subprocess
                cmd = [
                    sys.executable, "-u", "-m", "clarityocr.converter", 
                    "--output-dir", output_dir, 
                    "--pdf", input_path
                ]
                # Map preset to batch parameters
                p_vals = {"speed": (2,4,4), "balanced": (4,8,8), "quality": (8,16,16)}.get(preset, (4,8,8))
                cmd += ["--layout-batch", str(p_vals[0]), "--recognition-batch", str(p_vals[1]), "--detection-batch", str(p_vals[2])]
                
                env = dict(os.environ)
                env["PYTHONIOENCODING"] = "utf-8"
                env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
                
                project_root = str(Path(__file__).resolve().parent.parent)
                result = subprocess.run(
                    cmd,
                    cwd=project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=600,
                    env=env,
                )

                # Prefer deterministic expected path, then fall back to discovered markdown.
                base_name = Path(input_path).stem
                md_path = os.path.join(output_dir, f"{base_name}.md")

                if result.returncode != 0:
                    logger.warning("Primary converter failed, fallback to simple converter: %s", result.stdout)
                    _fallback_markdown_from_pdf(input_path, md_path)

                if not os.path.exists(md_path):
                    discovered = sorted(Path(output_dir).glob("*.md"))
                    if discovered:
                        md_path = str(discovered[0])
                        base_name = Path(md_path).stem
                    else:
                        logger.warning(
                            "Primary converter produced no markdown. Fallback to simple converter for %s",
                            input_path,
                        )
                        _fallback_markdown_from_pdf(input_path, md_path)
                        if os.path.exists(md_path):
                            base_name = Path(md_path).stem
                        else:
                            raise FileNotFoundError(f"Markdown output not found at {md_path}")

                polish_result = self._try_optional_polish(md_path, mode, config, job_id, file_id)
                with open(md_path, "r", encoding="utf-8") as md_file:
                    md_text = md_file.read()
                
                # 2. Extract Metadata & Naming
                meta = generate_metadata(md_path)
                meta["polish_applied"] = bool(polish_result.get("polish_applied"))
                meta["fallback_reason"] = polish_result.get("fallback_reason")
                meta["circuit_breaker_status"] = polish_result.get("circuit_breaker_status")
                meta["confidence_source"] = "ocr-native+heuristic"
                naming = generate_naming(input_path, md_text, output_dir=output_dir)
                
                # 3. Save Artifacts
                meta_path = os.path.join(output_dir, f"{naming['filename_slug']}.meta.json")
                naming_path = os.path.join(output_dir, f"{base_name}.naming.json")
                
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, indent=2)
                    
                with open(naming_path, 'w', encoding='utf-8') as f:
                    json.dump(naming, f, indent=2)
                
                # Generate Manifest with checksums
                manifest_path = os.path.join(output_dir, "batch_manifest.json")
                _write_manifest(
                    manifest_path,
                    [
                        {"path": f"{base_name}.md", "type": "md", "sha256": _file_sha256(md_path), "status": "completed"},
                        {"path": Path(meta_path).name, "type": "meta", "sha256": _file_sha256(meta_path), "status": "completed"},
                        {"path": Path(naming_path).name, "type": "naming", "sha256": _file_sha256(naming_path), "status": "completed"},
                    ],
                )
                
                with get_session() as session:
                    record_artifact(session, job_id, file_id, "md", md_path)
                    record_artifact(session, job_id, file_id, "meta", meta_path)
                    record_artifact(session, job_id, file_id, "naming", naming_path)
                    record_artifact(session, job_id, file_id, "manifest", manifest_path)
                
                # 4. Mark success
                duration_ms = int((time.time() - start_time) * 1000)
                with get_session() as session:
                    transition_file_status(session, file_id, "completed", duration_ms=duration_ms)
                    append_event(session, job_id, "file_done", file_id=file_id, payload={"duration_ms": duration_ms})
                self._post_file_finalize(job_id)
                    
            elif "merge" in mode:
                # Handle Merge API
                output_dir = os.path.join("output_v2", job_id)
                os.makedirs(output_dir, exist_ok=True)

                inputs = config.get("inputs", [])
                order_by = config.get("naming_policy", "filename")
                
                from .merger import merge_pipeline
                report = merge_pipeline(inputs, output_dir, order_by)
                
                report_path = os.path.join(output_dir, "merge_report.json")
                out_pdf = os.path.join(output_dir, "merged.pdf")
                merge_manifest_path = os.path.join(output_dir, "batch_manifest.json")
                merge_manifest_files = [
                    {"path": "merge_report.json", "type": "merge_report", "sha256": _file_sha256(report_path), "status": "completed"},
                ]
                if os.path.exists(out_pdf):
                    merge_manifest_files.insert(0, {"path": "merged.pdf", "type": "merged_pdf", "sha256": _file_sha256(out_pdf), "status": "completed"})
                _write_manifest(merge_manifest_path, merge_manifest_files)
                
                with get_session() as session:
                    if os.path.exists(out_pdf):
                        record_artifact(session, job_id, file_id, "merged_pdf", out_pdf)
                    record_artifact(session, job_id, file_id, "merge_report", report_path)
                    record_artifact(session, job_id, file_id, "manifest", merge_manifest_path)
                
                if mode == "merge_then_ocr" and os.path.exists(out_pdf):
                    import sys
                    import subprocess
                    ocr_out_dir = os.path.join(output_dir, "ocr")
                    os.makedirs(ocr_out_dir, exist_ok=True)
                    cmd = [
                        sys.executable, "-u", "-m", "clarityocr.converter", 
                        "--output-dir", ocr_out_dir, 
                        "--pdf", out_pdf
                    ]
                    p_vals = {"speed": (2,4,4), "balanced": (4,8,8), "quality": (8,16,16)}.get(preset, (4,8,8))
                    cmd += ["--layout-batch", str(p_vals[0]), "--recognition-batch", str(p_vals[1]), "--detection-batch", str(p_vals[2])]
                    
                    env = dict(os.environ)
                    env["PYTHONIOENCODING"] = "utf-8"
                    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
                    
                    project_root = str(Path(__file__).resolve().parent.parent)
                    result = subprocess.run(
                        cmd,
                        cwd=project_root,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=1200,
                        env=env,
                    )

                    md_path = os.path.join(ocr_out_dir, "merged.md")
                    if result.returncode != 0:
                        logger.warning("Primary post-merge converter failed, fallback to simple converter: %s", result.stdout)
                        _fallback_markdown_from_pdf(out_pdf, md_path)
                    elif not os.path.exists(md_path):
                        logger.warning(
                            "Primary post-merge converter produced no markdown. Fallback to simple converter for %s",
                            out_pdf,
                        )
                        _fallback_markdown_from_pdf(out_pdf, md_path)

                    if os.path.exists(md_path):
                        md_text = open(md_path, 'r', encoding='utf-8').read()
                        meta = generate_metadata(md_path)
                        naming = generate_naming(out_pdf, md_text, output_dir=ocr_out_dir)
                        
                        meta_path = os.path.join(ocr_out_dir, "merged.meta.json")
                        naming_path = os.path.join(ocr_out_dir, "merged.naming.json")
                        
                        with open(meta_path, 'w', encoding='utf-8') as f: json.dump(meta, f, indent=2)
                        with open(naming_path, 'w', encoding='utf-8') as f: json.dump(naming, f, indent=2)

                        manifest_path = os.path.join(ocr_out_dir, "batch_manifest.json")
                        _write_manifest(
                            manifest_path,
                            [
                                {"path": "merged.md", "type": "md", "sha256": _file_sha256(md_path), "status": "completed"},
                                {"path": "merged.meta.json", "type": "meta", "sha256": _file_sha256(meta_path), "status": "completed"},
                                {"path": "merged.naming.json", "type": "naming", "sha256": _file_sha256(naming_path), "status": "completed"},
                            ],
                        )
                        
                        with get_session() as session:
                            record_artifact(session, job_id, file_id, "md", md_path)
                            record_artifact(session, job_id, file_id, "meta", meta_path)
                            record_artifact(session, job_id, file_id, "naming", naming_path)
                            record_artifact(session, job_id, file_id, "manifest", manifest_path)

                # Mark success
                duration_ms = int((time.time() - start_time) * 1000)
                with get_session() as session:
                    transition_file_status(session, file_id, "completed", duration_ms=duration_ms)
                    append_event(session, job_id, "file_done", file_id=file_id, payload={"duration_ms": duration_ms})
                self._post_file_finalize(job_id)

        except Exception as e:
            logger.error(f"Error processing file {file_id}: {traceback.format_exc()}")
            duration_ms = int((time.time() - start_time) * 1000)
            with get_session() as session:
                f_db = session.query(JobFile).filter_by(id=file_id).first()
                if not f_db:
                    append_event(session, job_id, "file_failed_final", file_id=file_id, payload={"error": str(e)})
                    self._post_file_finalize(job_id)
                    return

                next_attempt = int(f_db.attempt or 0) + 1
                if next_attempt < int(f_db.max_attempts or 1):
                    # Requeue for retry and keep job progressing.
                    transition_file_status(
                        session,
                        file_id,
                        "queued",
                        attempt=next_attempt,
                        last_error_message=str(e),
                        last_error_code="E_INTERNAL",
                        duration_ms=duration_ms,
                    )
                    append_event(
                        session,
                        job_id,
                        "file_requeued",
                        file_id=file_id,
                        payload={"error": str(e), "attempt": next_attempt},
                    )
                else:
                    transition_file_status(
                        session,
                        file_id,
                        "failed_final",
                        attempt=next_attempt,
                        last_error_message=str(e),
                        last_error_code="E_INTERNAL",
                        duration_ms=duration_ms,
                    )
                    append_event(
                        session,
                        job_id,
                        "file_failed_final",
                        file_id=file_id,
                        payload={"error": str(e), "attempt": next_attempt},
                    )
            self._post_file_finalize(job_id)


# =============================================================================
# WEBHOOK DELIVERY
# =============================================================================

def send_webhook(webhook_url: str, secret: str, payload: dict):
    """Send webhook with HMAC signature. Returns True on 2xx response."""
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    signature = hmac.new((secret or "").encode("utf-8"), body, hashlib.sha256).hexdigest()
    headers = {
        "Content-Type": "application/json",
        "X-ClarityOCR-Signature": f"sha256={signature}",
    }
    try:
        resp = httpx.post(webhook_url, content=body, headers=headers, timeout=10.0)
        return 200 <= resp.status_code < 300
    except Exception:
        return False


# Global worker pool for the FastAPI lifecycle
_workers: list[PipelineWorker] = []
_workers_lock = threading.Lock()


def start_workers(num_workers: int = 1):
    worker_count = max(1, int(num_workers))
    with _workers_lock:
        if _workers:
            return
        for i in range(worker_count):
            w = PipelineWorker(gpu_id=f"cuda:{i}" if i == 0 else "cpu")
            w.start()
            _workers.append(w)


def stop_workers():
    with _workers_lock:
        workers = list(_workers)
        _workers.clear()
    for w in workers:
        w.stop_event.set()
    for w in workers:
        w.join(timeout=5.0)
