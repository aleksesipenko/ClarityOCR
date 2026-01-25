#!/usr/bin/env python3
"""
ClarityOCR Web UI Server

A local web interface for:
- Selecting and converting PDF files to Markdown
- LLM-based post-processing (OCR error correction)
- Real-time progress monitoring with GPU stats

Runs on http://127.0.0.1:8008 by default.
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional, Callable

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# =============================================================================
# REQUEST MODELS (Pydantic validation for POST bodies)
# =============================================================================


class JobStartRequest(BaseModel):
    """Request body for /api/job/start endpoint."""

    files: list[str] = Field(..., min_length=1, description="List of PDF file paths to process")
    input_dir: str | None = Field(None, description="Input directory path")
    output_dir: str | None = Field(None, description="Output directory path")
    max_pages: int = Field(500, ge=1, le=10000, description="Maximum pages per PDF")
    auto_fallback: bool = Field(True, description="Enable automatic batch size reduction on OOM")


class LLMJobStartRequest(BaseModel):
    """Request body for /api/llm/job/start endpoint."""

    files: list[str] = Field(..., min_length=1, description="List of Markdown file paths to polish")
    output_dir: str | None = Field(None, description="Output directory path")
    base_url: str = Field("http://localhost:1234/v1", description="LM Studio server URL")
    model: str = Field("local-model", description="Model identifier")
    temperature: float = Field(0.1, ge=0, le=2, description="Generation temperature")
    chunk_size: int = Field(800, ge=100, le=4000, description="Chunk size in tokens")
    max_tokens: int = Field(4096, ge=256, le=32000, description="Max output tokens")
    dry_run: bool = Field(False, description="Preview mode without saving")


class MojibakeFixRequest(BaseModel):
    """Request body for /api/postprocess/fix-mojibake endpoint."""

    file: str = Field(..., description="Path to Markdown file")


class LegacyPolishRequest(BaseModel):
    """Request body for /api/postprocess/polish-llm endpoint (legacy)."""

    file: str = Field(..., description="Path to Markdown file")
    base_url: str = Field("http://localhost:1234/v1", description="LM Studio server URL")
    dry_run: bool = Field(False, description="Preview mode without saving")


APP_HOST = "127.0.0.1"
APP_PORT = 8008

# =============================================================================
# POLISHED FILE TRACKING
# =============================================================================

POLISHED_MARKER_SUFFIX = ".polished"  # Marker file to track polished status


def is_file_polished(md_path: Path) -> bool:
    """Check if a markdown file has been polished (has .polished marker or .bak exists)."""
    marker_path = md_path.with_suffix(md_path.suffix + POLISHED_MARKER_SUFFIX)
    backup_path = md_path.with_suffix(".md.bak")
    return marker_path.exists() or backup_path.exists()


def mark_file_polished(md_path: Path) -> None:
    """Create a marker file indicating the file has been polished."""
    marker_path = md_path.with_suffix(md_path.suffix + POLISHED_MARKER_SUFFIX)
    marker_path.write_text(f"Polished at {time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")


# =============================================================================
# ASYNC GPU POLLING (independent of subprocess)
# =============================================================================

_gpu_poll_thread: Optional[threading.Thread] = None
_gpu_poll_stop = threading.Event()
_gpu_poll_interval = 2.0  # Poll every 2 seconds


def gpu_poll_worker():
    """Background thread that continuously polls GPU stats via nvidia-smi."""
    while not _gpu_poll_stop.is_set():
        try:
            stats = get_gpu_stats_nvidia_smi()
            if stats.get("available"):
                update_gpu_stats_cache(stats)
                # Emit SSE event if job is running
                with job_lock:
                    if current_job is not None and current_job.proc.poll() is None:
                        q_put({"type": "gpu_stats", **stats})
        except Exception:
            pass
        _gpu_poll_stop.wait(_gpu_poll_interval)


def start_gpu_polling():
    """Start the GPU polling background thread."""
    global _gpu_poll_thread
    if _gpu_poll_thread is not None and _gpu_poll_thread.is_alive():
        return
    _gpu_poll_stop.clear()
    _gpu_poll_thread = threading.Thread(target=gpu_poll_worker, daemon=True)
    _gpu_poll_thread.start()


def stop_gpu_polling():
    """Stop the GPU polling background thread."""
    global _gpu_poll_thread
    _gpu_poll_stop.set()
    if _gpu_poll_thread is not None:
        _gpu_poll_thread.join(timeout=3)
    _gpu_poll_thread = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: start/stop GPU polling and cleanup running jobs."""
    start_gpu_polling()
    yield
    # Cleanup: stop any running jobs to prevent orphaned processes
    with job_lock:
        if current_job is not None and current_job.proc.poll() is None:
            stop_process_tree(current_job.proc.pid)
    with llm_job_lock:
        if current_llm_job is not None and current_llm_job.proc.poll() is None:
            stop_process_tree(current_llm_job.proc.pid)
    stop_gpu_polling()


def package_root() -> Path:
    """Get the clarityocr package root directory."""
    return Path(__file__).resolve().parent


def get_python_executable() -> Path:
    """Get the Python executable for subprocesses."""
    import sys

    return Path(sys.executable)


def converter_module() -> str:
    """Get the converter module for subprocess invocation."""
    return "clarityocr.converter"


def polish_module() -> str:
    """Get the polish module for subprocess invocation."""
    return "clarityocr.polish"


def default_sources_dir() -> Path:
    """Default directory for PDF sources (current working directory)."""
    return Path.cwd()


def default_output_dir() -> Path:
    """Default output directory for converted Markdown files."""
    return Path.cwd() / "output"


def static_dir() -> Path:
    """Get path to static web assets."""
    return package_root() / "web" / "static"


def output_has_page_markers(md_path: Path) -> bool:
    try:
        with md_path.open("r", encoding="utf-8") as f:
            for _ in range(1000):
                line = f.readline()
                if not line:
                    return False
                if line.startswith("[p:"):
                    return True
        return False
    except Exception:
        return False


def pdf_page_count(pdf_path: Path) -> int:
    try:
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(str(pdf_path))
        n = int(len(doc))
        doc.close()
        return n
    except Exception:
        return 0


def safe_listdir(path: Path) -> tuple[list[dict[str, Any]], Optional[str]]:
    """List directory contents. Returns (items, error_message)."""
    try:
        items: list[dict[str, Any]] = []
        for p in sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if p.name.startswith("."):
                continue
            items.append({"name": p.name, "path": str(p), "is_dir": p.is_dir()})
        return items, None
    except PermissionError:
        return [], "Access denied"
    except FileNotFoundError:
        return [], "Folder not found"
    except Exception as e:
        return [], f"Error: {str(e)[:100]}"


def is_path_safe(path: Path, allowed_roots: list[Path] | None = None) -> bool:
    """Check if a path is safe to access (prevents path traversal attacks).

    By default, allows access to:
    - Current working directory and subdirectories
    - User's home directory and subdirectories
    - Drive roots on Windows (to allow browsing)

    Returns True if path is allowed, False otherwise.
    """
    try:
        resolved = path.resolve()
    except (OSError, ValueError):
        return False  # Invalid path

    # Default allowed roots
    if allowed_roots is None:
        allowed_roots = [
            Path.cwd().resolve(),
            Path.home().resolve(),
        ]
        # On Windows, allow drive roots for browsing (C:\, D:\, etc.)
        import sys

        if sys.platform == "win32":
            # Allow any drive root
            if len(resolved.parts) >= 1 and len(resolved.parts[0]) == 3:  # e.g. "C:\"
                drive_root = Path(resolved.parts[0])
                allowed_roots.append(drive_root.resolve())

    # Block known sensitive directories
    sensitive_patterns = [
        "windows/system32",
        "windows\\system32",
        "program files",
        "programdata",
        "/etc",
        "/var",
        "/usr",
        "/bin",
        "/sbin",
        ".ssh",
        ".gnupg",
        ".aws",
        ".azure",
    ]
    resolved_str = str(resolved).lower()
    for pattern in sensitive_patterns:
        if pattern in resolved_str:
            return False

    # Check if path is under any allowed root
    for root in allowed_roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue

    # On Windows, allow browsing any drive root itself
    import sys

    if sys.platform == "win32":
        # Check if it's a drive root (e.g., C:\)
        if len(resolved.parts) == 1:
            return True

    return False


# =============================================================================
# PROGRESS PARSING
# =============================================================================

# GPU stats cache (updated by subprocess output)
_gpu_stats_cache: dict[str, Any] = {
    "gpu_util": 0,
    "vram_used": 0.0,
    "vram_total": 16.0,
    "gpu_temp": 0,
    "last_update": 0.0,
}
_gpu_stats_lock = threading.Lock()


def get_gpu_stats_nvidia_smi() -> dict[str, Any]:
    """Get GPU stats directly via nvidia-smi."""
    try:
        import sys

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 5:
                return {
                    "gpu_util": int(parts[0].strip()),
                    "vram_used": round(float(parts[1].strip()) / 1024, 2),  # MB to GB
                    "vram_total": round(float(parts[2].strip()) / 1024, 2),
                    "gpu_temp": int(parts[3].strip()),
                    "name": parts[4].strip(),
                    "available": True,
                }
    except Exception:
        pass
    return {
        "gpu_util": 0,
        "vram_used": 0,
        "vram_total": 0,
        "gpu_temp": 0,
        "name": "N/A",
        "available": False,
    }


def update_gpu_stats_cache(stats: dict[str, Any]) -> None:
    """Update GPU stats cache from subprocess output."""
    with _gpu_stats_lock:
        _gpu_stats_cache.update(stats)
        _gpu_stats_cache["last_update"] = time.time()


def get_cached_gpu_stats() -> dict[str, Any]:
    """Get cached GPU stats."""
    with _gpu_stats_lock:
        return dict(_gpu_stats_cache)


PROGRESS_PATTERNS = {
    "file_start": re.compile(r"\[(\d+)/(\d+)\]\s+(.+)"),
    "page_progress": re.compile(r"Progress:\s*(\d+)/(\d+)"),
    "file_done": re.compile(r"DONE:\s*(\d+)\s*pages?\s+in\s+([\d.]+)s"),
    "file_failed": re.compile(r"FAILED:|TIMEOUT:"),
    "complete": re.compile(r"^=+\s*\nCOMPLETE\s*$|^COMPLETE$", re.MULTILINE),
    "batch_size": re.compile(r"batch=(\d+)"),
    "speed": re.compile(r"Speed:\s*([\d.]+)\s*p/min"),
    "vram": re.compile(r"VRAM:\s*([\d.]+)GB"),
    "eta": re.compile(r"ETA:\s*([\d:]+)"),
    "gpu_stats": re.compile(r"^\[GPU_STATS\]\s*(.+)$"),
    "timeout_info": re.compile(r"Timeout:\s*(\d+)s"),
    "ocr_preview": re.compile(r"^\[OCR_PREVIEW\]\s*(.+)$"),
}


@dataclass
class JobProgress:
    file_index: int = 0
    total_files: int = 0
    current_file: str = ""
    pages_done: int = 0
    total_pages: int = 0
    file_pages: int = 0
    batch_size: int = 0
    speed: float = 0.0
    vram: float = 0.0
    eta: str = ""
    completed: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class Job:
    proc: subprocess.Popen[str]
    started_at: float
    files: list[str]
    input_dir: str
    output_dir: str
    progress: JobProgress = field(default_factory=JobProgress)


job_lock = threading.Lock()
progress_lock = threading.Lock()  # Protects progress dataclass from race conditions
current_job: Optional[Job] = None
log_q: "queue.Queue[dict[str, Any]]" = queue.Queue(maxsize=5000)


# =============================================================================
# SSE BROADCASTER (fixes competing consumers bug)
# =============================================================================


class SSEBroadcaster:
    """Broadcast SSE messages to multiple subscribers.

    Solves the "competing consumers" problem where multiple clients
    would steal messages from each other with a single shared queue.
    Each subscriber gets their own queue with all messages.
    """

    def __init__(self, max_queue_size: int = 1000):
        self._subscribers: dict[int, queue.Queue[dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._next_id = 0
        self._max_queue_size = max_queue_size

    def subscribe(self) -> tuple[int, "queue.Queue[dict[str, Any]]"]:
        """Subscribe to broadcasts. Returns (subscriber_id, queue)."""
        with self._lock:
            sub_id = self._next_id
            self._next_id += 1
            q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=self._max_queue_size)
            self._subscribers[sub_id] = q
            return sub_id, q

    def unsubscribe(self, sub_id: int) -> None:
        """Unsubscribe from broadcasts."""
        with self._lock:
            self._subscribers.pop(sub_id, None)

    def broadcast(self, msg: dict[str, Any]) -> None:
        """Send message to all subscribers."""
        with self._lock:
            dead_subs = []
            for sub_id, q in self._subscribers.items():
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    # Drop oldest message to make room
                    try:
                        q.get_nowait()
                        q.put_nowait(msg)
                    except (queue.Empty, queue.Full):
                        dead_subs.append(sub_id)
            # Clean up dead subscribers
            for sub_id in dead_subs:
                self._subscribers.pop(sub_id, None)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)


# Global broadcasters for OCR and LLM jobs
ocr_broadcaster = SSEBroadcaster()
llm_broadcaster = SSEBroadcaster()

# =============================================================================
# LLM POLISH JOB STATE
# =============================================================================


@dataclass
class LLMPolishProgress:
    """Progress tracking for LLM polish job."""

    file_index: int = 0
    total_files: int = 0
    current_file: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    chars_processed: int = 0
    chars_total: int = 0
    speed_chars_per_sec: float = 0.0
    eta: str = ""
    completed: bool = False
    errors: list[str] = field(default_factory=list)
    files_modified: int = 0


@dataclass
class LLMPolishJob:
    """LLM Polish job state."""

    proc: subprocess.Popen[str]
    started_at: float
    files: list[str]
    output_dir: str
    progress: LLMPolishProgress = field(default_factory=LLMPolishProgress)
    stop_requested: bool = False


llm_job_lock = threading.Lock()
llm_progress_lock = threading.Lock()  # Protects LLM progress dataclass from race conditions
current_llm_job: Optional[LLMPolishJob] = None
llm_log_q: "queue.Queue[dict[str, Any]]" = queue.Queue(maxsize=5000)


def llm_q_put(msg: dict[str, Any]) -> None:
    """Put message to LLM job queue AND broadcast to all SSE subscribers."""
    # Broadcast to all LLM SSE subscribers (new architecture)
    llm_broadcaster.broadcast(msg)

    # Also put to legacy queue for backward compatibility
    try:
        llm_log_q.put_nowait(msg)
    except queue.Full:
        try:
            _ = llm_log_q.get_nowait()
        except queue.Empty:
            pass
        try:
            llm_log_q.put_nowait(msg)
        except queue.Full:
            pass


# Progress patterns for LLM polish output
LLM_PROGRESS_PATTERNS = {
    "file_start": re.compile(r"^Processing:\s*(.+)$"),
    "chunk_progress": re.compile(r"^\s*Chunk\s+(\d+)/(\d+):\s*([\d.]+)s,\s*(\d+)->(\d+)\s*chars"),
    "chunk_diff": re.compile(r"^\[CHUNK_DIFF\]\s*(.+)$"),
    "file_done": re.compile(
        r"^\s*SAVED:\s*(\d+)->(\d+)\s*chars\s*\(([+-]?[\d.]+)%\)\s*in\s*([\d.]+)s"
    ),
    "file_skip": re.compile(r"^\s*(SKIP|NO CHANGES)"),
    "summary": re.compile(r"^\s*Files processed:\s*(\d+)"),
    "modified": re.compile(r"^\s*Files modified:\s*(\d+)"),
    "total_time": re.compile(r"^\s*Total time:\s*([\d.]+)s"),
    "chunks_count": re.compile(r"^\s*Chunks:\s*(\d+)"),
    "primary_lang": re.compile(r"^\s*Primary language:\s*(.+)$"),
    "warning": re.compile(r"^\s*WARNING:"),
    "error": re.compile(r"^\s*ERROR:"),
}


def parse_llm_progress_line(line: str, progress: LLMPolishProgress) -> Optional[dict[str, Any]]:
    """Parse LLM polish output line and update progress."""

    # File start: Processing: filename.md
    if m := LLM_PROGRESS_PATTERNS["file_start"].match(line):
        progress.current_file = m.group(1).strip()
        progress.chunk_index = 0
        progress.total_chunks = 0
        return {
            "type": "llm_file_start",
            "file": progress.current_file,
            **asdict(progress),
        }

    # Chunks count: Chunks: 15 (avg 3200 chars)
    if m := LLM_PROGRESS_PATTERNS["chunks_count"].search(line):
        progress.total_chunks = int(m.group(1))
        return {
            "type": "llm_chunks_info",
            "total_chunks": progress.total_chunks,
            **asdict(progress),
        }

    # Chunk progress: Chunk 1/15: 2.3s, 3200->3180 chars
    if m := LLM_PROGRESS_PATTERNS["chunk_progress"].match(line):
        progress.chunk_index = int(m.group(1))
        progress.total_chunks = int(m.group(2))
        chunk_time = float(m.group(3))
        chars_in = int(m.group(4))
        chars_out = int(m.group(5))
        progress.chars_processed += chars_in
        if chunk_time > 0:
            progress.speed_chars_per_sec = chars_in / chunk_time
        # Estimate ETA
        if progress.total_chunks > 0 and progress.chunk_index < progress.total_chunks:
            remaining_chunks = progress.total_chunks - progress.chunk_index
            avg_chars = (
                progress.chars_processed / progress.chunk_index
                if progress.chunk_index > 0
                else chars_in
            )
            remaining_chars = remaining_chunks * avg_chars
            if progress.speed_chars_per_sec > 0:
                eta_seconds = remaining_chars / progress.speed_chars_per_sec
                progress.eta = f"{int(eta_seconds // 60)}:{int(eta_seconds % 60):02d}"
        return {"type": "llm_chunk_done", **asdict(progress)}

    # Chunk diff: [CHUNK_DIFF] {"original": "...", "result": "..."}
    if m := LLM_PROGRESS_PATTERNS["chunk_diff"].match(line):
        try:
            diff_data = json.loads(m.group(1))
            return {
                "type": "llm_chunk_diff",
                "original": diff_data.get("original", ""),
                "result": diff_data.get("result", ""),
            }
        except json.JSONDecodeError:
            pass

    # File done: SAVED: 12000->11800 chars (-1.7%) in 45.2s
    if m := LLM_PROGRESS_PATTERNS["file_done"].search(line):
        progress.files_modified += 1
        progress.file_index += 1
        return {"type": "llm_file_done", "modified": True, **asdict(progress)}

    # File skip: SKIP or NO CHANGES
    if LLM_PROGRESS_PATTERNS["file_skip"].search(line):
        progress.file_index += 1
        return {"type": "llm_file_done", "modified": False, **asdict(progress)}

    # Summary: Files processed: 5
    if m := LLM_PROGRESS_PATTERNS["summary"].search(line):
        progress.completed = True
        return {
            "type": "llm_job_complete",
            "success": len(progress.errors) == 0,
            **asdict(progress),
        }

    # Error
    if LLM_PROGRESS_PATTERNS["error"].search(line):
        progress.errors.append(line)
        return {"type": "llm_error", "error": line}

    return None


def llm_reader_thread(proc: subprocess.Popen[str], job: LLMPolishJob) -> None:
    """Read LLM polish subprocess output and emit SSE events."""
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")

            # Parse progress (with lock to prevent race conditions)
            with llm_progress_lock:
                event = parse_llm_progress_line(line, job.progress)

            # Always send log line
            llm_q_put({"type": "llm_log", "line": line})

            # Send progress event if parsed
            if event:
                llm_q_put(event)

    except Exception as e:
        llm_q_put({"type": "llm_log", "line": f"[server] LLM reader error: {e}"})
    finally:
        returncode = proc.wait()
        with llm_progress_lock:
            was_completed = job.progress.completed
            if not was_completed:
                job.progress.completed = True
            progress_copy = asdict(job.progress)
        if not was_completed:
            llm_q_put(
                {
                    "type": "llm_job_complete",
                    "success": returncode == 0,
                    "returncode": returncode,
                    **progress_copy,
                }
            )


def q_put(msg: dict[str, Any]) -> None:
    """Put message to legacy queue AND broadcast to all SSE subscribers."""
    # Broadcast to all SSE subscribers (new architecture)
    ocr_broadcaster.broadcast(msg)

    # Also put to legacy queue for backward compatibility
    try:
        log_q.put_nowait(msg)
    except queue.Full:
        try:
            _ = log_q.get_nowait()
        except queue.Empty:
            pass
        try:
            log_q.put_nowait(msg)
        except queue.Full:
            pass


def parse_progress_line(line: str, progress: JobProgress) -> Optional[dict[str, Any]]:
    """Parse a line and update progress. Returns event dict if significant."""

    # GPU stats: [GPU_STATS] {"gpu_util": 85, "vram_used": 12.5, ...}
    if m := PROGRESS_PATTERNS["gpu_stats"].match(line):
        try:
            stats = json.loads(m.group(1))
            update_gpu_stats_cache(stats)
            return {"type": "gpu_stats", **stats}
        except json.JSONDecodeError:
            pass

    # OCR Preview: [OCR_PREVIEW] {"filename": "...", "preview": "...", "pages": N}
    if m := PROGRESS_PATTERNS["ocr_preview"].match(line):
        try:
            preview_data = json.loads(m.group(1))
            return {"type": "ocr_preview", **preview_data}
        except json.JSONDecodeError:
            pass

    # File start: [1/5] filename.pdf
    if m := PROGRESS_PATTERNS["file_start"].search(line):
        progress.file_index = int(m.group(1))
        progress.total_files = int(m.group(2))
        progress.current_file = m.group(3).strip()
        return {"type": "file_start", **asdict(progress)}

    # Page progress: Progress: 120/350
    if m := PROGRESS_PATTERNS["page_progress"].search(line):
        progress.pages_done = int(m.group(1))
        progress.total_pages = int(m.group(2))
        return {"type": "progress", **asdict(progress)}

    # File done: DONE: 45 pages in 12.3s
    if m := PROGRESS_PATTERNS["file_done"].search(line):
        progress.file_pages = int(m.group(1))
        # Extract additional metrics from the same line block
        if bm := PROGRESS_PATTERNS["batch_size"].search(line):
            progress.batch_size = int(bm.group(1))
        if sm := PROGRESS_PATTERNS["speed"].search(line):
            progress.speed = float(sm.group(1))
        if vm := PROGRESS_PATTERNS["vram"].search(line):
            progress.vram = float(vm.group(1))
        if em := PROGRESS_PATTERNS["eta"].search(line):
            progress.eta = em.group(1)
        return {"type": "file_done", **asdict(progress)}

    # File failed or timed out
    if PROGRESS_PATTERNS["file_failed"].search(line):
        progress.errors.append(progress.current_file)
        return {"type": "file_failed", "file": progress.current_file}

    # Job complete
    if PROGRESS_PATTERNS["complete"].search(line):
        progress.completed = True
        return {"type": "job_complete", "success": len(progress.errors) == 0}

    return None


def reader_thread(proc: subprocess.Popen[str], job: Job) -> None:
    """Read subprocess output, parse progress, emit typed events."""
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")

            # Try to parse for progress events (with lock to prevent race conditions)
            with progress_lock:
                event = parse_progress_line(line, job.progress)

            # GPU_STATS and OCR_PREVIEW lines go to UI panels only, not to terminal log
            if event and event.get("type") in ("gpu_stats", "ocr_preview"):
                q_put(event)  # Send to SSE for UI panels
                continue  # Don't add to terminal log

            # All other lines go to terminal log
            q_put({"type": "log", "line": line})

            if event:
                q_put(event)

    except Exception as e:
        q_put({"type": "log", "line": f"[server] log reader error: {e}"})
    finally:
        # Wait for process to complete
        returncode = proc.wait()

        # Emit job_complete if not already emitted
        with progress_lock:
            if not job.progress.completed:
                job.progress.completed = True
        q_put(
            {
                "type": "job_complete",
                "success": returncode == 0,
                "returncode": returncode,
            }
        )


def stop_process_tree(pid: int) -> None:
    """Stop a process and all its children (cross-platform)."""
    import sys

    try:
        if sys.platform == "win32":
            # Windows: use taskkill to terminate process tree
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            # POSIX (Linux/macOS): try psutil first, fallback to kill
            try:
                import psutil

                parent = psutil.Process(pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
            except ImportError:
                # Fallback if psutil not available: just kill the main process
                import signal
                import os

                try:
                    os.kill(pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
    except Exception:
        pass


app = FastAPI(lifespan=lifespan)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust for production if needed)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_static_path = static_dir()  # Resolve once at startup
app.mount("/static", StaticFiles(directory=str(_static_path)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(_static_path / "index.html"))


@app.get("/llm-polish")
def llm_polish_page() -> FileResponse:
    """Serve the LLM Polish page."""
    return FileResponse(str(_static_path / "llm-polish.html"))


@app.get("/api/browse")
def api_browse(path: Optional[str] = None) -> JSONResponse:
    # Handle empty or whitespace path
    if path is not None:
        path = path.strip()
        if not path:
            path = None

    base = Path(path) if path else default_sources_dir()

    # Try to resolve the path to handle encoding issues
    try:
        base = base.resolve()
    except (OSError, ValueError):
        pass  # Keep original if resolve fails

    # Security: validate path to prevent traversal attacks
    if not is_path_safe(base):
        return JSONResponse(
            {
                "path": str(base),
                "parent": None,
                "items": [],
                "error": "Access denied: path not allowed",
            },
            status_code=403,
        )

    if not base.exists() or not base.is_dir():
        # More informative error for debugging
        if not base.exists():
            err_detail = f"Path does not exist: {base}"
        else:
            err_detail = f"Not a directory (is file): {base}"
        return JSONResponse(
            {
                "path": str(base),
                "parent": str(base.parent) if base.parent != base else None,
                "items": [],
                "error": err_detail,
            },
            status_code=400,
        )

    items, error = safe_listdir(base)
    parent = str(base.parent) if base.parent != base else None

    return JSONResponse({"path": str(base), "parent": parent, "items": items, "error": error})


@app.get("/api/scan")
def api_scan(
    dir: Optional[str] = None, output_dir: Optional[str] = None, max_pages: int = 500
) -> JSONResponse:
    input_dir = Path(dir) if dir else default_sources_dir()
    out_dir = Path(output_dir) if output_dir else default_output_dir()

    # Security: validate paths to prevent traversal attacks
    if not is_path_safe(input_dir):
        return JSONResponse(
            {"error": "Access denied: input path not allowed", "items": []}, status_code=403
        )
    if not is_path_safe(out_dir):
        return JSONResponse(
            {"error": "Access denied: output path not allowed", "items": []}, status_code=403
        )

    if not input_dir.exists() or not input_dir.is_dir():
        return JSONResponse(
            {"error": f"Not a directory: {input_dir}", "items": []}, status_code=400
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(input_dir.glob("*.pdf"))
    items: list[dict[str, Any]] = []
    total_pages = 0
    pending_count = 0

    for pdf in pdfs:
        pages = pdf_page_count(pdf)
        md_path = out_dir / f"{pdf.stem}.md"
        done = bool(md_path.exists() and output_has_page_markers(md_path))
        polished = bool(done and is_file_polished(md_path))
        too_long = bool(pages and pages > max_pages)

        items.append(
            {
                "name": pdf.name,
                "path": str(pdf.resolve()),
                "pages": pages,
                "done": done,
                "polished": polished,
                "too_long": too_long,
            }
        )

        total_pages += pages
        if not done and not too_long:
            pending_count += 1

    return JSONResponse(
        {
            "input_dir": str(input_dir),
            "output_dir": str(out_dir),
            "count": len(items),
            "total_pages": total_pages,
            "pending_count": pending_count,
            "items": items,
        }
    )


@app.get("/api/job/status")
def api_job_status() -> JSONResponse:
    with job_lock:
        if current_job is None:
            return JSONResponse({"running": False})
        running = current_job.proc.poll() is None
        # Use progress_lock to safely copy progress fields
        with progress_lock:
            progress_copy = asdict(current_job.progress)
        return JSONResponse(
            {
                "running": running,
                "pid": current_job.proc.pid,
                "started_at": current_job.started_at,
                "files": current_job.files,
                "input_dir": current_job.input_dir,
                "output_dir": current_job.output_dir,
                "progress": progress_copy,
            }
        )


@app.post("/api/job/start")
def api_job_start(payload: JobStartRequest) -> JSONResponse:
    input_dir = str(payload.input_dir or default_sources_dir())
    output_dir = str(payload.output_dir or default_output_dir())
    max_pages = payload.max_pages
    files = payload.files
    auto_fallback = payload.auto_fallback

    py = get_python_executable()

    with job_lock:
        global current_job
        if current_job is not None and current_job.proc.poll() is None:
            raise HTTPException(status_code=409, detail="Job already running")

        # Run converter as module: python -m clarityocr.converter
        cmd = [
            str(py),
            "-u",
            "-m",
            converter_module(),
            "--output-dir",
            output_dir,
            "--max-pages",
            str(max_pages),
        ]
        # Add auto-fallback control if specified
        if not auto_fallback:
            cmd += ["--no-auto-fallback"]
        for f in files:
            cmd += ["--pdf", str(f)]

        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"

        q_put({"type": "log", "line": "[server] starting OCR job..."})
        q_put(
            {
                "type": "log",
                "line": "[server] " + " ".join(json.dumps(c) if " " in c else c for c in cmd),
            }
        )

        proc = subprocess.Popen(
            cmd,
            cwd=str(Path.cwd()),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

        job = Job(
            proc=proc,
            started_at=time.time(),
            files=[str(f) for f in files],
            input_dir=input_dir,
            output_dir=output_dir,
            progress=JobProgress(total_files=len(files)),
        )
        current_job = job

        t = threading.Thread(target=reader_thread, args=(proc, job), daemon=True)
        t.start()

    return JSONResponse({"ok": True, "pid": proc.pid, "file_count": len(files)})


@app.post("/api/job/stop")
def api_job_stop() -> JSONResponse:
    with job_lock:
        global current_job
        if current_job is None:
            return JSONResponse({"ok": True, "stopped": False})
        pid = current_job.proc.pid
        running = current_job.proc.poll() is None
        if running:
            q_put({"type": "log", "line": "[server] stopping OCR job..."})
            q_put({"type": "job_stopped", "reason": "user"})
            stop_process_tree(pid)
        current_job = None
    return JSONResponse({"ok": True, "stopped": running, "pid": pid})


@app.get("/api/job/stream")
def api_job_stream() -> StreamingResponse:
    def gen():
        # Subscribe to the OCR broadcaster - each client gets their own queue
        sub_id, client_q = ocr_broadcaster.subscribe()
        try:
            yield "event: ready\ndata: {}\n\n"
            last_heartbeat = time.time()

            while True:
                try:
                    msg = client_q.get(timeout=0.5)
                    payload = json.dumps(msg, ensure_ascii=False)
                    event_type = msg.get("type", "log")
                    yield f"event: {event_type}\ndata: {payload}\n\n"
                except queue.Empty:
                    pass

                now = time.time()
                if now - last_heartbeat > 10:
                    yield "event: ping\ndata: {}\n\n"
                    last_heartbeat = now
        finally:
            # Clean up subscription when client disconnects
            ocr_broadcaster.unsubscribe(sub_id)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/vram")
def api_vram() -> JSONResponse:
    """Get current VRAM usage via nvidia-smi (works without loading torch).

    Falls back to cached stats from subprocess if available.
    """
    # Try nvidia-smi first (doesn't require torch)
    stats = get_gpu_stats_nvidia_smi()
    if stats.get("available"):
        return JSONResponse(stats)

    # Fall back to cached stats from subprocess
    cached = get_cached_gpu_stats()
    if cached.get("last_update", 0) > 0:
        stats = {
            "gpu_util": cached.get("gpu_util", 0),
            "vram_used": cached.get("vram_used", 0),
            "vram_total": cached.get("vram_total", 16),
            "gpu_temp": cached.get("gpu_temp", 0),
            "name": cached.get("name", "NVIDIA GPU"),
            "available": True,
            "cached": True,
        }
        return JSONResponse(stats)

    return JSONResponse({"used": 0, "total": 0, "name": "N/A", "available": False})


@app.get("/api/config")
def api_config() -> JSONResponse:
    """Return current fixed batch configuration (no presets)."""
    return JSONResponse(
        {
            "layout_batch": 4,
            "recognition_batch": 8,
            "detection_batch": 8,
            "description": "Fixed reliable configuration - always works without OOM",
        }
    )


# =============================================================================
# POST-PROCESSING ENDPOINTS (LLM Polish)
# =============================================================================


@app.post("/api/postprocess/fix-mojibake")
def api_fix_mojibake(payload: MojibakeFixRequest) -> JSONResponse:
    """Fix mojibake in markdown files (encoding errors).

    Note: This functionality is now built into the converter.
    This endpoint is kept for backwards compatibility.
    """
    return JSONResponse(
        {
            "ok": True,
            "message": "Mojibake fixing is now integrated into the converter. Files are automatically fixed during conversion.",
        }
    )


@app.get("/api/postprocess/llm-status")
def api_llm_status() -> JSONResponse:
    """Check if LM Studio is running and accessible."""
    try:
        req = urllib.request.Request(
            "http://localhost:1234/v1/models",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            models = [m.get("id", "unknown") for m in data.get("data", [])]
            return JSONResponse(
                {
                    "available": True,
                    "models": models,
                    "url": "http://localhost:1234",
                }
            )
    except urllib.error.URLError:
        return JSONResponse({"available": False, "error": "LM Studio not running"})
    except Exception as e:
        return JSONResponse({"available": False, "error": str(e)})


@app.get("/api/llm/job/status")
def api_llm_job_status() -> JSONResponse:
    """Get current LLM polish job status."""
    with llm_job_lock:
        if current_llm_job is None:
            return JSONResponse({"running": False})
        running = current_llm_job.proc.poll() is None
        return JSONResponse(
            {
                "running": running,
                "pid": current_llm_job.proc.pid,
                "started_at": current_llm_job.started_at,
                "files": current_llm_job.files,
                "progress": asdict(current_llm_job.progress),
            }
        )


@app.post("/api/llm/job/start")
def api_llm_job_start(payload: LLMJobStartRequest) -> JSONResponse:
    """Start LLM polish job for specified files."""
    files = payload.files
    base_url = payload.base_url
    dry_run = payload.dry_run

    py = get_python_executable()

    with llm_job_lock:
        global current_llm_job
        if current_llm_job is not None and current_llm_job.proc.poll() is None:
            raise HTTPException(status_code=409, detail="LLM job already running")

        # Run polish as module: python -m clarityocr.polish
        cmd = [
            str(py),
            "-u",
            "-m",
            polish_module(),
            "--base-url",
            base_url,
        ]
        if dry_run:
            cmd.append("--dry-run")

        # Add all files
        for f in files:
            cmd.extend(["--file", str(f)])

        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"

        llm_q_put({"type": "llm_log", "line": "[server] Starting LLM polish job..."})
        llm_q_put({"type": "llm_log", "line": f"[server] Processing {len(files)} files"})

        proc = subprocess.Popen(
            cmd,
            cwd=str(Path.cwd()),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

        job = LLMPolishJob(
            proc=proc,
            started_at=time.time(),
            files=[str(f) for f in files],
            output_dir=str(default_output_dir()),
            progress=LLMPolishProgress(total_files=len(files)),
        )
        current_llm_job = job

        # Start reader thread
        t = threading.Thread(target=llm_reader_thread, args=(proc, job), daemon=True)
        t.start()

    return JSONResponse({"ok": True, "pid": proc.pid, "file_count": len(files)})


@app.post("/api/llm/job/stop")
def api_llm_job_stop() -> JSONResponse:
    """Stop the current LLM polish job."""
    with llm_job_lock:
        global current_llm_job
        if current_llm_job is None:
            return JSONResponse({"ok": True, "stopped": False})

        pid = current_llm_job.proc.pid
        running = current_llm_job.proc.poll() is None

        if running:
            llm_q_put({"type": "llm_log", "line": "[server] Stopping LLM polish job..."})
            llm_q_put({"type": "llm_job_stopped", "reason": "user"})
            current_llm_job.stop_requested = True
            stop_process_tree(pid)

        current_llm_job = None

    return JSONResponse({"ok": True, "stopped": running, "pid": pid})


@app.get("/api/llm/job/stream")
def api_llm_job_stream() -> StreamingResponse:
    """SSE stream for LLM polish job progress."""

    def gen():
        # Subscribe to the LLM broadcaster - each client gets their own queue
        sub_id, client_q = llm_broadcaster.subscribe()
        try:
            yield "event: ready\ndata: {}\n\n"
            last_heartbeat = time.time()

            while True:
                try:
                    msg = client_q.get(timeout=0.5)
                    payload = json.dumps(msg, ensure_ascii=False)
                    event_type = msg.get("type", "llm_log")
                    yield f"event: {event_type}\ndata: {payload}\n\n"
                except queue.Empty:
                    pass

                now = time.time()
                if now - last_heartbeat > 10:
                    yield "event: ping\ndata: {}\n\n"
                    last_heartbeat = now
        finally:
            # Clean up subscription when client disconnects
            llm_broadcaster.unsubscribe(sub_id)

    return StreamingResponse(gen(), media_type="text/event-stream")


# Keep old endpoint for backwards compatibility but use new module invocation
@app.post("/api/postprocess/polish-llm")
def api_polish_llm(payload: LegacyPolishRequest) -> JSONResponse:
    """Run polish on specified file (legacy endpoint)."""
    file_path = payload.file
    dry_run = payload.dry_run
    base_url = payload.base_url

    py = get_python_executable()

    cmd = [str(py), "-m", polish_module(), "--file", str(file_path), "--base-url", base_url]
    if dry_run:
        cmd.append("--dry-run")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes for LLM processing
            cwd=str(Path.cwd()),
        )
        return JSONResponse(
            {
                "ok": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None,
            }
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="LLM processing timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/files/list")
def api_files_list(dir: Optional[str] = None) -> JSONResponse:
    """List converted markdown files with stats."""
    out_dir = Path(dir) if dir else default_output_dir()

    if not out_dir.exists():
        return JSONResponse({"files": [], "error": "Directory not found"})

    files = []
    for md_path in sorted(out_dir.glob("*.md")):
        try:
            stat = md_path.stat()
            has_markers = output_has_page_markers(md_path)
            polished = is_file_polished(md_path)
            files.append(
                {
                    "name": md_path.name,
                    "path": str(md_path.resolve()),
                    "size_kb": round(stat.st_size / 1024, 1),
                    "modified": stat.st_mtime,
                    "has_page_markers": has_markers,
                    "done": has_markers,
                    "polished": polished,
                }
            )
        except Exception:
            continue

    return JSONResponse({"dir": str(out_dir), "count": len(files), "files": files})


def run_server(host: str = "127.0.0.1", port: int = 8008) -> None:
    """Run the ClarityOCR web server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def main() -> None:
    """Entry point for clarityocr-server command."""
    run_server(host=APP_HOST, port=APP_PORT)


if __name__ == "__main__":
    main()
