#!/usr/bin/env python3
"""
PDF to Markdown Converter v3 - ROBUST Edition
Optimized for RTX 4070 Ti SUPER (16GB VRAM)

Features:
- Adaptive timeout with hang prevention
- VRAM pressure monitoring and auto-reduction
- Real-time GPU stats emission for Web UI
- Automatic recovery from CUDA errors
- File logging for diagnostics
"""

import os
import sys
import time
import gc
import re
import json
import argparse
import subprocess
import threading
import logging
from pathlib import Path
import warnings
from typing import Optional, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import multiprocessing as mp
import queue
import traceback
import io

warnings.filterwarnings("ignore")

# =============================================================================
# LOGGING SETUP (file-only, doesn't pollute terminal)
# =============================================================================


# Use user's home directory for logs, not package directory
def _get_log_dir() -> Path:
    """Get log directory in user's home folder."""
    log_dir = Path.home() / ".clarityocr" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


LOG_DIR = _get_log_dir()

# Create timestamped log file
_log_filename = LOG_DIR / f"ocr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# File handler - detailed debug info
_file_handler = logging.FileHandler(_log_filename, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
)

# Create logger (no console handlers!)
logger = logging.getLogger("ocr")
logger.setLevel(logging.DEBUG)
logger.addHandler(_file_handler)
logger.propagate = False  # Don't propagate to root logger


def log_debug(msg: str) -> None:
    """Log to file only (not terminal)."""
    logger.debug(msg)


def log_info(msg: str) -> None:
    """Log to file only."""
    logger.info(msg)


def log_warning(msg: str) -> None:
    """Log to file only."""
    logger.warning(msg)


def log_error(msg: str) -> None:
    """Log to file only."""
    logger.error(msg)


def log_gpu_stats(stats: Dict[str, Any]) -> None:
    """Log GPU stats to file (not terminal)."""
    logger.debug(
        f"GPU: util={stats.get('gpu_util', -1)}% "
        f"vram={stats.get('vram_used', 0):.2f}/{stats.get('vram_total', 0):.0f}GB "
        f"temp={stats.get('gpu_temp', -1)}°C"
    )


# =============================================================================
# CUDA ENVIRONMENT SETUP (MUST be before torch import)
# =============================================================================
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:512,expandable_segments:True,garbage_collection_threshold:0.8",
)
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")  # Faster startup

if sys.platform == "win32":
    _reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(_reconfigure):
        _reconfigure(encoding="utf-8")

import torch

# =============================================================================
# CUDA PERFORMANCE & STABILITY TOGGLES
# =============================================================================
try:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # Faster, less reproducible
except Exception:
    pass

try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ============================================================================
# CONFIG
# ============================================================================


# Dynamic VRAM detection with fallback
def _detect_vram() -> float:
    """Detect available VRAM in GB."""
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        pass
    return 16.0  # Fallback to 16GB


TOTAL_VRAM_GB = _detect_vram()
RESERVED_VRAM_GB = 2.0
MAX_PAGES = 500  # Skip PDFs longer than this

# Use current working directory as default, not hardcoded paths
DEFAULT_INPUT_DIR = Path.cwd()
DEFAULT_OUTPUT_DIR = Path.cwd() / "output"

# Don't create output dir at import time - do it lazily when needed
# DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# ROBUST TIMEOUT & VRAM MANAGEMENT
# ============================================================================

# Timeout settings - DISABLED BY DEFAULT (use --timeout to enable)
# When enabled, timeout = BASE + pages * PER_PAGE
BASE_TIMEOUT_S = 300  # 5 minutes minimum per PDF
PER_PAGE_TIMEOUT_S = 10.0  # 10s per page (very generous)
MAX_TIMEOUT_S = 0  # 0 = NO TIMEOUT (disabled by default)

# VRAM pressure thresholds - STRICT
VRAM_PRESSURE_THRESHOLD = 0.80  # Start reducing at 80% (not 85%)
VRAM_CRITICAL_THRESHOLD = 0.90  # Force minimum at 90% (not 95%)
VRAM_OVERFLOW_THRESHOLD = 0.98  # Above this = treat as OOM (shared memory in use)
VRAM_REDUCTION_FACTOR = 0.5  # Reduce by 50% (not 25%)
MIN_BATCH_SIZE = 2  # Never go below this

# Stats emission - continuous in background
STATS_EMIT_INTERVAL_S = 1.0  # Emit every 1 second (not 2)

# Watchdog settings
WATCHDOG_CHECK_INTERVAL_S = 5.0  # Check VRAM every 5 seconds during conversion
STALL_DETECTION_TIMEOUT_S = 30.0  # If no progress for 30s, consider stalled

# Stats emission state
_LAST_STATS_EMIT_TIME = 0.0
_stats_thread: Optional[threading.Thread] = None
_stats_stop = threading.Event()
_print_lock = threading.Lock()  # Prevent JSON output interleaving between threads


def calculate_timeout(page_count: int) -> Optional[int]:
    """Calculate adaptive timeout based on page count. Returns None if timeout disabled."""
    if MAX_TIMEOUT_S == 0:
        return None  # Timeout disabled
    timeout = int(BASE_TIMEOUT_S + page_count * PER_PAGE_TIMEOUT_S)
    return min(MAX_TIMEOUT_S, max(BASE_TIMEOUT_S, timeout))


def get_gpu_stats() -> Dict[str, Any]:
    """Get GPU stats via nvidia-smi (works outside of torch context)."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 4:
                return {
                    "gpu_util": int(parts[0].strip()),
                    "vram_used": float(parts[1].strip()) / 1024,  # MB to GB
                    "vram_total": float(parts[2].strip()) / 1024,
                    "gpu_temp": int(parts[3].strip()),
                }
    except Exception:
        pass

    # Fallback to torch if nvidia-smi fails
    if torch.cuda.is_available():
        try:
            vram_used = torch.cuda.memory_reserved() / 1024**3
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return {
                "gpu_util": -1,  # Unknown without nvidia-smi
                "vram_used": round(vram_used, 2),
                "vram_total": round(vram_total, 2),
                "gpu_temp": -1,
            }
        except Exception:
            pass

    return {"gpu_util": 0, "vram_used": 0, "vram_total": 0, "gpu_temp": 0}


def get_vram_pressure() -> float:
    """Get VRAM pressure as ratio (0.0 to 1.0)."""
    stats = get_gpu_stats()
    if stats["vram_total"] > 0:
        return stats["vram_used"] / stats["vram_total"]
    return 0.0


def is_vram_overflow() -> bool:
    """Check if VRAM has overflowed into shared memory (effective OOM).

    When PyTorch runs out of dedicated VRAM, it silently uses shared memory
    (system RAM over PCIe) which causes massive slowdowns but no OOM error.
    """
    pressure = get_vram_pressure()
    if pressure >= VRAM_OVERFLOW_THRESHOLD:
        log_warning(f"VRAM OVERFLOW detected: {pressure * 100:.0f}%")
        print(f"  [vram] OVERFLOW detected: {pressure * 100:.0f}% - treating as OOM")
        return True
    return False


def start_stats_thread():
    """Start background thread for continuous GPU stats emission."""
    global _stats_thread
    if _stats_thread is not None and _stats_thread.is_alive():
        return

    _stats_stop.clear()

    def stats_worker():
        while not _stats_stop.is_set():
            try:
                stats = get_gpu_stats()
                # Emit JSON for Web UI (will be filtered from terminal by server.py)
                with _print_lock:
                    print(f"[GPU_STATS] {json.dumps(stats)}", flush=True)
                # Also log to file for diagnostics
                log_gpu_stats(stats)
            except Exception as e:
                log_error(f"Stats worker error: {e}")
            _stats_stop.wait(STATS_EMIT_INTERVAL_S)

    _stats_thread = threading.Thread(target=stats_worker, daemon=True)
    _stats_thread.start()
    log_info(f"Stats thread started, logging to {_log_filename}")


def stop_stats_thread():
    """Stop the background stats thread."""
    global _stats_thread
    _stats_stop.set()
    if _stats_thread is not None:
        _stats_thread.join(timeout=2)
    _stats_thread = None


def emit_ocr_preview(filename: str, text: str, page_count: int) -> None:
    """Emit OCR text preview for Web UI consumption.

    Output format: [OCR_PREVIEW] {"filename": "...", "preview": "...", "pages": N}
    The server parses this and sends to frontend via SSE.
    """
    # Take first ~1500 chars for preview, but cut at word boundary
    max_preview = 1500
    if len(text) <= max_preview:
        preview = text
    else:
        # Find last space before max_preview to avoid cutting mid-word
        cut_pos = text[:max_preview].rfind(" ")
        if cut_pos < max_preview // 2:
            cut_pos = max_preview
        preview = text[:cut_pos] + "..."

    # Escape newlines for JSON
    preview_escaped = preview.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "")

    # Emit as JSON line for server to parse
    data = {
        "filename": filename,
        "preview": preview_escaped,
        "pages": page_count,
        "total_chars": len(text),
    }
    with _print_lock:
        print(f"[OCR_PREVIEW] {json.dumps(data, ensure_ascii=False)}", flush=True)


class VRAMWatchdog:
    """Watchdog that monitors VRAM during conversion and triggers early abort."""

    def __init__(self, check_interval: float = WATCHDOG_CHECK_INTERVAL_S):
        self.check_interval = check_interval
        self.should_abort = threading.Event()
        self.abort_reason = ""
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last_check_time = 0.0

    def start(self):
        """Start watchdog monitoring."""
        self._stop.clear()
        self.should_abort.clear()
        self.abort_reason = ""

        def watchdog_worker():
            consecutive_overflows = 0
            while not self._stop.is_set():
                try:
                    pressure = get_vram_pressure()

                    # Check for overflow (>98% = shared memory in use)
                    if pressure >= VRAM_OVERFLOW_THRESHOLD:
                        consecutive_overflows += 1
                        log_warning(
                            f"Watchdog: VRAM overflow {consecutive_overflows}/2 ({pressure * 100:.0f}%)"
                        )
                        if consecutive_overflows >= 2:  # 2 consecutive checks
                            self.abort_reason = f"VRAM overflow ({pressure * 100:.0f}%)"
                            self.should_abort.set()
                            log_error(f"Watchdog ABORT: {self.abort_reason}")
                            print(
                                f"\n  [watchdog] ABORTING: {self.abort_reason}",
                                flush=True,
                            )
                            break
                    else:
                        consecutive_overflows = 0

                    # Check for critical pressure (log only, don't print to terminal)
                    if pressure >= VRAM_CRITICAL_THRESHOLD:
                        log_warning(f"Watchdog: CRITICAL VRAM {pressure * 100:.0f}%")

                except Exception as e:
                    log_error(f"Watchdog error: {e}")

                self._stop.wait(self.check_interval)

        self._thread = threading.Thread(target=watchdog_worker, daemon=True)
        self._thread.start()
        log_debug("VRAM watchdog started")

    def stop(self):
        """Stop watchdog monitoring."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self._thread = None

    def check(self) -> bool:
        """Check if abort was triggered. Returns True if should abort."""
        return self.should_abort.is_set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
        return False


def cuda_health_check() -> bool:
    """Check if CUDA is healthy and responsive."""
    if not torch.cuda.is_available():
        return False
    try:
        # Quick tensor allocation and sync
        t = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        del t
        return True
    except Exception as e:
        print(f"  [cuda] Health check failed: {e}")
        return False


def cuda_full_cleanup() -> None:
    """Aggressive CUDA cleanup."""
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    gc.collect()


class ConversionTimeoutError(Exception):
    """Raised when PDF conversion times out."""

    pass


class ConversionTimeoutHandler:
    """Windows-compatible timeout handler using threading.Timer."""

    def __init__(self, timeout_seconds: int):
        self.timeout = timeout_seconds
        self.timer: Optional[threading.Timer] = None
        self.timed_out = False
        self._lock = threading.Lock()

    def _timeout_callback(self):
        with self._lock:
            self.timed_out = True
        print(f"\n  [timeout] Operation exceeded {self.timeout}s limit!", flush=True)

    def __enter__(self):
        self.timed_out = False
        self.timer = threading.Timer(self.timeout, self._timeout_callback)
        self.timer.daemon = True
        self.timer.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timer:
            self.timer.cancel()
        return False

    def check(self) -> bool:
        """Check if timeout occurred. Call periodically during long operations."""
        with self._lock:
            return self.timed_out


# =============================================================================
# MULTIPROCESSING-BASED CONVERSION WITH HARD TIMEOUT
# =============================================================================
# Python threads cannot interrupt blocking CUDA operations.
# Using multiprocessing.Process allows us to terminate() hung conversions.


def _serialize_image(img: Any) -> bytes:
    """Serialize PIL Image or bytes to bytes for cross-process transfer."""
    if isinstance(img, bytes):
        return img
    if hasattr(img, "save"):
        # PIL Image
        buf = io.BytesIO()
        # Determine format from image mode or default to PNG
        fmt = "PNG" if img.mode in ("RGBA", "LA", "PA") else "JPEG"
        img.save(buf, format=fmt)
        return buf.getvalue()
    raise TypeError(f"Cannot serialize image of type {type(img)}")


def _deserialize_image(data: bytes, name: str) -> bytes:
    """Deserialize image bytes. Returns bytes directly for saving to disk."""
    return data  # Keep as bytes - will be saved to disk


def _conversion_worker(
    pdf_path: str,
    layout_bs: int,
    recognition_bs: int,
    detection_bs: int,
    output_queue: "mp.Queue[Dict[str, Any]]",
) -> None:
    """
    Worker function running in subprocess.

    Loads models fresh (required for clean CUDA context in spawn'd process),
    converts the PDF, and puts results on the queue.
    """
    try:
        # Import inside worker to avoid serialization issues with CUDA tensors
        import torch
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered

        # Configure CUDA in worker
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True

        # Load models in this process
        models = create_model_dict()
        config = build_marker_config(layout_bs, recognition_bs, detection_bs)
        converter = PdfConverter(config=config, artifact_dict=models)

        # Convert
        rendered = converter(pdf_path)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        text, metadata, images = text_from_rendered(rendered)

        # Get actual page count
        actual_pages = len(rendered.children) if hasattr(rendered, "children") else 1

        # Serialize images for cross-process transfer
        serialized_images: Dict[str, bytes] = {}
        for img_name, img_data in images.items():
            try:
                serialized_images[img_name] = _serialize_image(img_data)
            except Exception as e:
                log_warning(f"Failed to serialize image {img_name}: {e}")

        output_queue.put(
            {
                "success": True,
                "text": text,
                "images": serialized_images,
                "actual_pages": actual_pages,
            }
        )

    except Exception as e:
        output_queue.put(
            {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        # Clean up GPU in worker before exit
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def convert_pdf_with_timeout(
    pdf_path: Path,
    layout_bs: int,
    recognition_bs: int,
    detection_bs: int,
    timeout_s: Optional[int],
    queue_timeout: float = 15.0,
) -> Tuple[str, Dict[str, bytes], int]:
    """
    Convert PDF with optional timeout enforcement using multiprocessing.

    Args:
        pdf_path: Path to PDF file
        layout_bs, recognition_bs, detection_bs: Batch sizes for marker
        timeout_s: Maximum seconds to wait for conversion (None = no timeout)
        queue_timeout: Seconds to wait for result from queue after process exits

    Returns:
        Tuple of (markdown_text, images_dict, actual_page_count)

    Raises:
        ConversionTimeoutError: If conversion exceeds timeout_s
        RuntimeError: If worker crashes or returns error
    """
    ctx = mp.get_context("spawn")  # Required for CUDA
    result_queue: "mp.Queue[Dict[str, Any]]" = ctx.Queue()

    p = ctx.Process(
        target=_conversion_worker,
        args=(str(pdf_path), layout_bs, recognition_bs, detection_bs, result_queue),
    )
    p.start()

    try:
        start_time = time.time()
        result: Optional[Dict[str, Any]] = None

        while True:
            if timeout_s is not None and (time.time() - start_time) > timeout_s:
                print(f"  [timeout] Conversion exceeded {timeout_s}s - terminating...")
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    print(f"  [timeout] Process not responding to terminate - killing...")
                    p.kill()
                    p.join(timeout=2)
                raise ConversionTimeoutError(
                    f"Conversion of {pdf_path.name} exceeded {timeout_s}s limit"
                )

            try:
                result = result_queue.get(timeout=0.5)
                break
            except queue.Empty:
                if not p.is_alive():
                    break

        if result is None:
            if p.exitcode and p.exitcode != 0:
                raise RuntimeError(f"Worker crashed with exit code {p.exitcode}")
            raise RuntimeError("Worker completed but produced no output")

        if result.get("success", False):
            return (
                result["text"],
                result.get("images", {}),
                result.get("actual_pages", 1),
            )
        raise RuntimeError(f"Conversion failed: {result.get('error', 'Unknown error')}")

    finally:
        # Ensure cleanup even on exceptions
        if p.is_alive():
            p.kill()
            p.join(timeout=1)
        try:
            result_queue.close()
        except Exception:
            pass


def should_reduce_batch(current_bs: int, min_bs: int = MIN_BATCH_SIZE) -> Tuple[bool, int]:
    """
    Check if batch size should be reduced based on VRAM pressure.
    Returns (should_reduce, new_batch_size).

    AGGRESSIVE: Halve batch size when under pressure, don't wait for OOM.
    """
    if current_bs <= min_bs:
        return False, current_bs

    pressure = get_vram_pressure()

    if pressure >= VRAM_OVERFLOW_THRESHOLD:
        # Overflow: drop to minimum immediately
        print(f"  [vram] OVERFLOW ({pressure * 100:.0f}%), forcing min batch!")
        return True, min_bs

    if pressure >= VRAM_CRITICAL_THRESHOLD:
        # Critical (>90%): drop to minimum
        print(f"  [vram] CRITICAL ({pressure * 100:.0f}%), forcing min batch!")
        return True, min_bs

    if pressure >= VRAM_PRESSURE_THRESHOLD:
        # High (>80%): halve the batch size
        new_bs = max(min_bs, current_bs // 2)
        if new_bs < current_bs:
            print(
                f"  [vram] High pressure ({pressure * 100:.0f}%), reducing batch {current_bs} -> {new_bs}"
            )
            return True, new_bs

    return False, current_bs


def warmup_cuda() -> None:
    """Warm up CUDA to avoid cold start delays."""
    if not torch.cuda.is_available():
        return
    try:
        print("  Warming up CUDA...")
        # Allocate and free some memory to prime the allocator
        for size in [1, 10, 100]:
            t = torch.randn(size, size, device="cuda")
            _ = t @ t
            del t
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  [cuda] Warmup warning: {e}")


# ============================================================================
# FIXED BATCH CONFIGURATION (optimized for RTX 4070 Ti SUPER 16GB)
# ============================================================================
# Single reliable configuration - no presets, no OOM, consistent performance.
# These values are conservative but ALWAYS work, even for 500+ page PDFs.
# VRAM usage: ~8-12GB (leaves headroom for OS and spikes)

FIXED_LAYOUT_BATCH = 4  # Layout model batch size
FIXED_RECOGNITION_BATCH = 8  # Text recognition batch size
FIXED_DETECTION_BATCH = 8  # Object detection batch size
FIXED_WORKERS = 1  # Single worker (most stable)

# Fallback tiers for OOM recovery (halve sizes on each retry)
FALLBACK_TIERS = [8, 4, 2]

# ============================================================================
# HELPERS
# ============================================================================

_PAGE_MARKER_BLOCK_RE = re.compile(r"^\{(\d+)\}\s*\n-+\s*$", re.MULTILINE)
_PAGE_MARKER_LINE_RE = re.compile(r"^\{(\d+)\}\s*$", re.MULTILINE)
_PAGE_MARKER_ANY_RE = re.compile(r"^\[p:\d+\]", re.MULTILINE)

# Mojibake detection: Latin-1 interpreted CP1251 produces Unicode codepoints
# in U+00C0-U+00FF range that look like accented Western European letters.
# Example: "Îñíîâíàÿ" should be "Основная"
_MOJIBAKE_PATTERN = re.compile(r"[\u00c0-\u00ff]{3,}")  # 3+ consecutive Latin-1 Extended chars


def fix_mojibake(text: str) -> str:
    """Detect and fix CP1251 text misread as Latin-1 (common OCR issue for Russian PDFs).

    Pattern: "Îñíîâíàÿ çàäà÷à" -> "Основная задача"
    This happens when PDF embeds CP1251 text without proper Unicode mapping.
    """
    # Count high Latin-1 Extended characters (mojibake indicators)
    high_latin1_count = len(re.findall(r"[\u00c0-\u00ff]", text))

    # Count proper Cyrillic characters
    cyrillic_count = len(re.findall(r"[\u0400-\u04ff]", text))

    # Skip if already has lots of Cyrillic and few high Latin-1 chars
    if cyrillic_count > 100 and high_latin1_count < cyrillic_count * 0.1:
        return text  # Already good

    # Skip if no mojibake indicators
    if high_latin1_count < 50:
        return text  # Not enough evidence

    try:
        # Try to fix: encode as Latin-1 (what marker read it as), decode as CP1251
        fixed = text.encode("latin-1", errors="surrogateescape").decode("cp1251", errors="replace")

        # Verify fix improved the text (should contain more Cyrillic now)
        cyrillic_after = len(re.findall(r"[\u0400-\u04ff]", fixed))

        # Require at least 10 Cyrillic chars AND significant improvement
        # This prevents corrupting Western European (French, German) documents
        # where cyrillic_count=0 would make any cyrillic_after > 0 pass
        if cyrillic_after > max(10, cyrillic_count * 2):  # Significant improvement
            print(
                f"  [encoding] Fixed mojibake: {cyrillic_count} -> {cyrillic_after} Cyrillic chars"
            )
            return fixed

        return text  # Fix didn't help, keep original

    except (UnicodeDecodeError, UnicodeEncodeError) as e:
        log_warning(f"Mojibake fix failed: {e}")
        return text  # Can't fix, keep original


def normalize_page_markers(md_text: str) -> str:
    """Normalize marker pagination markers into stable [p:N] lines."""

    # Marker often emits:
    #   {12}
    #   ----------------------------------------
    # Convert it to a single line: [p:12]
    md_text = _PAGE_MARKER_BLOCK_RE.sub(lambda m: f"[p:{m.group(1)}]", md_text)
    md_text = _PAGE_MARKER_LINE_RE.sub(lambda m: f"[p:{m.group(1)}]", md_text)

    # Ensure the document starts with a page marker.
    # (Some renderers may only insert markers between pages.)
    if not md_text.lstrip().startswith("[p:"):
        md_text = "[p:1]\n\n" + md_text.lstrip("\n")

    # If markers are 0-based, shift to 1-based.
    markers = _PAGE_MARKER_LINE_RE.findall(md_text)
    if markers and any(m == "0" for m in markers):

        def _shift(m: re.Match[str]) -> str:
            return f"[p:{int(m.group(1)) + 1}]"

        md_text = _PAGE_MARKER_LINE_RE.sub(_shift, md_text)

    return md_text


def count_page_markers(md_text: str) -> int:
    markers = _PAGE_MARKER_LINE_RE.findall(md_text)
    if not markers:
        return 0
    try:
        return max(int(m) for m in markers)
    except ValueError:
        return 0


def output_has_page_markers(md_path: Path) -> bool:
    """Fast check: does an existing converted MD contain [p:N] markers?"""
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            for _ in range(800):
                line = f.readline()
                if not line:
                    break
                if line.startswith("[p:"):
                    return True
        return False
    except Exception:
        return False


def get_vram_usage():
    """Get current VRAM usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1024**3
    return 0.0


def clear_vram():
    """Clear CUDA cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def is_cuda_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "cuda out of memory" in msg or "out of memory" in msg or "cublas" in msg and "alloc" in msg
    )


def build_marker_config(layout_bs: int, recognition_bs: int, detection_bs: int):
    from marker.config.parser import ConfigParser

    return ConfigParser(
        {
            "output_format": "markdown",
            "layout_batch_size": layout_bs,
            "recognition_batch_size": recognition_bs,
            "detection_batch_size": detection_bs,
            "disable_tqdm": False,  # Show marker's progress bars
            # Enable page-level pagination so we can insert stable [p:N] markers.
            "paginate_output": True,
            # Keep the renderer's separator minimal; we normalize to [p:N] below.
            "page_separator": "\n",
        }
    ).generate_config_dict()


def get_page_count(pdf_path):
    """Get page count from PDF (handles non-ASCII paths)"""
    pdf_path = Path(pdf_path)

    # Try pypdfium2 first (fastest)
    try:
        import pypdfium2 as pdfium

        # pypdfium2 can fail on non-ASCII paths on Windows
        # Try with resolved path first, then with short path if needed
        try:
            doc = pdfium.PdfDocument(str(pdf_path.resolve()))
            count = len(doc)
            doc.close()
            if count > 0:
                return int(count)
        except Exception:
            pass

        # Try opening with bytes (works better with non-ASCII)
        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            doc = pdfium.PdfDocument(pdf_bytes)
            count = len(doc)
            doc.close()
            if count > 0:
                return int(count)
        except Exception:
            pass
    except ImportError:
        pass

    # Fallback to pypdf (slower, but reliable)
    try:
        from pypdf import PdfReader  # type: ignore

        return int(len(PdfReader(str(pdf_path)).pages))
    except Exception:
        pass

    # Fallback to PyPDF2
    try:
        from PyPDF2 import PdfReader  # type: ignore

        return int(len(PdfReader(str(pdf_path)).pages))
    except Exception:
        pass

    # Last resort: assume 100 pages (generous timeout)
    log_warning(f"Could not determine page count for {pdf_path.name}, assuming 100")
    return 100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PDF → Markdown via marker (with page markers)")
    p.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directory with PDF files")
    p.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for output markdown",
    )
    p.add_argument("--max-pages", type=int, default=MAX_PAGES, help="Skip PDFs longer than this")
    p.add_argument(
        "--pdf",
        dest="pdfs",
        action="append",
        default=[],
        help="Process only this PDF (can be repeated)",
    )
    p.add_argument(
        "--scan",
        action="store_true",
        help="Scan PDFs and print JSON (no OCR/model load)",
    )
    # Performance settings (simplified - no presets, fixed reliable config)
    p.add_argument(
        "--no-auto-fallback",
        action="store_true",
        help="Disable automatic batch size reduction on OOM",
    )
    return p.parse_args()


def scan_pdfs(input_dir: Path, output_dir: Path, max_pages: int) -> int:
    pdfs = sorted(input_dir.glob("*.pdf"))
    items = []
    for pdf in pdfs:
        md_path = output_dir / f"{pdf.stem}.md"
        pages = get_page_count(pdf)
        items.append(
            {
                "name": pdf.name,
                "path": str(pdf.resolve()),
                "pages": pages,
                "too_long": bool(pages and pages > max_pages),
                "done": bool(md_path.exists() and output_has_page_markers(md_path)),
            }
        )

    print(
        json.dumps(
            {"input_dir": str(input_dir), "count": len(items), "items": items},
            ensure_ascii=False,
        )
    )
    return 0


def format_time(seconds):
    """Format seconds as MM:SS"""
    if seconds < 0:
        return "--:--"
    mins, secs = divmod(int(seconds), 60)
    return f"{mins:02d}:{secs:02d}"


# ============================================================================
# MAIN
# ============================================================================


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    max_pages = args.max_pages
    selected_pdfs = [Path(p) for p in args.pdfs]

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.scan:
        return scan_pdfs(input_dir=input_dir, output_dir=output_dir, max_pages=max_pages)

    print("=" * 70)
    print("PDF → Markdown Converter")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(
        f"VRAM: {TOTAL_VRAM_GB - RESERVED_VRAM_GB:.0f}GB available ({RESERVED_VRAM_GB:.0f}GB reserved)"
    )
    print("=" * 70)
    print()

    # Find PDFs
    if selected_pdfs:
        pdfs = [p for p in selected_pdfs if p.exists()]
    else:
        pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found!")
        return

    # Count pages
    print("Scanning PDFs...")
    pdf_info = []
    skipped = []
    already_done = []
    total_pages = 0
    for pdf in pdfs:
        # Skip if already converted
        output_path = output_dir / f"{pdf.stem}.md"
        if output_path.exists() and output_has_page_markers(output_path):
            already_done.append(pdf.name)
            print(f"  {pdf.name[:50]:<50}        [DONE]")
            continue

        pages = get_page_count(pdf)
        if pages > max_pages:
            skipped.append((pdf, pages))
            print(f"  {pdf.name[:50]:<50} {pages:>4} pages  [SKIP: >{max_pages}]")
        else:
            pdf_info.append((pdf, pages))
            total_pages += pages
            print(f"  {pdf.name[:50]:<50} {pages:>4} pages")

    print()
    if already_done:
        print(f"Already done: {len(already_done)} PDFs")
    print(f"Processing: {len(pdf_info)} PDFs, {total_pages} pages")
    if skipped:
        print(f"Skipped: {len(skipped)} PDFs (>{max_pages} pages)")
    print("=" * 70)
    print()

    # Note: With multiprocessing, each PDF conversion subprocess loads models independently
    # This adds ~20-30s overhead per PDF but enables hard timeout (can kill hung conversions)
    print("Using multiprocessing mode (per-PDF model loading, hard timeout capable)")

    # Use fixed reliable batch configuration (no presets)
    layout_bs = FIXED_LAYOUT_BATCH
    recognition_bs = FIXED_RECOGNITION_BATCH
    detection_bs = FIXED_DETECTION_BATCH
    auto_fallback = not args.no_auto_fallback

    print(
        f"Batch config: layout={layout_bs}, recognition={recognition_bs}, detection={detection_bs}"
    )
    print(f"Auto-fallback: {auto_fallback}")

    # Start continuous GPU stats emission thread
    start_stats_thread()

    log_info("=" * 50)
    log_info(f"Starting conversion: {len(pdf_info)} PDFs, {total_pages} pages")
    log_info(f"Batch: L={layout_bs} R={recognition_bs} D={detection_bs}")
    log_info(f"Auto-fallback: {auto_fallback}")
    log_info("=" * 50)

    print()
    print("=" * 70)
    print("STARTING CONVERSION")
    print("=" * 70)
    print()

    # Stats
    processed_pages = 0
    processed_pdfs = 0
    failed = 0
    page_times = []
    start_time = time.time()
    peak_vram = 0.0

    for i, (pdf, expected_pages) in enumerate(pdf_info, 1):
        pdf_name = pdf.name
        short_name = pdf_name[:55] if len(pdf_name) > 55 else pdf_name

        # Calculate adaptive timeout for this PDF (None = no timeout)
        timeout_s = calculate_timeout(expected_pages)
        timeout_str = f"{timeout_s}s" if timeout_s else "OFF"

        log_info(f"PDF [{i}/{len(pdf_info)}] {pdf_name}")
        log_debug(f"  Expected pages: {expected_pages}, Timeout: {timeout_str}")

        print(f"\n{'─' * 70}")
        print(f"[{i}/{len(pdf_info)}] {short_name}")
        print(
            f"Pages: {expected_pages} | Progress: {processed_pages}/{total_pages} total | Timeout: {timeout_str}"
        )
        print(f"{'─' * 70}")

        pdf_start = time.time()

        success = False
        last_error: Optional[Exception] = None
        timed_out = False

        # Build batch size sequence: start with configured, then fallback tiers
        batch_sequence = [detection_bs]  # Start with current detection batch size
        if auto_fallback:
            for tier in FALLBACK_TIERS:
                if tier < detection_bs and tier not in batch_sequence:
                    batch_sequence.append(tier)

        for bs in batch_sequence:
            if timed_out:
                break

            # Check VRAM pressure and potentially reduce batch size
            if auto_fallback:
                reduce, new_bs = should_reduce_batch(bs)
                if reduce and new_bs < bs:
                    # Insert reduced batch size if not already in sequence
                    if new_bs not in batch_sequence:
                        batch_sequence.insert(batch_sequence.index(bs) + 1, new_bs)
                    bs = new_bs

            # Scale other batch sizes proportionally
            scale = bs / detection_bs if detection_bs > 0 else 1
            current_layout = max(2, int(layout_bs * scale))
            current_recognition = max(2, int(recognition_bs * scale))

            try:
                if bs != batch_sequence[0]:
                    print(
                        f"  Retrying with batch_size={bs} (layout={current_layout}, recognition={current_recognition})..."
                    )
                    cuda_full_cleanup()  # Aggressive cleanup before retry

                # Convert with timeout monitoring AND VRAM watchdog
                # Use multiprocessing for hard timeout (can kill hung CUDA calls)
                # Check CUDA health before starting
                if not cuda_health_check():
                    print("  [cuda] GPU unhealthy, attempting recovery...")
                    cuda_full_cleanup()
                    if not cuda_health_check():
                        raise RuntimeError("CUDA GPU unresponsive")

                print(f"  Converting with batch_size={bs} (timeout: {timeout_str})...")

                # Use multiprocessing-based conversion with hard timeout
                text, images, actual_pages = convert_pdf_with_timeout(
                    pdf_path=pdf,
                    layout_bs=current_layout,
                    recognition_bs=current_recognition,
                    detection_bs=bs,
                    timeout_s=timeout_s,
                )

                # Fix encoding issues (CP1251 misread as Latin-1) and normalize markers
                text = fix_mojibake(text)
                text = normalize_page_markers(text)
                marker_pages = count_page_markers(text)
                if marker_pages > 0:
                    actual_pages = marker_pages

                # Use actual pages from conversion if available
                if actual_pages == 0:
                    actual_pages = expected_pages

                pdf_time = time.time() - pdf_start
                page_time = pdf_time / actual_pages if actual_pages > 0 else pdf_time

                # Update stats
                processed_pdfs += 1
                processed_pages += actual_pages
                for _ in range(actual_pages):
                    page_times.append(page_time)

                # Track VRAM
                vram = get_vram_usage()
                peak_vram = max(peak_vram, vram)

                # Save
                output_path = output_dir / f"{pdf.stem}.md"
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(text)

                log_debug(f"  Saved markdown to {output_path}")

                # Save images (handle both bytes and PIL.Image objects)
                if images:
                    img_dir = output_dir / pdf.stem
                    img_dir.mkdir(exist_ok=True)
                    saved_images = 0
                    for img_name, img_data in images.items():
                        try:
                            img_path = img_dir / img_name
                            if isinstance(img_data, bytes):
                                # Already bytes, write directly
                                with open(img_path, "wb") as f:
                                    f.write(img_data)
                            elif hasattr(img_data, "save"):
                                # PIL.Image object, save it properly
                                img_data.save(str(img_path))
                            else:
                                log_warning(
                                    f"  Unknown image type for {img_name}: {type(img_data)}"
                                )
                                continue
                            saved_images += 1
                        except Exception as img_err:
                            log_warning(f"  Failed to save image {img_name}: {img_err}")
                    log_debug(f"  Saved {saved_images} images to {img_dir}")

                # Progress summary
                elapsed_total = time.time() - start_time
                avg_page = sum(page_times) / len(page_times) if page_times else 0
                remaining_pages = total_pages - processed_pages
                eta = remaining_pages * avg_page
                pages_per_min = processed_pages / (elapsed_total / 60) if elapsed_total > 0 else 0

                print()
                print(f"✓ DONE: {actual_pages} pages in {pdf_time:.1f}s ({page_time:.2f}s/page)")
                print(
                    f"  Total: {processed_pages}/{total_pages} pages | "
                    f"Speed: {pages_per_min:.1f} p/min | "
                    f"VRAM: {vram:.1f}GB | "
                    f"ETA: {format_time(eta)} | "
                    f"batch={bs}"
                )

                log_info(
                    f"  SUCCESS: {actual_pages} pages in {pdf_time:.1f}s ({page_time:.2f}s/page)"
                )
                log_debug(f"  Speed: {pages_per_min:.1f} p/min, VRAM: {vram:.1f}GB, batch={bs}")

                # Emit OCR text preview for Web UI (first 1000 chars)
                emit_ocr_preview(pdf_name, text, actual_pages)

                success = True
                break

            except ConversionTimeoutError as e:
                last_error = e
                timed_out = True
                log_error(f"  TIMEOUT: {e}")
                print(f"  [timeout] {e}")
                cuda_full_cleanup()
                # Don't continue to smaller batches on timeout - PDF is problematic
                break

            except Exception as e:
                last_error = e
                log_error(f"  Exception at batch={bs}: {e}")

                # Always clear VRAM before retrying.
                cuda_full_cleanup()

                # Check if this is VRAM-related (OOM or watchdog abort)
                error_msg = str(e).lower()
                is_vram_issue = (
                    is_cuda_oom(e) or "vram watchdog" in error_msg or "vram overflow" in error_msg
                )

                if torch.cuda.is_available() and is_vram_issue:
                    log_warning(f"  VRAM issue, will try smaller batch")
                    print(f"  VRAM issue at batch_size={bs}: {e}")
                    continue  # Try smaller batch

                # Non-VRAM error: still try smaller batches once
                print(f"  Error at batch_size={bs}: {e}")
                continue

        if not success:
            failed += 1
            log_error(f"  FAILED: {last_error}")
            if timed_out:
                print(f"\n✗ TIMEOUT: PDF conversion exceeded time limit")
            else:
                print(f"\n✗ FAILED: {last_error}")

        # Clean VRAM after each PDF
        cuda_full_cleanup()

    # Stop stats thread and final summary
    stop_stats_thread()
    total_time = time.time() - start_time
    avg_page = sum(page_times) / len(page_times) if page_times else 0
    speed = processed_pages / (total_time / 60) if total_time > 0 else 0

    # Log final summary
    log_info("=" * 50)
    log_info("COMPLETE")
    log_info(f"PDFs: {processed_pdfs}/{len(pdfs)} (failed: {failed})")
    log_info(f"Pages: {processed_pages}")
    log_info(f"Total time: {format_time(total_time)} ({total_time:.0f}s)")
    log_info(f"Speed: {speed:.1f} pages/min, Avg: {avg_page:.2f}s/page")
    log_info(f"Peak VRAM: {peak_vram:.1f}GB")
    log_info(f"Log file: {_log_filename}")
    log_info("=" * 50)

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"PDFs:       {processed_pdfs}/{len(pdfs)} (failed: {failed})")
    print(f"Pages:      {processed_pages}")
    print(f"Total time: {format_time(total_time)} ({total_time:.0f}s)")
    print(f"Avg/page:   {avg_page:.2f}s")
    print(f"Speed:      {speed:.1f} pages/min")
    print(f"Peak VRAM:  {peak_vram:.1f}GB")
    print(f"Output:     {output_dir}")
    print(f"Log:        {_log_filename}")
    print("=" * 70)


if __name__ == "__main__":
    raise SystemExit(main())
