"""
ClarityOCR — A/B Canary Routing (Phase 3.2).

Controls LLM model routing for polish stage.
When CANARY_MODEL is set, a fraction (CANARY_PCT, default 10%) of polish
requests are routed to the canary model; the rest go to the primary model.

Environment variables:
    V2_LLM_MODEL_PRIMARY   Primary model name (default: Qwen/Qwen2.5-0.5B-Instruct)
    V2_LLM_MODEL_CANARY    Canary model name (default: None → canary disabled)
    V2_CANARY_PCT          Percentage 0-100 of requests to canary (default: 10)
    V2_LLM_BASE_URL        Base URL for both models (default: http://localhost:8000/v1)

Usage (in pipeline):
    from clarityocr.canary import pick_model, CANARY_ACTIVE

    model_name, is_canary = pick_model(job_id=job_id)
    # use model_name in LLM call
"""

import hashlib
import os
from typing import Tuple


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


PRIMARY_MODEL: str = _env_str("V2_LLM_MODEL_PRIMARY", "Qwen/Qwen2.5-0.5B-Instruct")
CANARY_MODEL: str | None = _env_str("V2_LLM_MODEL_CANARY", "") or None
CANARY_PCT: int = max(0, min(100, _env_int("V2_CANARY_PCT", 10)))

CANARY_ACTIVE: bool = bool(CANARY_MODEL)


def pick_model(job_id: str = "", file_id: str = "") -> Tuple[str, bool]:
    """
    Pick a model for a given job/file combination.

    Returns:
        (model_name, is_canary)  — is_canary=True means this request goes to canary model.

    Routing is deterministic per (job_id, file_id) so retries get the same model.
    If canary is disabled, always returns (PRIMARY_MODEL, False).
    """
    if not CANARY_ACTIVE or not CANARY_MODEL:
        return PRIMARY_MODEL, False

    if CANARY_PCT <= 0:
        return PRIMARY_MODEL, False

    if CANARY_PCT >= 100:
        return CANARY_MODEL, True

    # Stable hash-based routing: same job+file always gets same model
    seed = f"{job_id}:{file_id}"
    digest = int(hashlib.md5(seed.encode()).hexdigest(), 16)
    bucket = digest % 100

    if bucket < CANARY_PCT:
        return CANARY_MODEL, True
    return PRIMARY_MODEL, False


def canary_status() -> dict:
    """Return current canary routing status as a dict (for health/admin endpoints)."""
    return {
        "canary_active": CANARY_ACTIVE,
        "primary_model": PRIMARY_MODEL,
        "canary_model": CANARY_MODEL,
        "canary_pct": CANARY_PCT if CANARY_ACTIVE else 0,
    }
