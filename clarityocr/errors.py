"""
Error taxonomy and retry policies for ClarityOCR pipeline.
Phase 4.1: Structured error codes with distinct retry behaviors.
"""

from enum import Enum
from typing import Dict, Any


class ErrorCode(Enum):
    """Structured error codes for ClarityOCR pipeline."""

    # Transient errors (should retry)
    E_NETWORK_TIMEOUT = "E_NETWORK_TIMEOUT"
    E_VRAM_OOM = "E_VRAM_OOM"
    E_LLM_TIMEOUT = "E_LLM_TIMEOUT"

    # Fatal errors (do not retry)
    E_INVALID_PDF = "E_INVALID_PDF"
    E_CORRUPTED_FILE = "E_CORRUPTED_FILE"
    E_EMPTY_OCR_OUTPUT = "E_EMPTY_OCR_OUTPUT"

    # System errors
    E_LEASE_TIMEOUT = "E_LEASE_TIMEOUT"
    E_INTERNAL = "E_INTERNAL"
    E_RESOURCE_LIMIT = "E_RESOURCE_LIMIT"

    # Quality gate errors
    E_POLISH_HALLUCINATION = "E_POLISH_HALLUCINATION"
    E_LOW_OCR_CONFIDENCE = "E_LOW_OCR_CONFIDENCE"


RETRY_POLICY: Dict[ErrorCode, Dict[str, int]] = {
    # Transient errors - retry with backoff
    ErrorCode.E_NETWORK_TIMEOUT: {"max_attempts": 3, "backoff_sec": 5},
    ErrorCode.E_VRAM_OOM: {"max_attempts": 2, "backoff_sec": 30},
    ErrorCode.E_LLM_TIMEOUT: {"max_attempts": 2, "backoff_sec": 10},

    # Fatal errors - do not retry
    ErrorCode.E_INVALID_PDF: {"max_attempts": 1, "backoff_sec": 0},
    ErrorCode.E_CORRUPTED_FILE: {"max_attempts": 1, "backoff_sec": 0},
    ErrorCode.E_EMPTY_OCR_OUTPUT: {"max_attempts": 1, "backoff_sec": 0},

    # System errors - limited retries
    ErrorCode.E_LEASE_TIMEOUT: {"max_attempts": 3, "backoff_sec": 0},
    ErrorCode.E_INTERNAL: {"max_attempts": 2, "backoff_sec": 5},
    ErrorCode.E_RESOURCE_LIMIT: {"max_attempts": 1, "backoff_sec": 0},

    # Quality gate errors - do not retry
    ErrorCode.E_POLISH_HALLUCINATION: {"max_attempts": 1, "backoff_sec": 0},
    ErrorCode.E_LOW_OCR_CONFIDENCE: {"max_attempts": 1, "backoff_sec": 0},
}


class PipelineError(Exception):
    """
    Structured exception for pipeline errors.
    Associates error code with message for proper retry handling.
    """

    def __init__(self, error_code: ErrorCode, message: str):
        self.error_code = error_code
        self.message = message
        super().__init__(f"[{error_code.value}] {message}")

    def get_retry_policy(self) -> Dict[str, int]:
        """Get retry policy for this error code."""
        return RETRY_POLICY.get(self.error_code, {"max_attempts": 1, "backoff_sec": 0})
