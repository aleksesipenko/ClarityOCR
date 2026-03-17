"""
Quality gates for ClarityOCR pipeline.
Phase 4.2: Per-stage quality checks with configurable thresholds.
"""

import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple


def _env_float(name: str, default: float) -> float:
    """Safe env var parsing — invalid values fall back to default."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (ValueError, TypeError):
        return default


@dataclass
class QualityGate:
    """Quality gate configuration."""
    name: str
    threshold: float
    failure_mode: str  # "warn" | "fail"


# OCR Quality Gates
OCR_GATES = [
    QualityGate(
        name="ocr_confidence",
        threshold=_env_float("QG_OCR_CONFIDENCE_MIN", 0.7),
        failure_mode="warn"
    ),
    QualityGate(
        name="empty_output",
        threshold=1.0,
        failure_mode="fail"
    ),
]

# Polish Quality Gates
POLISH_GATES = [
    QualityGate(
        name="polish_diff_ratio",
        threshold=_env_float("QG_POLISH_DIFF_RATIO_MAX", 0.3),
        failure_mode="warn"
    ),
    QualityGate(
        name="polish_length_ratio",
        threshold=_env_float("QG_POLISH_LENGTH_RATIO_MAX", 3.0),
        failure_mode="fail"
    ),
]


def _estimate_ocr_confidence(md_text: str) -> float:
    """
    Heuristic OCR confidence estimation based on text quality indicators.
    Returns confidence in range [0.0, 1.0].

    Factors:
    - Presence of common OCR artifacts
    - Character diversity
    - Word coherence
    """
    if not md_text or len(md_text.strip()) < 10:
        return 0.0

    text = md_text.lower()

    # Penalty factors
    penalty = 0.0

    # Check for excessive special characters (OCR noise)
    special_chars = len(re.findall(r'[^a-zA-Z0-9\s\.,;:!?\-\'\"()]', text))
    char_ratio = special_chars / len(text)
    if char_ratio > 0.1:
        penalty += 0.2

    # Check for repeated characters (OCR artifacts like "oooo" or "||||")
    repeated_patterns = len(re.findall(r'(.)\1{3,}', text))
    if repeated_patterns > 5:
        penalty += 0.15

    # Check for low word diversity (sign of garbled OCR)
    words = text.split()
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            penalty += 0.15

    # Baseline confidence
    confidence = 1.0 - min(penalty, 0.8)

    return max(0.0, min(1.0, confidence))


def check_ocr_quality(md_text: str) -> Tuple[bool, Optional[str], Optional[dict]]:
    """
    Check OCR output quality.

    Returns:
        (passed, warning_message, metadata)
    """
    confidence = _estimate_ocr_confidence(md_text)

    for gate in OCR_GATES:
        if gate.name == "ocr_confidence" and confidence < gate.threshold:
            msg = f"OCR confidence {confidence:.2f} below threshold {gate.threshold}"
            if gate.failure_mode == "fail":
                return False, msg, {"confidence": confidence}
            else:
                return True, msg, {"confidence": confidence}

        if gate.name == "empty_output" and len(md_text.strip()) == 0:
            msg = "OCR output is empty"
            return False, msg, {"confidence": 0.0}

    return True, None, {"confidence": confidence}


def check_polish_quality(original_text: str, polished_text: str) -> Tuple[bool, Optional[str], Optional[dict]]:
    """
    Check polish output quality for hallucination detection.

    Returns:
        (passed, warning_message, metadata)
    """
    if not original_text or not polished_text:
        return True, None, {"length_ratio": 1.0, "diff_ratio": 0.0}

    original_len = len(original_text)
    polished_len = len(polished_text)

    length_ratio = polished_len / original_len if original_len > 0 else 1.0

    # Calculate diff ratio (simple character-level difference)
    # A more sophisticated approach would use diff algorithms
    diff_chars = abs(polished_len - original_len)
    diff_ratio = diff_chars / original_len if original_len > 0 else 0.0

    metadata = {
        "length_ratio": round(length_ratio, 2),
        "diff_ratio": round(diff_ratio, 2),
        "original_length": original_len,
        "polished_length": polished_len,
    }

    for gate in POLISH_GATES:
        if gate.name == "polish_length_ratio" and length_ratio > gate.threshold:
            msg = f"Polish output length ratio {length_ratio:.2f} exceeds threshold {gate.threshold} (possible hallucination)"
            if gate.failure_mode == "fail":
                return False, msg, metadata
            else:
                return True, msg, metadata

        if gate.name == "polish_diff_ratio" and diff_ratio > gate.threshold:
            msg = f"Polish diff ratio {diff_ratio:.2f} exceeds threshold {gate.threshold}"
            if gate.failure_mode == "fail":
                return False, msg, metadata
            else:
                return True, msg, metadata

    return True, None, metadata
