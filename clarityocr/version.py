"""
ClarityOCR versioning (Phase 5.2).

Version is sourced from:
1. CLARITYOCR_VERSION env var (set at Docker build time via --build-arg)
2. Git describe (if running from a git repo)
3. Fallback: "dev"

Build with:
    docker build --build-arg VERSION=$(git rev-parse --short HEAD) -t clarityocr:$(git rev-parse --short HEAD) .
"""

import os
import subprocess
from pathlib import Path


def _git_version() -> str:
    try:
        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            ["git", "-C", str(repo_root), "describe", "--tags", "--always", "--dirty"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def get_version() -> str:
    env_ver = os.getenv("CLARITYOCR_VERSION", "").strip()
    if env_ver:
        return env_ver
    git_ver = _git_version()
    if git_ver:
        return git_ver
    return "dev"


__version__ = get_version()
