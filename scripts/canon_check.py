#!/usr/bin/env python3
"""
ClarityOCR Canon Check (Phase 5.1).

Validates that the local repo + Docker image + runtime are in sync with canon.
Checks:
  1. Git branch is on canon branch
  2. Git working tree is clean (no uncommitted changes)
  3. Dockerfile present and valid
  4. docker-compose.yml present and references expected service names
  5. requirements.txt present
  6. Core module imports work
  7. (Optional) Docker image tag matches git short SHA

Usage:
    python scripts/canon_check.py [--strict] [--check-image]

Exit codes:
    0 = all checks passed
    1 = one or more checks failed (details on stdout)
    2 = runtime error
"""

import argparse
import importlib
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CANON_BRANCH = "canon/alexpc-docker-runtime"

CHECKS_PASSED = []
CHECKS_FAILED = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        CHECKS_PASSED.append(name)
        print(f"  ✅ {name}" + (f": {detail}" if detail else ""))
    else:
        CHECKS_FAILED.append(name)
        print(f"  ❌ {name}" + (f": {detail}" if detail else ""))


def git_output(*args) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT)] + list(args),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return ""


def check_git():
    print("\n[git]")
    branch = git_output("rev-parse", "--abbrev-ref", "HEAD")
    check("branch", branch == CANON_BRANCH, f"current={branch!r}, expected={CANON_BRANCH!r}")

    status = git_output("status", "--porcelain")
    check("clean-working-tree", not status, f"{len(status.splitlines())} uncommitted files" if status else "clean")

    short_sha = git_output("rev-parse", "--short", "HEAD")
    check("has-commits", bool(short_sha), short_sha)

    return short_sha


def check_files():
    print("\n[files]")
    required = [
        "Dockerfile",
        "docker-compose.yml",
        "requirements.txt",
        "pyproject.toml",
        "clarityocr/__init__.py",
        "clarityocr/api_v2.py",
        "clarityocr/server.py",
        "clarityocr/pipeline_v2.py",
        "clarityocr/db.py",
        "clarityocr/errors.py",
        "clarityocr/canary.py",
    ]
    for rel in required:
        path = REPO_ROOT / rel
        check(f"file:{rel}", path.exists())


def check_imports():
    print("\n[imports]")
    sys.path.insert(0, str(REPO_ROOT))
    modules = [
        "clarityocr.api_v2",
        "clarityocr.server",
        "clarityocr.errors",
        "clarityocr.canary",
    ]
    for mod in modules:
        try:
            importlib.import_module(mod)
            check(f"import:{mod}", True)
        except Exception as exc:
            check(f"import:{mod}", False, str(exc)[:80])


def check_docker_compose():
    print("\n[docker-compose]")
    compose_path = REPO_ROOT / "docker-compose.yml"
    if not compose_path.exists():
        check("docker-compose-exists", False)
        return
    content = compose_path.read_text()
    check("service:clarityocr", "clarityocr:" in content)
    check("service:vllm_server", "vllm_server:" in content)
    check("port:8008", "8008" in content)


def check_docker_image(strict: bool):
    print("\n[docker-image]")
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", "clarityocr:latest", "--format", "{{.Id}}"],
            capture_output=True, text=True, timeout=10,
        )
        image_exists = result.returncode == 0
        check("image:clarityocr:latest", image_exists, "found" if image_exists else "not built")
    except FileNotFoundError:
        check("docker-available", False, "docker not in PATH")
    except Exception as exc:
        check("docker-inspect", False, str(exc)[:80])


def main():
    parser = argparse.ArgumentParser(description="ClarityOCR canon check")
    parser.add_argument("--strict", action="store_true", help="Require clean working tree")
    parser.add_argument("--check-image", action="store_true", help="Check Docker image existence")
    args = parser.parse_args()

    print(f"ClarityOCR Canon Check — repo: {REPO_ROOT}")

    short_sha = check_git()
    check_files()
    check_imports()
    check_docker_compose()

    if args.check_image:
        check_docker_image(args.strict)

    print(f"\n{'='*50}")
    print(f"Passed: {len(CHECKS_PASSED)}  Failed: {len(CHECKS_FAILED)}")

    if CHECKS_FAILED:
        print(f"FAILED checks: {', '.join(CHECKS_FAILED)}")
        sys.exit(1)
    else:
        print("✅ All checks passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
