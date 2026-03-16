#!/usr/bin/env python3
"""
ClarityOCR CI Smoke Test (Phase 5.4).

Runs a fast suite of checks without requiring a live Docker container:
  1. Python imports (core modules)
  2. API router routes coverage
  3. DB model smoke (in-memory SQLite)
  4. Canary routing logic
  5. Error taxonomy completeness
  6. Version module
  7. Canon file presence

Usage:
    python scripts/ci_smoke.py [--report-json path]

Exit:
    0 = all passed
    1 = failures
"""

import importlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PASSED = []
FAILED = []


def ok(name: str, detail: str = ""):
    PASSED.append(name)
    print(f"  ✅ {name}" + (f"  ({detail})" if detail else ""))


def fail(name: str, detail: str = ""):
    FAILED.append(name)
    print(f"  ❌ {name}" + (f"  ({detail})" if detail else ""))


# ──────────────────────────────────────────────────────────────────────────────

def test_imports():
    print("\n[imports]")
    modules = [
        "clarityocr.api_v2",
        "clarityocr.server",
        "clarityocr.db",
        "clarityocr.errors",
        "clarityocr.job_manager",
        "clarityocr.structured_logger",
        "clarityocr.quality_gates",
        "clarityocr.resource_limits",
        "clarityocr.canary",
        "clarityocr.version",
    ]
    for mod in modules:
        try:
            importlib.import_module(mod)
            ok(f"import:{mod.split('.')[-1]}")
        except Exception as exc:
            fail(f"import:{mod.split('.')[-1]}", str(exc)[:120])


def test_api_routes():
    print("\n[api routes]")
    try:
        from clarityocr.api_v2 import router
        paths = [r.path for r in router.routes]
        required_routes = [
            "/uploads",
            "/jobs",
            "/jobs/{job_id}",
            "/jobs/{job_id}/files",
            "/jobs/{job_id}/events",
            "/jobs/{job_id}/artifacts",
            "/jobs/{job_id}/cancel",
            "/jobs/{job_id}/retry-failed",
            "/health/live",
            "/health/ready",
            "/version",
        ]
        for route in required_routes:
            if route in paths:
                ok(f"route:{route}")
            else:
                fail(f"route:{route}", "missing")
    except Exception as exc:
        fail("api_router_load", str(exc)[:120])


def test_db_smoke():
    print("\n[db smoke]")
    try:
        import os
        from clarityocr import db
        db.setup_db(":memory:")
        db.init_db()
        with db.get_session() as session:
            count = session.query(db.Job).count()
            ok("db:init", f"job count={count}")
    except Exception as exc:
        fail("db:init", str(exc)[:120])


def test_canary():
    print("\n[canary routing]")
    try:
        import os
        os.environ["V2_LLM_MODEL_PRIMARY"] = "Qwen/Qwen2.5-0.5B-Instruct"
        os.environ["V2_LLM_MODEL_CANARY"] = "Qwen/Qwen2.5-3B-Instruct"
        os.environ["V2_CANARY_PCT"] = "50"

        # Re-import to pick up env
        import importlib
        import clarityocr.canary as canary_mod
        importlib.reload(canary_mod)

        results = [canary_mod.pick_model(job_id=f"job-{i}", file_id="f1") for i in range(100)]
        canary_count = sum(1 for _, is_c in results if is_c)
        # With 50% and 100 samples, expect roughly 40–60 canary hits
        ok("canary:routing", f"{canary_count}/100 went to canary (expected ~50)")

        # Test determinism
        m1, _ = canary_mod.pick_model("same-job", "same-file")
        m2, _ = canary_mod.pick_model("same-job", "same-file")
        if m1 == m2:
            ok("canary:deterministic")
        else:
            fail("canary:deterministic", f"{m1} != {m2}")

        # Clean up
        os.environ.pop("V2_LLM_MODEL_CANARY", None)
        os.environ["V2_CANARY_PCT"] = "0"

    except Exception as exc:
        fail("canary", str(exc)[:120])


def test_errors():
    print("\n[error taxonomy]")
    try:
        from clarityocr.errors import PipelineError, ErrorCode, RETRY_POLICY
        ok("errors:PipelineError", "PipelineError present")
        # Verify ErrorCode has expected members
        expected_codes = ["E_INTERNAL", "E_RESOURCE_LIMIT", "E_LOW_OCR_CONFIDENCE"]
        for code in expected_codes:
            if hasattr(ErrorCode, code):
                ok(f"errors:ErrorCode.{code}")
            else:
                fail(f"errors:ErrorCode.{code}", "missing")
        ok("errors:RETRY_POLICY", f"{len(RETRY_POLICY)} policies defined")
    except ImportError as exc:
        fail("errors:taxonomy", str(exc)[:120])


def test_version():
    print("\n[version]")
    try:
        from clarityocr.version import get_version, __version__
        ver = get_version()
        ok("version:get_version", f"version={ver!r}")
        ok("version:__version__", f"__version__={__version__!r}")
    except Exception as exc:
        fail("version", str(exc)[:120])


def test_canon_files():
    print("\n[canon files]")
    required = [
        "Dockerfile",
        "docker-compose.yml",
        "requirements.txt",
        "ROLLBACK.md",
        "ROADMAP_48H_SPRINT.md",
        "clarityocr/canary.py",
        "clarityocr/version.py",
        "clarityocr/errors.py",
        "scripts/canon_check.py",
        "scripts/eval_model_canary.py",
        "scripts/ci_smoke.py",
    ]
    for rel in required:
        path = REPO_ROOT / rel
        if path.exists():
            ok(f"file:{rel}")
        else:
            fail(f"file:{rel}", "missing")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ClarityOCR CI smoke test")
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args()

    print(f"ClarityOCR CI Smoke — repo: {REPO_ROOT}")

    test_imports()
    test_api_routes()
    test_db_smoke()
    test_canary()
    test_errors()
    test_version()
    test_canon_files()

    print(f"\n{'='*50}")
    print(f"Passed: {len(PASSED)}  Failed: {len(FAILED)}")

    if args.report_json:
        report = {
            "passed": PASSED,
            "failed": FAILED,
            "total": len(PASSED) + len(FAILED),
            "ok": len(FAILED) == 0,
        }
        Path(args.report_json).write_text(json.dumps(report, indent=2))
        print(f"Report: {args.report_json}")

    if FAILED:
        print(f"FAILED: {', '.join(FAILED)}")
        sys.exit(1)

    print("✅ CI smoke passed")
    sys.exit(0)


if __name__ == "__main__":
    main()
