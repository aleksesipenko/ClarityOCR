# ClarityOCR Enterprise Upgrade Plan

**Version:** 1.0
**Date:** 2026-03-16
**Target:** Production-grade OCR pipeline with observability, operator tooling, and quality gates

---

## Executive Summary

ClarityOCR is a functional OCR pipeline with solid foundations: SQLite job tracking, worker threads, circuit breaker for LLM polish, and basic web UI. **Current weaknesses:**

1. **No progress visibility** — jobs are black boxes until completion
2. **Thin web UI** — upload/submit only, no live monitoring or diagnostics
3. **Grounding model stale** — Qwen2.5-0.5B is lightweight but dated; Qwen 3.5 offers better quality
4. **No structured logging** — file logs exist but lack job/stage correlation
5. **Brittle error handling** — retries exist, but no error taxonomy or graceful degradation
6. **Drift discipline missing** — no CI, no automated canon checks, manual deploy

This plan phases the work to maximize operator value first (visibility + control), then quality (model upgrade), then hardening (error handling + CI/CD).

---

## Phase 1: Progress & Observability

**Goal:** Make running jobs visible with named stages, timing, and structured logs.

### 1.1 Job Stage Model

**Current state:** Jobs have `status` (queued/running/completed/partial/failed/canceled) but no substage visibility within "running."

**Change:** Introduce file-level stage tracking.

#### Files to create/modify:

- **`clarityocr/db.py`** (modify)
  - Add `JobFile.stage` column (nullable string, default `NULL`)
  - Values: `upload` → `ocr` → `merge` → `polish` → `convert` → `done`
  - Add `JobFile.stage_started_at` (datetime, nullable)
  - Add `JobFile.stage_progress_pct` (integer 0-100, nullable)
  - Add `JobFile.pages_total` (integer, nullable)
  - Add `JobFile.pages_done` (integer, nullable)

- **`clarityocr/pipeline_v2.py`** (modify)
  - In `_process_file()`, wrap each logical step with stage transitions:
    ```python
    with _stage_context(session, file_id, "ocr"):
        # run converter subprocess
    with _stage_context(session, file_id, "polish"):
        # run polish subprocess
    ```
  - Helper: `_stage_context(session, file_id, stage_name)` → context manager that:
    - Sets `stage`, `stage_started_at`, emits event `file_stage_start`
    - On exit: emits event `file_stage_done` with duration

- **`clarityocr/job_manager.py`** (modify)
  - Add `set_file_stage(session, file_id, stage, progress_pct=None, pages_done=None, pages_total=None)`
  - Update `JobFile` with optimistic lock

#### API contract changes:

- **`GET /api/v2/jobs/{job_id}/files`** (extend response)
  ```json
  {
    "id": "file-uuid",
    "status": "running",
    "stage": "ocr",
    "stage_progress_pct": 45,
    "pages_done": 9,
    "pages_total": 20,
    "duration_ms": 12300
  }
  ```

- **`GET /api/v2/jobs/{job_id}`** (extend response)
  ```json
  {
    "job_id": "...",
    "status": "running",
    "overall_progress_pct": 33,
    "eta_seconds": 47,
    "files_done": 2,
    "files_total": 6
  }
  ```

#### Acceptance criteria:

1. Fresh job shows stage transitions in events (`GET /api/v2/jobs/{job_id}/events`)
2. Running file shows current stage + progress percentage
3. Job-level `overall_progress_pct` = `(completed_files + sum(running_files.stage_progress_pct/100)) / total_files`
4. ETA calculation: linear extrapolation from `(pages_done / pages_total) * avg_page_time`

**Complexity:** M (DB migration + pipeline instrumentation)

**Dependencies:** None

**Risk:** Low — additive changes, no breaking changes to existing API

---

### 1.2 Structured Logging

**Current state:** File logs (`~/.clarityocr/logs/ocr_*.log`) exist but are unstructured text.

**Change:** Emit JSON logs with job_id/file_id/stage/duration fields.

#### Files to create/modify:

- **`clarityocr/converter.py`** (modify)
  - Replace `log_info(msg)` with `log_structured(level, msg, **context)`
  - Context: `{"job_id": "...", "file_id": "...", "stage": "ocr"}`

- **New file:** `clarityocr/structured_logger.py`
  ```python
  import logging
  import json
  from typing import Any, Dict

  class StructuredLogger:
      def __init__(self, name: str):
          self.logger = logging.getLogger(name)
          handler = logging.FileHandler(...)
          handler.setFormatter(JsonFormatter())
          self.logger.addHandler(handler)

      def log(self, level: str, msg: str, **context):
          record = {
              "timestamp": datetime.utcnow().isoformat(),
              "level": level,
              "message": msg,
              **context
          }
          self.logger.log(getattr(logging, level.upper()), json.dumps(record))
  ```

- **`clarityocr/pipeline_v2.py`** (modify)
  - Add context to all `logger.info()` calls: `logger.info("OCR started", extra={"job_id": job_id, "file_id": file_id, "stage": "ocr"})`

#### API contract changes:

None (internal logging only)

#### Acceptance criteria:

1. Log file contains valid JSON lines (one per log entry)
2. Each OCR/polish/merge operation logs: `job_id`, `file_id`, `stage`, `duration_ms`
3. Logs can be piped to `jq` for filtering: `cat ocr_*.log | jq 'select(.job_id=="abc")'`

**Complexity:** S (logging wrappers, no logic changes)

**Dependencies:** None

**Risk:** Very low — internal observability improvement

---

### 1.3 Health Endpoint Enhancement

**Current state:** `/api/health/ready` returns hardcoded `{"ocr_core": "ready", "db": "ready", "llm": "degraded"}`

**Change:** Real component probes.

#### Files to modify:

- **`clarityocr/api_v2.py`** (modify `readiness()` function)
  ```python
  def readiness():
      ocr_status = _probe_ocr_core()
      db_status = _probe_db()
      llm_status = _probe_llm()

      overall = "ready" if all(s == "ready" for s in [ocr_status, db_status, llm_status]) else "degraded"

      return {
          "status": overall,
          "components": {
              "ocr_core": ocr_status,
              "db": db_status,
              "llm": llm_status
          }
      }
  ```

- **New functions in `api_v2.py`:**
  - `_probe_ocr_core()` → check if marker-pdf imports work
  - `_probe_db()` → run `SELECT 1` on DB
  - `_probe_llm()` → try `httpx.get(llm_base_url + "/health", timeout=2)`

#### API contract changes:

- **`GET /api/health/ready`** (breaking change — structure now nested)
  ```json
  {
    "status": "ready",
    "components": {
      "ocr_core": "ready",
      "db": "ready",
      "llm": "degraded"
    }
  }
  ```

#### Acceptance criteria:

1. Health endpoint returns `"status": "ready"` when all probes pass
2. Health endpoint returns `"status": "degraded"` when LLM is down but OCR/DB healthy
3. Probe failures log errors to structured logs

**Complexity:** S (simple probe functions)

**Dependencies:** None

**Risk:** Low — existing `/api/health` endpoint is deprecated, `/api/health/ready` is new

---

## Phase 2: Operator UI

**Goal:** Replace thin upload-only UI with real-time job monitoring console.

### 2.1 Progress Dashboard

**Current state:** Web UI (`clarityocr/web/static/index.html` + `v2-app.js`) allows upload + job submit, shows job status as text blob.

**Change:** Live progress bars, stage timeline, file-level breakdown.

#### Files to create/modify:

- **`clarityocr/web/static/v2-app.js`** (heavy modification)
  - Add `JobMonitor` class that polls `GET /api/v2/jobs/{job_id}` every 2 seconds
  - Render progress bars per file (HTML5 `<progress>` element)
  - Show stage timeline: `upload [✓] → ocr [45%] → polish [waiting] → done`
  - Show ETA countdown

- **`clarityocr/web/static/v2-style.css`** (add)
  - Styles for progress bars, stage badges, file cards
  - Visual hierarchy: job → files → stages

- **`clarityocr/web/static/index.html`** (modify)
  - Add `<section id="jobMonitor">` placeholder
  - Add WebSocket connection stubs (for future Phase 2.3)

#### API contract changes:

None (uses existing `/api/v2/jobs/{job_id}` with Phase 1 enhancements)

#### Acceptance criteria:

1. Submit job → UI shows live progress bar updating every 2 seconds
2. Click file card → expand to show per-stage breakdown
3. Job completion → progress bar turns green, ETA disappears
4. Error state → progress bar turns red, shows error message from `last_error_message`

**Complexity:** M (significant frontend work, no backend changes)

**Dependencies:** Phase 1 (progress fields must exist in API)

**Risk:** Low — UI-only changes, no impact on pipeline

---

### 2.2 Error Display

**Current state:** Errors appear only in `/api/v2/jobs/{job_id}/files` as `last_error_message` strings.

**Change:** Structured error cards with actionable context.

#### Files to modify:

- **`clarityocr/web/static/v2-app.js`** (add error rendering)
  ```javascript
  function renderError(file) {
    if (file.status !== "failed_final") return "";

    const errorCard = `
      <div class="error-card">
        <h4>❌ ${file.input_path.split('/').pop()}</h4>
        <p class="error-code">${file.last_error_code || "E_UNKNOWN"}</p>
        <p class="error-msg">${file.last_error_message}</p>
        <button onclick="retryFile('${file.id}')">Retry</button>
      </div>
    `;
    return errorCard;
  }
  ```

- **`clarityocr/api_v2.py`** (add retry endpoint)
  - Already exists: `POST /api/v2/jobs/{job_id}/retry-failed`
  - Add file-level retry: `POST /api/v2/files/{file_id}/retry`

#### API contract changes:

- **New endpoint:** `POST /api/v2/files/{file_id}/retry`
  - Sets file status back to `queued` (if `attempt < max_attempts`)
  - Returns `{"status": "queued", "attempt": 2}`

#### Acceptance criteria:

1. Failed file shows error card with code + message
2. Click "Retry" → file requeues, progress bar reappears
3. Failed job (all files failed) shows aggregated error summary

**Complexity:** S (UI rendering + trivial endpoint)

**Dependencies:** None

**Risk:** Very low

---

### 2.3 Job Queue Overview

**Current state:** No way to see all running/queued jobs at once.

**Change:** Dashboard showing all jobs with filtering/sorting.

#### Files to create:

- **New page:** `clarityocr/web/static/queue.html`
  - Table view: `job_id | status | mode | files | progress | started | actions`
  - Filters: status (running/queued/completed), mode
  - Sort: by `created_at` desc (default), by `progress` asc

- **`clarityocr/web/static/v2-app.js`** (add)
  - `QueueView` class that polls `GET /api/v2/jobs?limit=50`
  - Click row → navigate to job detail view

#### API contract changes:

None (uses existing `GET /api/v2/jobs` endpoint)

#### Acceptance criteria:

1. Navigate to `/queue.html` → see all jobs
2. Filter by status → only matching jobs shown
3. Sort by progress → jobs with lowest progress first
4. Click job → navigate to job detail view (existing `index.html` with `?job_id=...`)

**Complexity:** M (new page, moderate frontend work)

**Dependencies:** None

**Risk:** Very low

---

## Phase 3: Grounding Model Upgrade

**Goal:** Replace Qwen2.5-0.5B with Qwen 3.5 for polish lane, validate quality improvement.

### 3.1 Qwen 3.5 Evaluation

**Current state:** `clarityocr/polish.py` uses vLLM endpoint with `V2_LLM_MODEL` env var (default `"local-model"`).

**Change:** Shadow mode — run both models in parallel, log outputs, no user-facing changes.

#### Files to create/modify:

- **New script:** `scripts/eval_qwen35.py`
  ```python
  import subprocess
  from pathlib import Path

  test_files = [
      "test_data/sample_ocr_1.md",
      "test_data/sample_ocr_2.md",
      # ... 10 representative files
  ]

  for file in test_files:
      # Run with Qwen2.5-0.5B
      result_old = run_polish(file, model="Qwen2.5-0.5B")

      # Run with Qwen 3.5
      result_new = run_polish(file, model="Qwen3.5-3B-Instruct")

      # Save both outputs
      save_comparison(file, result_old, result_new)
  ```

- **New directory:** `test_data/` with 10 representative OCR outputs (Russian + English)

- **New file:** `clarityocr/polish_dual.py` (temporary shadow mode version)
  - Calls LLM twice (old + new model)
  - Logs both outputs to structured logs
  - Returns old model result (no user-facing change)

#### API contract changes:

None (internal evaluation only)

#### Acceptance criteria:

1. Script runs 10 test files through both models
2. Output saved to `eval_results/{file_name}_{model}.md`
3. Manual review shows Qwen 3.5 fixes more OCR errors without hallucinations
4. Evaluation report documents:
   - Error correction rate (manual count of fixes)
   - Hallucination rate (manual count of wrong edits)
   - Output length ratio (should stay 0.9-1.1x)

**Complexity:** S (isolated evaluation script)

**Dependencies:** None (requires vLLM with Qwen 3.5 model loaded)

**Risk:** Very low — offline evaluation, no production impact

---

### 3.2 A/B Migration Path

**Current state:** Single model configured via env var.

**Change:** Canary rollout — X% of jobs use new model, rest use old.

#### Files to modify:

- **`clarityocr/pipeline_v2.py`** (modify `_try_optional_polish()`)
  ```python
  def _select_polish_model(job_id: str, config: dict) -> str:
      canary_pct = int(os.getenv("POLISH_CANARY_PCT", "0"))
      if canary_pct == 0:
          return config.get("llm_model", "Qwen2.5-0.5B")

      # Deterministic canary selection (hash job_id)
      import hashlib
      job_hash = int(hashlib.md5(job_id.encode()).hexdigest(), 16)
      if (job_hash % 100) < canary_pct:
          return "Qwen3.5-3B-Instruct"
      else:
          return config.get("llm_model", "Qwen2.5-0.5B")
  ```

- **`clarityocr/db.py`** (add column)
  - `Job.polish_model_used` (string, nullable) — track which model was used

#### API contract changes:

None (internal routing only)

#### Acceptance criteria:

1. Set `POLISH_CANARY_PCT=10` → ~10% of jobs use Qwen 3.5
2. `GET /api/v2/jobs/{job_id}` shows `polish_model_used: "Qwen3.5-3B-Instruct"`
3. Structured logs contain `polish_model` field
4. After 100 jobs, compare quality metrics (manual spot-check)

**Complexity:** S (deterministic routing + logging)

**Dependencies:** Phase 3.1 (evaluation must show Qwen 3.5 is better)

**Risk:** Low — canary is gradual, deterministic rollback via env var

---

### 3.3 Cutover & vLLM Config

**Current state:** vLLM config is external (not managed by ClarityOCR).

**Change:** Full cutover to Qwen 3.5 + document vLLM setup.

#### Files to modify:

- **`clarityocr/polish.py`** (change default)
  ```python
  model = config.get("llm_model", os.getenv("V2_LLM_MODEL", "Qwen3.5-3B-Instruct"))
  ```

- **New file:** `docs/VLLM_SETUP.md`
  ```markdown
  # vLLM Setup for ClarityOCR Polish Lane

  ## Recommended Model: Qwen3.5-3B-Instruct

  ### Installation
  ```bash
  pip install vllm
  vllm serve Qwen/Qwen3.5-3B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 8192
  ```

  ### Docker Compose (Production)
  ```yaml
  services:
    vllm:
      image: vllm/vllm-openai:latest
      command: ["--model", "Qwen/Qwen3.5-3B-Instruct", "--port", "8000"]
      ports:
        - "8000:8000"
      deploy:
        resources:
          reservations:
            devices:
              - driver: nvidia
                count: 1
                capabilities: [gpu]
  ```
  ```

- **`docker-compose.local.yml`** (add vLLM service)
  ```yaml
  services:
    vllm:
      image: vllm/vllm-openai:latest
      command: ["--model", "Qwen/Qwen3.5-3B-Instruct", "--port", "8000"]
      ports:
        - "8000:8000"
      deploy:
        resources:
          reservations:
            devices:
              - driver: nvidia
                count: 1
                capabilities: [gpu]

    clarityocr:
      environment:
        - V2_LLM_BASE_URL=http://vllm:8000/v1
        - V2_LLM_MODEL=Qwen3.5-3B-Instruct
      depends_on:
        - vllm
  ```

#### API contract changes:

None

#### Acceptance criteria:

1. Fresh install uses Qwen 3.5 by default
2. vLLM service starts successfully in Docker Compose
3. Polish lane connects to vLLM and processes jobs
4. Quality gate: manual spot-check of 10 polished outputs shows no regressions vs Phase 3.1 evaluation

**Complexity:** M (vLLM integration + Docker config)

**Dependencies:** Phase 3.2 (canary must show no issues)

**Risk:** Medium — new model could have edge cases; mitigation = canary + rollback via env var

---

## Phase 4: Pipeline Hardening

**Goal:** Error taxonomy, quality gates, graceful degradation, resource limits.

### 4.1 Error Taxonomy

**Current state:** Generic `E_INTERNAL` code, no distinction between transient/fatal errors.

**Change:** Structured error codes with retry policies.

#### Files to create:

- **New file:** `clarityocr/errors.py`
  ```python
  from enum import Enum

  class ErrorCode(Enum):
      # Transient (retry)
      E_NETWORK_TIMEOUT = "E_NETWORK_TIMEOUT"
      E_VRAM_OOM = "E_VRAM_OOM"
      E_LLM_TIMEOUT = "E_LLM_TIMEOUT"

      # Fatal (do not retry)
      E_INVALID_PDF = "E_INVALID_PDF"
      E_CORRUPTED_FILE = "E_CORRUPTED_FILE"
      E_EMPTY_OCR_OUTPUT = "E_EMPTY_OCR_OUTPUT"

      # System errors
      E_LEASE_TIMEOUT = "E_LEASE_TIMEOUT"
      E_INTERNAL = "E_INTERNAL"

  RETRY_POLICY = {
      ErrorCode.E_NETWORK_TIMEOUT: {"max_attempts": 3, "backoff_sec": 5},
      ErrorCode.E_VRAM_OOM: {"max_attempts": 2, "backoff_sec": 30},
      ErrorCode.E_LLM_TIMEOUT: {"max_attempts": 2, "backoff_sec": 10},

      ErrorCode.E_INVALID_PDF: {"max_attempts": 1, "backoff_sec": 0},
      ErrorCode.E_CORRUPTED_FILE: {"max_attempts": 1, "backoff_sec": 0},
      ErrorCode.E_EMPTY_OCR_OUTPUT: {"max_attempts": 1, "backoff_sec": 0},

      ErrorCode.E_LEASE_TIMEOUT: {"max_attempts": 3, "backoff_sec": 0},
      ErrorCode.E_INTERNAL: {"max_attempts": 2, "backoff_sec": 5},
  }
  ```

#### Files to modify:

- **`clarityocr/pipeline_v2.py`** (modify error handling)
  - Catch specific exceptions → map to `ErrorCode`
  - Example:
    ```python
    except subprocess.TimeoutExpired:
        raise PipelineError(ErrorCode.E_NETWORK_TIMEOUT, "Converter timed out")
    except FileNotFoundError:
        raise PipelineError(ErrorCode.E_INVALID_PDF, "PDF not found or corrupted")
    ```

- **`clarityocr/job_manager.py`** (modify retry logic)
  - Use `RETRY_POLICY` to decide whether to requeue
  - Set `JobFile.max_attempts` dynamically based on error code

#### API contract changes:

- **`GET /api/v2/jobs/{job_id}/files`** (add field)
  ```json
  {
    "last_error_code": "E_VRAM_OOM",
    "last_error_message": "CUDA out of memory",
    "retry_policy": {"max_attempts": 2, "backoff_sec": 30}
  }
  ```

#### Acceptance criteria:

1. Transient error → file requeues up to `max_attempts`
2. Fatal error → file fails immediately, no retries
3. Error codes appear in structured logs and API responses
4. Operator can filter errors by code in dashboard

**Complexity:** M (new error hierarchy + retry plumbing)

**Dependencies:** None

**Risk:** Low — improves existing error handling, no breaking changes

---

### 4.2 Quality Gates

**Current state:** Empty OCR output check exists (`_assert_non_empty_ocr_text`), but no other quality checks.

**Change:** Per-stage quality gates with configurable thresholds.

#### Files to create:

- **New file:** `clarityocr/quality_gates.py`
  ```python
  from dataclasses import dataclass

  @dataclass
  class QualityGate:
      name: str
      threshold: float
      failure_mode: str  # "warn" | "fail"

  OCR_GATES = [
      QualityGate("ocr_confidence", threshold=0.7, failure_mode="warn"),
      QualityGate("empty_output", threshold=1.0, failure_mode="fail"),
  ]

  POLISH_GATES = [
      QualityGate("polish_diff_ratio", threshold=0.3, failure_mode="warn"),
      QualityGate("polish_length_ratio", threshold=2.0, failure_mode="fail"),
  ]

  def check_gate(gate: QualityGate, value: float) -> tuple[bool, str]:
      if gate.name == "ocr_confidence" and value < gate.threshold:
          return False, f"OCR confidence {value:.2f} below threshold {gate.threshold}"
      # ... more gate logic
      return True, ""
  ```

#### Files to modify:

- **`clarityocr/pipeline_v2.py`** (add gate checks)
  ```python
  # After OCR
  confidence = _estimate_ocr_confidence(md_text)
  for gate in OCR_GATES:
      passed, msg = check_gate(gate, confidence)
      if not passed:
          if gate.failure_mode == "fail":
              raise PipelineError(ErrorCode.E_LOW_OCR_CONFIDENCE, msg)
          else:
              append_event(session, job_id, "quality_warning", payload={"gate": gate.name, "message": msg})

  # After polish
  diff_ratio = len(polished) / len(original)
  for gate in POLISH_GATES:
      # ... similar logic
  ```

#### API contract changes:

None (quality warnings appear as events)

#### Acceptance criteria:

1. Low OCR confidence → warning event logged, job continues
2. Polish output 3x longer than input → job fails with `E_POLISH_HALLUCINATION`
3. Quality gate failures visible in `/api/v2/jobs/{job_id}/events`
4. Configurable thresholds via env vars (e.g., `QG_OCR_CONFIDENCE_MIN=0.7`)

**Complexity:** M (gate framework + instrumentation)

**Dependencies:** None

**Risk:** Low — gates are additive, no breaking changes

---

### 4.3 Graceful Degradation

**Current state:** Circuit breaker for polish exists, but polish is required for `ocr_plus_polish` mode.

**Change:** If vLLM down, skip polish but still deliver raw OCR.

#### Files to modify:

- **`clarityocr/pipeline_v2.py`** (modify polish handling)
  ```python
  if mode == "ocr_plus_polish":
      if not _polish_breaker.allow_request():
          append_event(session, job_id, "polish_skipped_circuit_open")
          # Continue without polish
      else:
          try:
              polish_result = _try_optional_polish(...)
          except Exception as e:
              append_event(session, job_id, "polish_failed_graceful", payload={"error": str(e)})
              # Continue without polish
  ```

- **`clarityocr/db.py`** (add field)
  - `Artifact.degraded` (boolean, default `False`) — mark artifacts produced without polish

#### API contract changes:

- **`GET /api/v2/jobs/{job_id}/artifacts`** (add field)
  ```json
  {
    "type": "md",
    "path": "output/job123/file456/doc.md",
    "degraded": true,
    "degradation_reason": "polish_circuit_open"
  }
  ```

#### Acceptance criteria:

1. vLLM down → polish skipped, job completes with `degraded: true`
2. Job status shows warning: `"completed_with_degradation"`
3. Operator can filter degraded artifacts in dashboard
4. Circuit breaker recovers when vLLM comes back online

**Complexity:** S (existing circuit breaker + degradation flag)

**Dependencies:** None

**Risk:** Very low — improves availability

---

### 4.4 Resource Limits

**Current state:** No per-job limits; large PDFs can consume unbounded memory/time.

**Change:** Configurable limits with enforcement.

#### Files to create:

- **New file:** `clarityocr/resource_limits.py`
  ```python
  from dataclasses import dataclass

  @dataclass
  class ResourceLimits:
      max_pages_per_file: int = 500
      max_file_size_mb: int = 512  # Already exists in upload
      max_processing_time_sec: int = 3600  # 1 hour
      max_memory_mb: int = 8192

  def enforce_limits(job_id: str, file_path: str, limits: ResourceLimits):
      # Check file size (already done in upload)
      # Check page count
      page_count = _count_pdf_pages(file_path)
      if page_count > limits.max_pages_per_file:
          raise ResourceLimitError(f"File has {page_count} pages, limit is {limits.max_pages_per_file}")
  ```

#### Files to modify:

- **`clarityocr/pipeline_v2.py`** (add timeout wrapper)
  ```python
  import signal

  def _with_timeout(func, timeout_sec):
      def handler(signum, frame):
          raise TimeoutError("Processing exceeded time limit")

      signal.signal(signal.SIGALRM, handler)
      signal.alarm(timeout_sec)
      try:
          return func()
      finally:
          signal.alarm(0)

  # In _process_file()
  result = _with_timeout(lambda: subprocess.run(...), timeout=limits.max_processing_time_sec)
  ```

#### API contract changes:

None (errors appear as `E_RESOURCE_LIMIT_EXCEEDED`)

#### Acceptance criteria:

1. Upload 1000-page PDF → rejected with `413 Payload Too Large`
2. Processing exceeds 1 hour → job fails with `E_TIMEOUT`
3. Limits configurable via env vars (`LIMIT_MAX_PAGES=500`)

**Complexity:** S (simple guards + timeout wrapper)

**Dependencies:** None

**Risk:** Low — protects system from resource exhaustion

---

## Phase 5: Deploy & Drift Discipline

**Goal:** Automated canon/runtime reconciliation, CI smoke tests, rollback procedure.

### 5.1 Canon/Runtime Reconciliation

**Current state:** No automated checks for config drift.

**Change:** Script that validates runtime matches expected config.

#### Files to create:

- **New script:** `scripts/check_canon.sh`
  ```bash
  #!/bin/bash
  set -e

  echo "Checking ClarityOCR canon..."

  # Check DB schema matches models
  python3 -c "from clarityocr.db import setup_db; setup_db('/tmp/test.db')" || {
      echo "ERROR: DB schema broken"
      exit 1
  }

  # Check vLLM reachable
  curl -f http://localhost:8000/health || {
      echo "WARNING: vLLM not reachable"
  }

  # Check API responds
  curl -f http://localhost:8008/api/health || {
      echo "ERROR: API not responding"
      exit 1
  }

  # Check worker count matches config
  expected_workers=$(grep V2_WORKERS docker-compose.local.yml | awk '{print $2}')
  # ... more checks

  echo "✓ Canon check passed"
  ```

#### Files to modify:

- **`.github/workflows/canon-check.yml`** (new)
  ```yaml
  name: Canon Check
  on: [push, pull_request]
  jobs:
    check:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v3
        - run: ./scripts/check_canon.sh
  ```

#### API contract changes:

None

#### Acceptance criteria:

1. Script runs successfully on fresh install
2. Script detects DB schema drift (add column manually, run script → fails)
3. CI fails if canon check fails

**Complexity:** S (shell script + CI config)

**Dependencies:** None

**Risk:** Very low — CI validation only

---

### 5.2 Docker Image Versioning

**Current state:** `docker-compose.local.yml` builds `clarityocr:local` with no version tags.

**Change:** Semantic versioning + registry push.

#### Files to modify:

- **New file:** `VERSION`
  ```
  1.0.0
  ```

- **`Dockerfile`** (add label)
  ```dockerfile
  ARG VERSION=dev
  LABEL version="${VERSION}"
  ```

- **New script:** `scripts/build_and_tag.sh`
  ```bash
  #!/bin/bash
  VERSION=$(cat VERSION)
  docker build -t clarityocr:${VERSION} -t clarityocr:latest --build-arg VERSION=${VERSION} .
  docker tag clarityocr:${VERSION} registry.example.com/clarityocr:${VERSION}
  docker push registry.example.com/clarityocr:${VERSION}
  ```

- **`docker-compose.local.yml`** (modify)
  ```yaml
  services:
    clarityocr:
      image: clarityocr:${VERSION:-latest}
      # Remove build section (use pre-built images)
  ```

#### API contract changes:

- **`GET /api/health`** (add field)
  ```json
  {
    "status": "ok",
    "api": "v2",
    "version": "1.0.0"
  }
  ```

#### Acceptance criteria:

1. Build script tags image with version from `VERSION` file
2. API returns version in health endpoint
3. Rollback = change `VERSION` env var in docker-compose, restart

**Complexity:** S (versioning + build script)

**Dependencies:** None

**Risk:** Very low

---

### 5.3 Rollback Procedure

**Current state:** No documented rollback process.

**Change:** Runbook for rollback.

#### Files to create:

- **New file:** `docs/ROLLBACK.md`
  ```markdown
  # Rollback Procedure

  ## Quick Rollback (Docker)

  1. Stop current version:
     ```bash
     docker-compose down
     ```

  2. Change version in `docker-compose.yml`:
     ```yaml
     services:
       clarityocr:
         image: clarityocr:1.0.0  # Change to previous version
     ```

  3. Restart:
     ```bash
     docker-compose up -d
     ```

  ## Database Rollback

  ClarityOCR uses SQLite with additive schema changes only.
  Rollback = restore DB from backup:

  ```bash
  cp ~/.clarityocr/clarity_v2.db.backup ~/.clarityocr/clarity_v2.db
  ```

  ## Config Rollback

  Version-specific config stored in `config/{version}/`.
  Restore old config:

  ```bash
  cp config/1.0.0/openclaw.json config/openclaw.json
  ```
  ```

#### API contract changes:

None

#### Acceptance criteria:

1. Follow rollback procedure → previous version starts successfully
2. Jobs created on new version are visible on old version (backward compatibility)
3. No data loss during rollback

**Complexity:** S (documentation only)

**Dependencies:** Phase 5.2 (versioning)

**Risk:** Very low

---

### 5.4 CI Smoke Tests

**Current state:** No automated tests for deployment.

**Change:** End-to-end smoke test in CI.

#### Files to create:

- **New file:** `.github/workflows/smoke-test.yml`
  ```yaml
  name: Smoke Test
  on: [push, pull_request]
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v3

        - name: Start services
          run: docker-compose -f docker-compose.test.yml up -d

        - name: Wait for API
          run: |
            for i in {1..30}; do
              curl -f http://localhost:8008/api/health && break
              sleep 2
            done

        - name: Run smoke test
          run: |
            python3 scripts/smoke_test.py

        - name: Teardown
          run: docker-compose -f docker-compose.test.yml down
  ```

- **New script:** `scripts/smoke_test.py`
  ```python
  import requests
  from pathlib import Path

  API_BASE = "http://localhost:8008/api/v2"

  # 1. Upload test file
  with open("test_data/sample.pdf", "rb") as f:
      resp = requests.post(f"{API_BASE}/uploads", files={"files": f})
      assert resp.status_code == 201
      upload_id = resp.json()["upload_id"]

  # 2. Start job
  resp = requests.post(f"{API_BASE}/jobs", json={
      "client_id": "smoke-test",
      "client_request_id": "test-1",
      "inputs": resp.json()["inputs"],
      "mode": "ocr_only",
      "preset": "speed"
  })
  assert resp.status_code == 202
  job_id = resp.json()["job_id"]

  # 3. Poll until completion
  import time
  for _ in range(60):
      resp = requests.get(f"{API_BASE}/jobs/{job_id}")
      status = resp.json()["status"]
      if status in ["completed", "failed"]:
          break
      time.sleep(2)

  assert status == "completed", f"Job failed: {status}"
  print("✓ Smoke test passed")
  ```

#### API contract changes:

None

#### Acceptance criteria:

1. CI runs smoke test on every push
2. Smoke test fails if API broken
3. Smoke test fails if job processing broken

**Complexity:** M (Docker test setup + smoke test script)

**Dependencies:** None

**Risk:** Low — CI-only, no production impact

---

## Summary Table

| Phase | Deliverable | Complexity | Dependencies | Risk | Value |
|-------|------------|------------|--------------|------|-------|
| 1.1 | Job stage model | M | None | Low | ★★★★★ |
| 1.2 | Structured logging | S | None | Very Low | ★★★★☆ |
| 1.3 | Health endpoint | S | None | Low | ★★★☆☆ |
| 2.1 | Progress dashboard | M | 1.1 | Low | ★★★★★ |
| 2.2 | Error display | S | None | Very Low | ★★★★☆ |
| 2.3 | Job queue overview | M | None | Very Low | ★★★☆☆ |
| 3.1 | Qwen 3.5 eval | S | None | Very Low | ★★★☆☆ |
| 3.2 | A/B migration | S | 3.1 | Low | ★★★☆☆ |
| 3.3 | Cutover + vLLM config | M | 3.2 | Medium | ★★★★☆ |
| 4.1 | Error taxonomy | M | None | Low | ★★★★☆ |
| 4.2 | Quality gates | M | None | Low | ★★★☆☆ |
| 4.3 | Graceful degradation | S | None | Very Low | ★★★★☆ |
| 4.4 | Resource limits | S | None | Low | ★★★☆☆ |
| 5.1 | Canon check | S | None | Very Low | ★★☆☆☆ |
| 5.2 | Docker versioning | S | None | Very Low | ★★★☆☆ |
| 5.3 | Rollback procedure | S | 5.2 | Very Low | ★★★☆☆ |
| 5.4 | CI smoke tests | M | None | Low | ★★★☆☆ |

**Complexity:**
- S (Small): 1-3 days
- M (Medium): 3-7 days
- L (Large): 7+ days

---

## Brutal Honesty: Current Weaknesses

1. **No progress visibility is operator-hostile.** Jobs are black boxes. Fix: Phase 1.
2. **Web UI is a toy.** Upload form only, no monitoring. Fix: Phase 2.
3. **Qwen2.5-0.5B is stale.** Qwen 3.5 exists and is better. Fix: Phase 3.
4. **Error handling is primitive.** No distinction between "retry" and "give up." Fix: Phase 4.1.
5. **No quality gates.** Empty output check exists, but no OCR confidence thresholds. Fix: Phase 4.2.
6. **Polish is fragile.** If vLLM down, job fails instead of degrading gracefully. Fix: Phase 4.3.
7. **No resource limits.** 10,000-page PDF will kill the worker. Fix: Phase 4.4.
8. **No CI.** Deploy = hope it works. Fix: Phase 5.4.
9. **No drift checks.** Runtime can diverge from config silently. Fix: Phase 5.1.
10. **No rollback plan.** Break prod = panic. Fix: Phase 5.3.

---

## Recommended Implementation Order

**Sprint 1 (Weeks 1-2):** Phase 1 (Observability)
→ Unblocks operator visibility

**Sprint 2 (Weeks 3-4):** Phase 2 (Operator UI)
→ Makes observability actionable

**Sprint 3 (Weeks 5-6):** Phase 3 (Qwen 3.5 upgrade)
→ Quality improvement, low risk with canary

**Sprint 4 (Weeks 7-8):** Phase 4 (Hardening)
→ Production readiness

**Sprint 5 (Week 9):** Phase 5 (CI/CD discipline)
→ Sustaining engineering

**Total:** ~9 weeks for full enterprise hardening

---

**End of Plan**
