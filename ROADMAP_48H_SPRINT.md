# ClarityOCR Enterprise Upgrade — 48-Hour Sprint Roadmap

**Owner:** Maya
**Start:** 2026-03-16 05:42 MSK
**Deadline:** 2026-03-18 05:42 MSK (hard)
**Exit criteria:** all changes deployed to Alex-PC Docker, smoke tested, integrated with Alfred + our contour

---

## Sprint Blocks

### Block 1: Phase 1 — Progress & Observability (hours 0–10)
**Target:** 2026-03-16 05:42 → 15:42 MSK

| # | Deliverable | Est | Status |
|---|------------|-----|--------|
| 1.1 | Job stage model (DB + pipeline instrumentation) | 3h | ✅ done (a9f19ce) |
| 1.2 | Structured JSON logging | 2h | ✅ done (4e92e07) |
| 1.3 | Health endpoint with real probes | 1h | ✅ done (e11efcc) |
| 1.4 | Progress/ETA in API response | 2h | ✅ done (e11efcc) |
| 1.5 | Local smoke test | 1h | ✅ imports ok, deps need runtime |

**Checkpoint:** `/api/v2/jobs/{id}` returns stage, progress_pct, eta_seconds. Structured logs emit JSON.

---

### Block 2: Phase 4 — Pipeline Hardening (hours 10–18)
**Target:** 2026-03-16 15:42 → 23:42 MSK

| # | Deliverable | Est | Status |
|---|------------|-----|--------|
| 4.1 | Error taxonomy (`errors.py` + retry policy) | 3h | ✅ done (994f046) |
| 4.2 | Quality gates (OCR confidence, polish hallucination) | 2h | ✅ done (12a566b) |
| 4.3 | Graceful degradation (vLLM down → raw OCR) | 1h | ✅ done (0415723) |
| 4.4 | Resource limits (pages, time, memory) | 1h | ✅ done (64ed0e2) |
| 4.5 | Local smoke test | 1h | ✅ integrated (e09ae0a) |

**Checkpoint:** error codes in API, quality warnings in events, graceful degradation verified.

---

### Block 3: Phase 2 — Operator UI (hours 18–26)
**Target:** 2026-03-16 23:42 → 2026-03-17 07:42 MSK

| # | Deliverable | Est | Status |
|---|------------|-----|--------|
| 2.1 | Progress dashboard (live bars, stage timeline) | 3h | todo |
| 2.2 | Error display with retry button | 2h | todo |
| 2.3 | Job queue overview page | 2h | todo |
| 2.4 | Local smoke test | 1h | todo |

**Checkpoint:** web UI shows live progress, error cards, queue overview.

---

### Block 4: Phase 3 + 5 — Model Upgrade + Deploy Discipline (hours 26–34)
**Target:** 2026-03-17 07:42 → 15:42 MSK

| # | Deliverable | Est | Status |
|---|------------|-----|--------|
| 3.1 | Qwen 3.5 eval script + test data | 2h | todo |
| 3.2 | A/B canary routing | 1h | todo |
| 3.3 | vLLM config + docker-compose update | 1h | todo |
| 5.1 | Canon check script | 1h | todo |
| 5.2 | Docker versioning | 1h | todo |
| 5.3 | Rollback runbook | 1h | todo |
| 5.4 | CI smoke test | 1h | todo |

**Checkpoint:** canary routing works, version tagging works, canon check green.

---

### Block 5: Integration — Alfred + Contour (hours 34–42)
**Target:** 2026-03-17 15:42 → 23:42 MSK

| # | Deliverable | Est | Status |
|---|------------|-----|--------|
| I.1 | Alfred VPS workspace: update ClarityOCR integration refs | 2h | todo |
| I.2 | Alfred template: update OCR pipeline config for new API | 2h | todo |
| I.3 | Our contour: update canon docs, STATE.json, build queue | 1h | todo |
| I.4 | Deploy to Alex-PC Docker (build, push, restart) | 2h | todo |
| I.5 | Integration smoke | 1h | todo |

**Checkpoint:** Alfred can hit new ClarityOCR API, our docs reflect reality.

---

### Block 6: Final Verification (hours 42–48)
**Target:** 2026-03-17 23:42 → 2026-03-18 05:42 MSK

| # | Deliverable | Est | Status |
|---|------------|-----|--------|
| V.1 | Full e2e smoke on Alex-PC Docker (upload→OCR→progress→done) | 2h | todo |
| V.2 | Alfred e2e: document via Alfred → ClarityOCR → result | 1h | todo |
| V.3 | Health/observability verification | 1h | todo |
| V.4 | Final report + owner notification | 1h | todo |
| V.5 | Buffer for fixes | 1h | todo |

**Exit:** everything runs in Docker on Alex-PC, Alfred integration proven, owner gets report.

---

## Cron Schedule

Maya builder cron drives the sprint:
- **Every 2 hours:** check progress, continue next deliverable, unblock if stuck
- **At block boundaries (hours 10, 18, 26, 34, 42):** milestone check, status report if blocked
- **Hour 46:** mandatory final smoke test pass
- **Hour 48:** owner report

## Risk Mitigations

1. **Alex-PC offline:** all code work happens on local clone first; deploy is Block 5
2. **Claude Code slow/fails:** Maya can implement directly for S-complexity items
3. **vLLM not available for Qwen 3.5:** Phase 3 ships canary routing code, model swap is config-only
4. **Time overrun on a phase:** cut scope to must-haves, defer nice-to-haves

## Integration Surfaces

| System | What changes | Where |
|--------|-------------|-------|
| ClarityOCR runtime | all 5 phases | Alex-PC Docker |
| Alfred VPS | OCR integration config | `/root/cronos-workspace/` |
| Alfred template | pipeline config | `projects/alfred-template/` |
| Our contour | canon docs, STATE | `docs/clarityocr/`, `docs/team/STATE.json` |

---

**Let's ship.**
