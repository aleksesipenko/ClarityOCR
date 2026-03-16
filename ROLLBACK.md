# ClarityOCR Rollback Runbook (Phase 5.3)

## When to roll back

- ClarityOCR API returns 5xx on >5% of requests for >2 minutes
- OCR output quality visibly degraded (blank outputs, garbage text)
- Container keeps restarting (`docker ps` shows `Restarting`)
- DB migrations failed or data is corrupt

---

## Quick Rollback (Alex-PC Docker — WSL)

### 1. Check current state

```bash
docker ps -a --filter name=clarityocr
docker logs --tail=50 clarityocr
```

### 2. Roll back to previous image tag

List available tags:
```bash
docker images clarityocr --format "table {{.Tag}}\t{{.CreatedAt}}\t{{.ID}}"
```

Roll back to a specific tag (e.g. `abc1234`):
```bash
docker stop clarityocr && docker rm clarityocr
cd /path/to/ClarityOCR
IMAGE_TAG=abc1234 docker compose up -d clarityocr
```

Or manually:
```bash
docker run -d \
  --name clarityocr \
  --gpus all \
  -p 8008:8008 \
  -v clarityocr_models:/root/.cache/huggingface \
  -v clarityocr_data:/root/.clarityocr \
  -e CLARITY_HOST=0.0.0.0 \
  -e CLARITY_PORT=8008 \
  clarityocr:PREVIOUS_TAG
```

### 3. Verify rollback

```bash
curl http://localhost:8008/api/v2/health/live
curl http://localhost:8008/api/v2/health/ready
curl http://localhost:8008/api/v2/version
```

---

## Full Rollback (git + rebuild)

```bash
cd /path/to/ClarityOCR
git log --oneline -10                         # find last good commit
git checkout <last-good-sha>                  # detach to known good
docker build --build-arg VERSION=<sha> -t clarityocr:<sha> -t clarityocr:latest .
docker compose down clarityocr
docker compose up -d clarityocr
```

---

## DB Rollback

ClarityOCR uses SQLite. The database file is at `/root/.clarityocr/clarityocr_v2.sqlite` (inside `clarityocr_data` volume).

Before any migration:
```bash
# Copy DB from volume to host
docker run --rm \
  -v clarityocr_data:/data \
  -v $(pwd):/backup \
  alpine cp /data/clarityocr_v2.sqlite /backup/clarityocr_v2.sqlite.bak
```

To restore:
```bash
docker run --rm \
  -v clarityocr_data:/data \
  -v $(pwd):/backup \
  alpine cp /backup/clarityocr_v2.sqlite.bak /data/clarityocr_v2.sqlite
```

---

## Canary Rollback

If canary routing causes issues:

```bash
# Disable canary immediately (no rebuild needed)
docker exec clarityocr env V2_LLM_MODEL_CANARY="" V2_CANARY_PCT=0 clarityocr serve ...
# OR restart with override:
V2_LLM_MODEL_CANARY="" V2_CANARY_PCT=0 docker compose up -d clarityocr
```

---

## Escalation

If none of the above works:
1. Stop the container: `docker stop clarityocr`
2. Check `docker logs clarityocr` for the root cause
3. Open a rebuild from the last known-good git tag
4. Notify Alex via Telegram
