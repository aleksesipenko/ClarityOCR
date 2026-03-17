# ClarityOCR v2 Pipeline Fix - Deployment Guide

## Quick Start

1. **Backup original**
   ```bash
   cp clarityocr/pipeline_v2.py clarityocr/pipeline_v2.py.backup
   ```

2. **Apply fix**
   ```bash
   cp pipeline_v2_fixed.py clarityocr/pipeline_v2.py
   ```

3. **Restart worker**
   ```bash
   # If running in Docker:
   docker restart <container_id>
   
   # If running locally:
   # Kill the Python process and restart
   ```

---

## Pre-Deployment Checklist

- [ ] Read `BUG_ANALYSIS_AND_FIX.md` to understand all three bugs
- [ ] Verify you're replacing `pipeline_v2.py` (not `pipeline_v1.py`)
- [ ] Backup current database (`clarity_v2.db`) in case rollback needed
- [ ] Note current job queue status (number of running/queued jobs)
- [ ] Have a test PDF ready for quick validation

---

## Deployment Steps

### Option 1: Direct File Replacement (Simplest)

```bash
# In the ClarityOCR container or Python environment:
cd /app/clarityocr  # or wherever clarityocr/ is located

# Backup
cp pipeline_v2.py pipeline_v2.py.$(date +%Y%m%d_%H%M%S)

# Replace
cp /path/to/pipeline_v2_fixed.py ./pipeline_v2.py

# Verify syntax
python3 -m py_compile pipeline_v2.py
echo "✓ Syntax OK"

# Restart worker process
pkill -f "clarityocr.pipeline_v2" || true
# API will auto-spawn new workers on next request
```

### Option 2: Docker In-Container Replace

```bash
# Copy file into running container
docker cp pipeline_v2_fixed.py <container>:/app/clarityocr/pipeline_v2.py

# Verify and restart
docker exec <container> python3 -m py_compile /app/clarityocr/pipeline_v2.py
docker restart <container>
```

### Option 3: Gradual Rollout (Recommended for Production)

If you have multiple workers:

```bash
# Replace on worker 1 only
docker cp pipeline_v2_fixed.py worker1:/app/clarityocr/pipeline_v2.py
docker restart worker1

# Monitor for 30 minutes (watch job completion rates)
# If stable, apply to remaining workers
docker cp pipeline_v2_fixed.py worker2:/app/clarityocr/pipeline_v2.py
docker restart worker2
```

---

## Post-Deployment Validation

### Test 1: Submit Small OCR Job

```bash
curl -X POST http://localhost:8008/api/ocr/submit \
  -H "Content-Type: application/json" \
  -d '{
    "client_request_id": "test_post_deploy_'$(date +%s)'",
    "files": ["/path/to/test.pdf"],
    "mode": "ocr_only",
    "preset": "speed"
  }'
```

### Test 2: Monitor Stage Transitions

```bash
# In another terminal, poll job status every 5 seconds:
JOB_ID="<job_id_from_above>"
while true; do
  curl -s http://localhost:8008/api/jobs/$JOB_ID | \
    jq '.files[0] | {status, stage, progress_pct}'
  sleep 5
done
```

**Expected output**:
```json
{
  "status": "running",
  "stage": "ocr",
  "progress_pct": 45
}
→ (after 30-60s)
{
  "status": "running",
  "stage": "ocr",
  "progress_pct": 90
}
→ (when OCR finishes)
{
  "status": "running",
  "stage": "polish",      ← FIXED: Now properly transitions!
  "progress_pct": 100
}
→ (when polish finishes)
{
  "status": "completed",  ← FIXED: Now reaches "completed"!
  "stage": "done",
  "progress_pct": 100
}
```

### Test 3: Verify Output Files Exist

```bash
# After job completes, check output directory:
ls -la /app/output_v2/$JOB_ID/$FILE_ID/

# Expected files:
# - document.md               (OCR markdown)
# - document.meta.json        (metadata)
# - document.naming.json      (naming suggestions)
# - batch_manifest.json       (checksums)

# If these exist, the fix is working! ✓
```

### Test 4: Parallel Job Test

```bash
# Submit 5 jobs quickly:
for i in {1..5}; do
  curl -X POST http://localhost:8008/api/ocr/submit \
    -H "Content-Type: application/json" \
    -d '{
      "client_request_id": "parallel_test_'$i'_'$(date +%s)'",
      "files": ["/path/to/test.pdf"],
      "mode": "ocr_only",
      "preset": "speed"
    }' &
done
wait

# Monitor all 5 jobs completing concurrently:
# With the fix, they should progress in parallel
# (not wait for each other)
```

---

## Rollback Procedure

If issues arise:

```bash
# Restore backup
cp clarityocr/pipeline_v2.py.backup clarityocr/pipeline_v2.py

# Restart worker
docker restart <container>

# Investigate logs:
docker logs <container> | tail -100 | grep -E "ERROR|Traceback"
```

---

## Expected Improvements

### Before Fix
- Jobs stuck at `stage=ocr, progress=99%`
- Status stays `running` forever
- Output files exist but API shows incomplete
- Multiple jobs serialize, queue backs up

### After Fix
- Jobs properly transition: `ocr` → `polish` → `done`
- Status properly reaches `completed`
- Multiple jobs can process concurrently
- Queue processes smoothly

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'concurrent.futures'"

**Cause**: Running on Python 2.x (should be 3.3+)

**Fix**: Ensure Python 3.3+ is used (concurrent.futures is built-in)

```bash
python3 --version  # Should be >= 3.3
```

### Issue: Jobs still showing `status=running` after 10 minutes

**Cause**: Either:
1. File replacement didn't take effect (process still running old code)
2. Database still has old records

**Fix**:
```bash
# Kill ALL Python processes
pkill -9 -f "clarityocr"

# Wait 5s, then restart API/worker
# This forces full reload

# Or manually check process:
ps aux | grep "clarityocr"
```

### Issue: "TypeError: _stage_context() got unexpected keyword argument 'next_stage'"

**Cause**: File replacement failed, still running old version

**Fix**: Verify the file was actually replaced:
```bash
grep "next_stage" clarityocr/pipeline_v2.py
# Should see: "def _stage_context(..., next_stage: Optional[str] = None):"
```

---

## Monitoring Queries (after deployment)

### Check job completion rate

```sql
-- Via SQLite (if you have direct DB access):
SELECT 
  status,
  COUNT(*) as count,
  AVG(duration_ms) as avg_duration_ms
FROM job_files
WHERE created_at > datetime('now', '-1 hour')
GROUP BY status
ORDER BY count DESC;

-- Expected: "completed" should be dominant
```

### Check for stuck jobs

```sql
SELECT job_id, status, stage, progress_pct, created_at
FROM (
  SELECT f.job_id, f.status, f.stage, f.progress_pct, 
         j.created_at,
         (strftime('%s', 'now') - strftime('%s', j.created_at)) as age_sec
  FROM job_files f
  JOIN jobs j ON f.job_id = j.job_id
)
WHERE age_sec > 600  -- older than 10 minutes
AND status = 'running';

-- Expected: Should be empty (or very few in progress)
```

---

## Support

If you encounter issues:

1. Check `docker logs <container>` for error messages
2. Consult `BUG_ANALYSIS_AND_FIX.md` for technical details
3. Compare line-by-line with `PATCH.diff` to verify changes
4. Review the `pipeline_v2_fixed.py` file to ensure syntax is correct

---

## Performance Notes

The fix includes **optional parallel job support** via `ThreadPoolExecutor`:

```python
start_workers(num_workers=2, max_concurrent_jobs=3)
#                               ↑ NEW PARAMETER
```

This allows:
- 2 worker threads
- Each can handle up to 3 concurrent I/O operations
- Total of 6 concurrent polish subprocesses

**Recommendation**: Start with `max_concurrent_jobs=3` and monitor memory usage. Increase if you have RAM headroom and many I/O-bound jobs.

---

## FAQ

**Q: Will this affect running jobs?**
A: Yes - when you restart the worker, running jobs will be re-acquired from the queue. They won't lose progress (thanks to the `_recover_stale_files` mechanism).

**Q: Do I need to migrate the database?**
A: No - the database schema is unchanged. This is a pure code fix.

**Q: Will this break any API responses?**
A: No - all API contracts are preserved. The stage transitions are now more accurate, which is a user-facing improvement.

**Q: How long does a typical OCR job take?**
A: 30-120 seconds depending on file size and preset. With the fix, you'll see accurate progress within this window.

---

## Version Check

After deployment, you can verify the fix is active:

```bash
grep -n "next_stage" clarityocr/pipeline_v2.py | head -3
# Should output:
# 56:                   next_stage: Optional[str] = None):
# 78:            if next_stage:
# 79:                set_file_stage(session, file_id, next_stage, progress_pct=100)
```

If these lines exist, the fix is deployed successfully! ✓
