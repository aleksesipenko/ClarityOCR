# ClarityOCR v2 Fix - Verification Checklist

Use this checklist to verify the fix is correctly applied and working.

---

## Pre-Deployment Verification

### Code Quality Check
- [ ] `pipeline_v2_fixed.py` has no syntax errors
  ```bash
  python3 -m py_compile pipeline_v2_fixed.py
  # Should complete without output
  ```

- [ ] File size is reasonable (~50KB)
  ```bash
  wc -l pipeline_v2_fixed.py
  # Should be ~1200 lines
  ```

- [ ] All three bugs are fixed in the code
  ```bash
  # Check for next_stage parameter
  grep "next_stage: Optional\[str\]" pipeline_v2_fixed.py | wc -l
  # Should return 1 (function definition)
  
  # Check for proper stage transitions
  grep "next_stage=" pipeline_v2_fixed.py | wc -l
  # Should return at least 2 (ocr→polish, polish→done)
  
  # Check for ThreadPoolExecutor
  grep "ThreadPoolExecutor" pipeline_v2_fixed.py | wc -l
  # Should return at least 1
  ```

### Database Check
- [ ] Backup database exists
  ```bash
  ls -lh ~/.clarityocr/clarity_v2.db.backup
  # Should exist and be recent
  ```

- [ ] Current database is readable
  ```bash
  sqlite3 ~/.clarityocr/clarity_v2.db "SELECT COUNT(*) FROM jobs;"
  # Should return a number
  ```

---

## Post-Deployment Verification

### Step 1: Verify File Replacement
```bash
# Confirm the new file is in place
md5sum clarityocr/pipeline_v2.py
# Record this hash

# Check for the fix marker (next_stage parameter)
grep -n "next_stage: Optional" clarityocr/pipeline_v2.py
# Should find: "def _stage_context(..., next_stage: Optional[str] = None):"
```

**Expected**: The file contains the `next_stage` parameter.

### Step 2: Start Worker and Submit Test Job
```bash
# Start API/worker
# (how depends on your setup)

# Submit a test job
JOB_RESPONSE=$(curl -s -X POST http://localhost:8008/api/ocr/submit \
  -H "Content-Type: application/json" \
  -d '{
    "client_request_id": "verification_test_'$(date +%s)'",
    "files": ["/path/to/test.pdf"],
    "mode": "ocr_only",
    "preset": "speed"
  }')

JOB_ID=$(echo "$JOB_RESPONSE" | jq -r '.job_id')
FILE_ID=$(echo "$JOB_RESPONSE" | jq -r '.files[0].id')

echo "Job ID: $JOB_ID"
echo "File ID: $FILE_ID"
```

**Expected**: API returns valid job_id and file_id (UUIDs).

### Step 3: Monitor Stage Transitions ⭐ CRITICAL
```bash
# Monitor file status in a loop
for i in {1..20}; do
  echo "=== Check $i ==="
  curl -s "http://localhost:8008/api/jobs/$JOB_ID/files/$FILE_ID" | \
    jq '{status: .status, stage: .stage, progress_pct: .stage_progress_pct}'
  sleep 5
done
```

**CRITICAL EXPECTATIONS**:

| Time | Status | Stage | Progress |
|------|--------|-------|----------|
| T=0-10s | running | ocr | 0-50% |
| T=10-30s | running | ocr | 50-100% |
| T=30-35s | **running** | **polish** | **0%** | ← **BUG #1 FIX**: Must transition here!
| T=35-40s | running | polish | 0-100% |
| T=40-45s | **completed** | **done** | **100%** | ← **BUG #2 FIX**: Must reach completed!

**If you see**:
- ✅ `stage` transitions from `ocr` → `polish` → `done` = **FIX IS WORKING**
- ✅ `status` reaches `completed` = **FIX IS WORKING**
- ❌ `stage` stays at `ocr` = **Fix not applied or old process still running**
- ❌ `status` stays at `running` = **Indentation bug not fixed**

### Step 4: Verify Output Files Exist
```bash
# After job completes, check output
OUTPUT_DIR="/app/output_v2/$JOB_ID/$FILE_ID"
ls -la "$OUTPUT_DIR"

# Expected files:
# - document.md (or {basename}.md)
# - document.meta.json
# - document.naming.json
# - batch_manifest.json

# Verify content
cat "$OUTPUT_DIR/batch_manifest.json" | jq .

# Should show all 3 artifacts with status="completed" and SHA256 hashes
```

**Expected**: All 4 files exist with non-empty content.

### Step 5: Test Parallel Jobs (Bug #3)
```bash
# Submit 3 jobs rapidly
JOB_IDS=()
for i in {1..3}; do
  RESP=$(curl -s -X POST http://localhost:8008/api/ocr/submit \
    -H "Content-Type: application/json" \
    -d '{
      "client_request_id": "parallel_'$i'_'$(date +%s)'",
      "files": ["/path/to/test.pdf"],
      "mode": "ocr_only",
      "preset": "speed"
    }')
  JID=$(echo "$RESP" | jq -r '.job_id')
  JOB_IDS+=("$JID")
  echo "Submitted job $i: $JID"
done

# Monitor all 3 jobs progressing concurrently
for job_id in "${JOB_IDS[@]}"; do
  while true; do
    STATUS=$(curl -s "http://localhost:8008/api/jobs/$job_id" | jq -r '.status')
    PROGRESS=$(curl -s "http://localhost:8008/api/jobs/$job_id/files" | jq -r '.[0].stage_progress_pct')
    echo "$job_id: $STATUS ($PROGRESS%)"
    
    if [ "$STATUS" = "completed" ]; then
      break
    fi
    sleep 10
  done
done
```

**Expected**: All 3 jobs should progress concurrently (not serialized one after another).

### Step 6: Check Logs for Errors
```bash
# Look for any error messages related to the fix
docker logs <container> 2>&1 | grep -E "ERROR|Traceback|ValueError" | tail -20

# Should be empty or unrelated to pipeline_v2
```

**Expected**: No errors related to the fix.

### Step 7: Database Verification
```bash
# Check that job records were updated correctly
sqlite3 ~/.clarityocr/clarity_v2.db << EOF
SELECT 
  j.job_id,
  f.status,
  f.stage,
  f.stage_progress_pct
FROM job_files f
JOIN jobs j ON f.job_id = j.job_id
ORDER BY j.created_at DESC
LIMIT 5;
EOF
```

**Expected**:
- `status` column shows "completed" (not "running")
- `stage` column shows "done" (not "ocr" or "polish")
- `stage_progress_pct` column shows 100

---

## Performance Baselines

Record these values BEFORE and AFTER to verify improvements:

### Before Fix
```bash
# Measure job completion rate
START_TIME=$(date +%s)
# ... submit 10 jobs ...
# ... wait for all to complete ...
END_TIME=$(date +%s)
echo "Time for 10 jobs: $((END_TIME - START_TIME)) seconds"

# Expected: ~60-120 seconds (serialized)
```

### After Fix
```bash
# Same test
echo "Time for 10 jobs: $((END_TIME - START_TIME)) seconds"

# Expected: ~30-60 seconds (parallelized)
# Should be ~2x faster due to concurrent I/O
```

---

## Regression Testing

### Test Each Mode
- [ ] `ocr_only` mode works end-to-end
- [ ] `ocr_plus_metadata` mode works
- [ ] `ocr_plus_polish` mode works
- [ ] Merge mode works (if applicable)

```bash
for mode in "ocr_only" "ocr_plus_metadata" "ocr_plus_polish"; do
  curl -s -X POST http://localhost:8008/api/ocr/submit \
    -H "Content-Type: application/json" \
    -d '{
      "client_request_id": "'$mode'_test_'$(date +%s)'",
      "files": ["/path/to/test.pdf"],
      "mode": "'$mode'",
      "preset": "speed"
    }' | jq '.status'
  # Should return "queued"
done
```

### Test Error Handling
- [ ] Invalid PDF fails gracefully (fallback converter)
- [ ] Timeout doesn't crash worker
- [ ] Circuit breaker doesn't crash system

```bash
# Test with invalid file
curl -s -X POST http://localhost:8008/api/ocr/submit \
  -H "Content-Type: application/json" \
  -d '{
    "client_request_id": "invalid_file_'$(date +%s)'",
    "files": ["/nonexistent/file.pdf"],
    "mode": "ocr_only",
    "preset": "speed"
  }'

# Should fail gracefully with error status (not crash)
```

---

## Sign-Off Checklist

- [ ] File replacement confirmed (grep shows `next_stage`)
- [ ] Worker restarted successfully
- [ ] Test job submitted and tracked
- [ ] Stage transitions observed: `ocr` → `polish` → `done`
- [ ] Status reached `completed` (not stuck at `running`)
- [ ] Output files exist in correct directory
- [ ] Multiple jobs process without blocking
- [ ] No error messages in logs
- [ ] Database records show correct status/stage
- [ ] All 3 processing modes tested successfully
- [ ] Error handling works (graceful degradation)
- [ ] Performance improved (jobs complete faster)

---

## Rollback Decision

**✅ APPROVE FOR PRODUCTION IF**:
- All stage transitions work correctly
- Status reaches "completed"
- No regression in other modes
- Performance is equal or better

**❌ ROLLBACK IF**:
- Jobs still stuck at "running" or stage doesn't advance
- Errors appear in logs related to next_stage parameter
- API becomes unresponsive
- Existing jobs are damaged

**Rollback command**:
```bash
cp clarityocr/pipeline_v2.py.backup clarityocr/pipeline_v2.py
docker restart <container>
```

---

## Contact for Issues

If verification fails:
1. Check the `DEBUGGING_SUMMARY.md` for explanations
2. Verify `next_stage` parameter exists: `grep -n "next_stage" clarityocr/pipeline_v2.py`
3. Check that file replacement was complete (not partial copy)
4. Compare with `PATCH.diff` line by line
5. Ensure Python 3.3+ is running (ThreadPoolExecutor availability)

---

## Sign-Off

**Verified by**: _________________  
**Date**: _________________  
**Environment**: _________________  
**Status**: ☐ READY FOR PRODUCTION  

---

**Questions? Consult**:
- `BUG_ANALYSIS_AND_FIX.md` - Technical details
- `DEPLOYMENT_GUIDE.md` - Deployment steps
- `DEBUGGING_SUMMARY.md` - Investigation narrative
