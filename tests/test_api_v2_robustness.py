import os
import zipfile
import pytest
from fastapi.testclient import TestClient

from clarityocr.server import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_liveness(client):
    """Verify /health/live returns HTTP 200 immediately."""
    resp = client.get("/api/v2/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "alive"}

def test_readiness(client):
    """Verify /health/ready returns specific module statuses."""
    resp = client.get("/api/v2/health/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert "ocr_core" in data
    assert "db" in data
    assert "llm" in data

def test_job_idempotency_contract(client):
    """
    Test job idempotency behavior:
    1. Submitting a new job returns 202 Accepted.
    2. Submitting the exact same job returns 202 Accepted with the SAME job_id.
    3. Submitting the same client_request_id with a DIFFERENT payload returns 409 Conflict.
    """
    payload = {
        "api_version": "v2",
        "meta_schema_version": "1.0",
        "client_id": "pytest_client",
        "client_request_id": "req-12345",
        "inputs": ["test1.pdf", "test2.pdf"],
        "mode": "ocr_only"
    }
    
    # 1. First request
    resp1 = client.post("/api/v2/jobs", json=payload)
    assert resp1.status_code == 202
    job_id = resp1.json()["job_id"]
    
    # 2. Duplicate exact request
    resp2 = client.post("/api/v2/jobs", json=payload)
    assert resp2.status_code == 202
    assert resp2.json()["job_id"] == job_id
    
    # 3. Same request ID, different payload
    bad_payload = payload.copy()
    bad_payload["inputs"] = ["different_file.pdf"]
    resp3 = client.post("/api/v2/jobs", json=bad_payload)
    assert resp3.status_code == 409
    assert "Idempotency conflict" in resp3.json()["detail"]

def test_job_lifecycle_and_cancellation(client):
    """
    Test that a job goes into queued state, and can be cancelled.
    """
    payload = {
        "api_version": "v2",
        "meta_schema_version": "1.0",
        "client_id": "pytest_client",
        "client_request_id": "req-cancel-1",
        "inputs": ["cancel_test.pdf"],
        "mode": "ocr_only"
    }
    resp = client.post("/api/v2/jobs", json=payload)
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    
    # Verify events
    events_resp = client.get(f"/api/v2/jobs/{job_id}/events")
    assert events_resp.status_code == 200
    events = events_resp.json()["events"]
    assert len(events) > 0
    assert events[0]["event_code"] == "job_queued"
    
    # Cancel the job
    cancel_resp = client.post(f"/api/v2/jobs/{job_id}/cancel")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "canceled"
    
    # Check status endpoint
    status_resp = client.get(f"/api/v2/jobs/{job_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "canceled"
    
    # Check files state is canceled
    files_resp = client.get(f"/api/v2/jobs/{job_id}/files")
    assert files_resp.status_code == 200
    files = files_resp.json()["files"]
    for f in files:
        assert f["status"] == "canceled"

def test_batch_limit(client):
    """Verify max batch size rejection."""
    payload = {
        "api_version": "v2",
        "meta_schema_version": "1.0",
        "client_request_id": "req-huge",
        "inputs": [f"file{i}.pdf" for i in range(1001)],
        "mode": "ocr_only"
    }
    
    resp = client.post("/api/v2/jobs", json=payload)
    assert resp.status_code == 400
    assert "Batch size exceeds limit" in resp.json()["detail"]


def test_ssrf_private_url_blocked(client):
    payload = {
        "api_version": "v2",
        "meta_schema_version": "1.0",
        "client_id": "pytest_client",
        "client_request_id": "req-ssrf-1",
        "inputs": ["http://127.0.0.1/private.pdf"],
        "mode": "ocr_only",
    }
    resp = client.post("/api/v2/jobs", json=payload)
    assert resp.status_code == 400
    assert "blocked IP" in resp.json()["detail"]


def test_zip_depth_limit_blocked(client, tmp_path, monkeypatch):
    monkeypatch.setenv("V2_MAX_ZIP_DEPTH", "1")

    zip_path = tmp_path / "nested.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("a/b/c.txt", "hello")

    payload = {
        "api_version": "v2",
        "meta_schema_version": "1.0",
        "client_id": "pytest_client",
        "client_request_id": "req-zip-depth-1",
        "inputs": [str(zip_path)],
        "mode": "ocr_only",
    }
    resp = client.post("/api/v2/jobs", json=payload)
    assert resp.status_code == 400
    assert "nesting depth exceeds limit" in resp.json()["detail"]

if __name__ == "__main__":
    pytest.main(["-v", __file__])
