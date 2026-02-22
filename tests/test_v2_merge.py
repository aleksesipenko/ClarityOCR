import os
import time
import uuid
import json
import pytest
import shutil
from types import SimpleNamespace
from fastapi.testclient import TestClient
from PIL import Image

# Use a real file DB for testing because workers run in separate threads
os.environ["CLARITY_ENV"] = "test"
os.environ["DB_PATH"] = "test_v2_merge.sqlite"

from clarityocr.server import app
from clarityocr.db import setup_db, close_db
from clarityocr.pipeline_v2 import start_workers, stop_workers

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_test_env():
    if os.path.exists("test_v2_merge.sqlite"):
        os.remove("test_v2_merge.sqlite")
    
    setup_db()
    start_workers(1)
    yield
    stop_workers()
    close_db()
    
    # Cleanup DB
    if os.path.exists("test_v2_merge.sqlite"):
        os.remove("test_v2_merge.sqlite")

def create_dummy_image(path, color, size=(100, 100)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new('RGB', size, color=color)
    img.save(path)


def wait_for_terminal_job_status(job_id: str, timeout_s: int = 30) -> str:
    deadline = time.time() + timeout_s
    status = "queued"
    while time.time() < deadline:
        res = client.get(f"/api/v2/jobs/{job_id}")
        status = res.json()["status"]
        if status in ["completed", "failed", "canceled"]:
            return status
        time.sleep(0.5)
    return status

def test_merge_endpoint_and_pipeline():
    # 1. Prepare dummy files
    test_dir = "test_merge_inputs"
    img1 = os.path.join(test_dir, "02_img.jpg")
    img2 = os.path.join(test_dir, "01_img.jpg")
    img3 = os.path.join(test_dir, "03_duplicate.jpg")
    
    create_dummy_image(img1, 'red')
    create_dummy_image(img2, 'blue')
    # Make img3 a perfect byte duplicate of img1 to test the sha256 dedupe logic
    shutil.copy(img1, img3)

    req_id = str(uuid.uuid4())
    payload = {
        "api_version": "v2",
        "meta_schema_version": "1.0",
        "client_id": "test_suite",
        "client_request_id": req_id,
        "inputs": [img1, img2, img3],
        "mode": "merge_only",
        "preset": "balanced",
        "naming_policy": "filename", # Triggers filename sort
        "polish": "off"
    }

    # 2. Submit API Request
    response = client.post("/api/v2/merge", json=payload)
    assert response.status_code == 202, f"Failed to submit merge job: {response.text}"
    job_id = response.json().get("job_id")
    assert job_id is not None

    # 3. Wait for workers to finish (poll job status)
    max_wait = 20
    status = "queued"
    for _ in range(max_wait):
        res = client.get(f"/api/v2/jobs/{job_id}")
        data = res.json()
        status = data["status"]
        if status in ["completed", "failed", "canceled"]:
            break
        time.sleep(1)

    assert status == "completed", f"Job ended in {status} state!"

    # 4. Verify JobFiles logic
    res = client.get(f"/api/v2/jobs/{job_id}/files")
    files = res.json().get("files", [])
    assert len(files) == 1, "Merge logic should only create ONE proxy file representing the task"
    assert files[0]["input_path"] == "<merge_job>"
    assert files[0]["status"] == "completed"

    # 5. Verify Artifacts
    output_dir = os.path.join("output_v2", job_id)
    pdf_path = os.path.join(output_dir, "merged.pdf")
    report_path = os.path.join(output_dir, "merge_report.json")
    manifest_path = os.path.join(output_dir, "batch_manifest.json")
    
    assert os.path.exists(pdf_path), "merged.pdf was not generated"
    assert os.path.exists(report_path), "merge_report.json was not generated"
    assert os.path.exists(manifest_path), "batch_manifest.json was not generated for merge-only mode"

    # Verify report logic
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    assert report["processed_files"] == 2, "Only 2 unique files should be processed"
    assert report["skipped_duplicate"] == 1, "One file should be skipped due to deduplication hash"
    
    warning_events = [w.get("event") for w in report.get("warnings", [])]
    assert "duplicate_detected" in warning_events, "duplicate_detected warning missing from report"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest_types = {f["type"] for f in manifest["files"]}
    assert "merged_pdf" in manifest_types
    assert "merge_report" in manifest_types

    # Cleanup artifacts
    shutil.rmtree(test_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)


def test_merge_then_ocr_completes_with_non_empty_md(monkeypatch):
    test_dir = "test_merge_then_ocr_ok"
    img1 = os.path.join(test_dir, "01_img.jpg")
    img2 = os.path.join(test_dir, "02_img.jpg")
    create_dummy_image(img1, "green")
    create_dummy_image(img2, "yellow")

    def fake_merge_pipeline(inputs, output_dir, order_by):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "merged.pdf"), "wb") as f:
            f.write(b"%PDF-1.4\n%fake\n")
        with open(os.path.join(output_dir, "merge_report.json"), "w", encoding="utf-8") as f:
            json.dump({"processed_files": len(inputs), "skipped_duplicate": 0, "warnings": []}, f)
        return {"processed_files": len(inputs), "skipped_duplicate": 0, "warnings": []}

    def fake_subprocess_run(cmd, **kwargs):
        output_dir = cmd[cmd.index("--output-dir") + 1]
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "merged.md"), "w", encoding="utf-8") as f:
            f.write("[p:1]\nReadable OCR text\n")
        return SimpleNamespace(returncode=0, stdout="ok")

    monkeypatch.setattr("clarityocr.merger.merge_pipeline", fake_merge_pipeline)
    monkeypatch.setattr("subprocess.run", fake_subprocess_run)

    req_id = str(uuid.uuid4())
    payload = {
        "api_version": "v2",
        "meta_schema_version": "1.0",
        "client_id": "test_suite",
        "client_request_id": req_id,
        "inputs": [img1, img2],
        "mode": "merge_then_ocr",
        "preset": "balanced",
        "naming_policy": "filename",
        "polish": "off"
    }

    response = client.post("/api/v2/merge-and-ocr", json=payload)
    assert response.status_code == 202, f"Failed to submit merge_then_ocr job: {response.text}"
    job_id = response.json()["job_id"]

    status = wait_for_terminal_job_status(job_id)
    assert status == "completed"

    artifacts_resp = client.get(f"/api/v2/jobs/{job_id}/artifacts")
    assert artifacts_resp.status_code == 200
    artifacts = artifacts_resp.json()["artifacts"]
    md_artifacts = [a for a in artifacts if a["type"] == "md"]
    assert md_artifacts, "Expected md artifact for merge_then_ocr"

    md_id = md_artifacts[0]["id"]
    md_download = client.get(f"/api/v2/artifacts/{md_id}/download")
    assert md_download.status_code == 200
    assert md_download.content.strip(), "Expected non-empty OCR markdown output"

    shutil.rmtree(test_dir, ignore_errors=True)
    shutil.rmtree(os.path.join("output_v2", job_id), ignore_errors=True)


def test_merge_then_ocr_fails_when_md_is_effectively_empty(monkeypatch):
    test_dir = "test_merge_then_ocr_empty"
    img1 = os.path.join(test_dir, "01_img.jpg")
    create_dummy_image(img1, "black")

    def fake_merge_pipeline(inputs, output_dir, order_by):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "merged.pdf"), "wb") as f:
            f.write(b"%PDF-1.4\n%fake\n")
        with open(os.path.join(output_dir, "merge_report.json"), "w", encoding="utf-8") as f:
            json.dump({"processed_files": len(inputs), "skipped_duplicate": 0, "warnings": []}, f)
        return {"processed_files": len(inputs), "skipped_duplicate": 0, "warnings": []}

    def fake_subprocess_run(cmd, **kwargs):
        output_dir = cmd[cmd.index("--output-dir") + 1]
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "merged.md"), "w", encoding="utf-8") as f:
            f.write("\n[p:1]\n \n")
        return SimpleNamespace(returncode=0, stdout="ok")

    monkeypatch.setattr("clarityocr.merger.merge_pipeline", fake_merge_pipeline)
    monkeypatch.setattr("subprocess.run", fake_subprocess_run)

    req_id = str(uuid.uuid4())
    payload = {
        "api_version": "v2",
        "meta_schema_version": "1.0",
        "client_id": "test_suite",
        "client_request_id": req_id,
        "inputs": [img1],
        "mode": "merge_then_ocr",
        "preset": "balanced",
        "naming_policy": "filename",
        "polish": "off"
    }

    response = client.post("/api/v2/merge-and-ocr", json=payload)
    assert response.status_code == 202, f"Failed to submit merge_then_ocr job: {response.text}"
    job_id = response.json()["job_id"]

    status = wait_for_terminal_job_status(job_id)
    assert status == "failed"

    files_resp = client.get(f"/api/v2/jobs/{job_id}/files")
    assert files_resp.status_code == 200
    files = files_resp.json()["files"]
    assert files and files[0]["status"] == "failed_final"
    assert "empty after normalization" in (files[0]["error"] or "").lower()

    shutil.rmtree(test_dir, ignore_errors=True)
    shutil.rmtree(os.path.join("output_v2", job_id), ignore_errors=True)
