import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clarityocr.db import Artifact, JobFile, get_session
from clarityocr.job_manager import record_artifact
from clarityocr.server import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "v2_uploads.sqlite"))
    monkeypatch.setenv("V2_UPLOAD_ROOT", str(tmp_path / "uploads"))
    with TestClient(app) as c:
        yield c


def test_upload_submit_and_artifact_download(client):
    files = [
        ("files", ("sample1.pdf", b"%PDF-1.4\n%test\n", "application/pdf")),
        ("files", ("sample2.jpg", b"\xff\xd8\xff\xe0\x00\x10JFIF", "image/jpeg")),
    ]
    upload_resp = client.post("/api/v2/uploads", files=files)
    assert upload_resp.status_code == 201
    upload_data = upload_resp.json()
    assert len(upload_data["inputs"]) == 2

    for server_path in upload_data["inputs"]:
        assert Path(server_path).exists()

    req_id = str(uuid.uuid4())
    start_payload = {
        "api_version": "v2",
        "meta_schema_version": "1.0",
        "client_id": "pytest_upload_client",
        "client_request_id": req_id,
        "inputs": upload_data["inputs"],
        "mode": "ocr_only",
        "preset": "balanced",
        "naming_policy": "on",
        "polish": "off",
    }
    start_resp = client.post("/api/v2/jobs", json=start_payload)
    assert start_resp.status_code == 202
    job_id = start_resp.json()["job_id"]

    jobs_resp = client.get("/api/v2/jobs?limit=10")
    assert jobs_resp.status_code == 200
    assert any(j["job_id"] == job_id for j in jobs_resp.json()["jobs"])

    artifact_path = Path(upload_data["inputs"][0]).with_suffix(".artifact.txt")
    artifact_path.write_text("artifact-body", encoding="utf-8")

    with get_session() as session:
        file_row = session.query(JobFile).filter(JobFile.job_id == job_id).first()
        artifact_id = str(uuid.uuid4())
        session.add(
            Artifact(
                id=artifact_id,
                job_id=job_id,
                file_id=file_row.id if file_row else None,
                type="test_artifact",
                path=str(artifact_path),
            )
        )
        session.commit()

    artifacts_resp = client.get(f"/api/v2/jobs/{job_id}/artifacts")
    assert artifacts_resp.status_code == 200
    artifacts = artifacts_resp.json()["artifacts"]
    assert any(a["id"] == artifact_id for a in artifacts)

    download_resp = client.get(f"/api/v2/artifacts/{artifact_id}/download")
    assert download_resp.status_code == 200
    assert download_resp.content == b"artifact-body"


def test_record_artifact_is_idempotent(client):
    files = [("files", ("sample.pdf", b"%PDF-1.4\n%test\n", "application/pdf"))]
    upload_resp = client.post("/api/v2/uploads", files=files)
    assert upload_resp.status_code == 201
    upload_data = upload_resp.json()

    start_payload = {
        "api_version": "v2",
        "meta_schema_version": "1.0",
        "client_id": "pytest_upload_client",
        "client_request_id": str(uuid.uuid4()),
        "inputs": upload_data["inputs"],
        "mode": "ocr_only",
        "preset": "balanced",
        "naming_policy": "on",
        "polish": "off",
    }
    start_resp = client.post("/api/v2/jobs", json=start_payload)
    assert start_resp.status_code == 202
    job_id = start_resp.json()["job_id"]

    artifact_path = Path(upload_data["inputs"][0]).with_suffix(".artifact-idempotent.txt")
    artifact_path.write_text("artifact-body", encoding="utf-8")

    with get_session() as session:
        file_row = session.query(JobFile).filter(JobFile.job_id == job_id).first()
        assert file_row is not None
        a1 = record_artifact(session, job_id, file_row.id, "md", str(artifact_path), sha256="abc")
        a2 = record_artifact(session, job_id, file_row.id, "md", str(artifact_path), sha256="abc")
        assert a1.id == a2.id

        total = (
            session.query(Artifact)
            .filter(
                Artifact.job_id == job_id,
                Artifact.file_id == file_row.id,
                Artifact.type == "md",
                Artifact.path == str(artifact_path),
            )
            .count()
        )
        assert total == 1
