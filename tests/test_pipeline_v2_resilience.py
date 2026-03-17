import hmac
import hashlib
import json
import time
from pathlib import Path

from clarityocr.pipeline_v2 import (
    LLMCircuitBreaker,
    send_webhook,
    _assert_non_empty_ocr_text,
    _fallback_markdown_from_input,
)


def test_circuit_breaker_opens_and_recovers():
    breaker = LLMCircuitBreaker(failure_threshold=2, cooldown_sec=1)

    assert breaker.allow_request() is True

    breaker.record_failure("first")
    assert breaker.allow_request() is True

    breaker.record_failure("second")
    assert breaker.allow_request() is False
    assert breaker.state()["status"] == "open"

    time.sleep(1.1)
    assert breaker.allow_request() is True

    breaker.record_success()
    state = breaker.state()
    assert state["status"] == "closed"
    assert state["consecutive_failures"] == 0


def test_send_webhook_hmac_signature(monkeypatch):
    captured = {}

    class Response:
        status_code = 204

    def fake_post(url, content, headers, timeout):
        captured["url"] = url
        captured["content"] = content
        captured["headers"] = headers
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("clarityocr.pipeline_v2.httpx.post", fake_post)

    secret = "top-secret"
    payload = {"job_id": "j1", "status": "completed"}
    ok = send_webhook("https://example.com/webhook", secret, payload)
    assert ok is True

    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()

    assert captured["url"] == "https://example.com/webhook"
    assert captured["content"] == body
    assert captured["headers"]["X-ClarityOCR-Signature"] == f"sha256={expected}"


def test_assert_non_empty_ocr_text_rejects_only_markers():
    payload = "\n[p:1]\n\n[p:2]\n   \n"
    try:
        _assert_non_empty_ocr_text(payload, "doc.md")
        assert False, "Expected ValueError for marker-only OCR output"
    except ValueError as exc:
        assert "empty" in str(exc).lower()


def test_assert_non_empty_ocr_text_accepts_real_text():
    _assert_non_empty_ocr_text("\n[p:1]\nHello world\n", "doc.md")


def test_fallback_image_converts_to_pdf_before_markdown(tmp_path, monkeypatch):
    img_path = tmp_path / "page.jpg"
    img_path.write_bytes(b"fake-jpeg")
    md_path = tmp_path / "out.md"
    calls = {}

    def fake_images_to_pdf(image_paths, output_pdf):
        calls["image_paths"] = [str(p) for p in image_paths]
        calls["output_pdf"] = str(output_pdf)
        Path(output_pdf).write_bytes(b"%PDF-1.4\n")

    def fake_convert_to_markdown(pdf_path, output_path):
        calls["pdf_path"] = str(pdf_path)
        calls["output_path"] = str(output_path)
        Path(output_path).write_text("fallback text", encoding="utf-8")

    monkeypatch.setattr("clarityocr.converter.images_to_pdf", fake_images_to_pdf)
    monkeypatch.setattr("clarityocr.simple_converter.convert_to_markdown", fake_convert_to_markdown)

    _fallback_markdown_from_input(str(img_path), str(md_path))

    assert md_path.read_text(encoding="utf-8") == "fallback text"
    assert calls["image_paths"] == [str(img_path)]
    assert calls["pdf_path"].lower().endswith(".pdf")
