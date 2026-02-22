import hmac
import hashlib
import json
import time

from clarityocr.pipeline_v2 import LLMCircuitBreaker, send_webhook


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
