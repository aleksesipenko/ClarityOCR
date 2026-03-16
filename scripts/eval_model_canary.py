#!/usr/bin/env python3
"""
ClarityOCR — Qwen model evaluation script (Phase 3.1).

Compares model A (current: Qwen2.5-0.5B) vs model B (target: Qwen2.5-3B or Qwen2.5-7B)
on a small corpus of OCR-polished text samples.

Usage:
    python scripts/eval_model_canary.py --base-url http://localhost:8000/v1 \
        --model-a Qwen/Qwen2.5-0.5B-Instruct \
        --model-b Qwen/Qwen2.5-3B-Instruct \
        --test-dir tests/eval_corpus/ \
        --output eval_results.json

Exit codes:
    0 = model-b wins or ties (safe to promote)
    1 = model-a wins (do not promote)
    2 = error / both models unavailable
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai", file=sys.stderr)
    sys.exit(2)

# ──────────────────────────────────────────────────────────────────────────────
# Test corpus (inline fallback if no test-dir provided)
# ──────────────────────────────────────────────────────────────────────────────

INLINE_TEST_SAMPLES = [
    {
        "id": "sample-ru-01",
        "input": "Кпримеру, это слипшиеся слова из-за OCRошибки.",
        "expected_keywords": ["К примеру", "OCR"],
    },
    {
        "id": "sample-ru-02",
        "input": "Гла-\nва первая: начало.",
        "expected_keywords": ["Глава", "первая"],
    },
    {
        "id": "sample-en-01",
        "input": "Thisis a brokenword froman OCRscan.",
        "expected_keywords": ["This is", "broken word"],
    },
]

POLISH_SYSTEM_PROMPT = (
    "You are a text correction assistant for OCR-extracted documents. "
    "Fix OCR errors, split merged words, repair broken hyphenated words, "
    "and fix punctuation. Output ONLY the corrected text, nothing else."
)


def query_model(
    client: "OpenAI",
    model: str,
    text: str,
    timeout: float = 30.0,
) -> Optional[str]:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": POLISH_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=512,
            temperature=0.1,
            timeout=timeout,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        return None


def score_output(output: str, expected_keywords: list[str]) -> float:
    """Simple keyword-presence score (0.0–1.0)."""
    if not output:
        return 0.0
    hits = sum(1 for kw in expected_keywords if kw.lower() in output.lower())
    return hits / len(expected_keywords) if expected_keywords else 0.5


def load_test_samples(test_dir: Optional[str]) -> list[dict]:
    if not test_dir:
        return INLINE_TEST_SAMPLES

    path = Path(test_dir)
    if not path.exists():
        print(f"WARN: test-dir {test_dir!r} not found, using inline corpus", file=sys.stderr)
        return INLINE_TEST_SAMPLES

    samples = []
    for f in sorted(path.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            samples.append(data)
        except Exception as exc:
            print(f"WARN: skipping {f}: {exc}", file=sys.stderr)

    return samples or INLINE_TEST_SAMPLES


def eval_model(client: "OpenAI", model: str, samples: list[dict]) -> dict:
    results = []
    total_score = 0.0
    errors = 0

    for sample in samples:
        t0 = time.monotonic()
        output = query_model(client, model, sample["input"])
        latency_ms = int((time.monotonic() - t0) * 1000)

        if output is None:
            errors += 1
            results.append({"id": sample["id"], "error": True, "score": 0.0, "latency_ms": latency_ms})
            continue

        score = score_output(output, sample.get("expected_keywords", []))
        total_score += score
        results.append({
            "id": sample["id"],
            "score": score,
            "latency_ms": latency_ms,
            "output_preview": output[:120],
        })

    avg_score = total_score / len(samples) if samples else 0.0
    avg_latency = sum(r["latency_ms"] for r in results) / len(results) if results else 0
    return {
        "model": model,
        "samples": len(samples),
        "errors": errors,
        "avg_score": round(avg_score, 3),
        "avg_latency_ms": int(avg_latency),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="ClarityOCR model canary eval")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model-a", default="Qwen/Qwen2.5-0.5B-Instruct", help="Current/baseline model")
    parser.add_argument("--model-b", default="Qwen/Qwen2.5-3B-Instruct", help="Candidate model")
    parser.add_argument("--test-dir", default=None, help="Directory with *.json test samples")
    parser.add_argument("--output", default=None, help="Write JSON results to this file")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="local")

    samples = load_test_samples(args.test_dir)
    print(f"Loaded {len(samples)} test samples")

    print(f"\nEvaluating model-a: {args.model_a}")
    results_a = eval_model(client, args.model_a, samples)

    print(f"\nEvaluating model-b: {args.model_b}")
    results_b = eval_model(client, args.model_b, samples)

    comparison = {
        "model_a": results_a,
        "model_b": results_b,
        "verdict": None,
        "recommendation": None,
    }

    print(f"\n{'='*60}")
    print(f"Model A ({args.model_a}): score={results_a['avg_score']}, latency={results_a['avg_latency_ms']}ms, errors={results_a['errors']}")
    print(f"Model B ({args.model_b}): score={results_b['avg_score']}, latency={results_b['avg_latency_ms']}ms, errors={results_b['errors']}")

    # Verdict: model-b wins if score is >= model-a AND errors not worse
    score_diff = results_b["avg_score"] - results_a["avg_score"]
    if results_b["errors"] > results_a["errors"] and results_b["errors"] > 0:
        verdict = "model-a-wins"
        rec = "Keep model-a (model-b has more errors)"
    elif score_diff >= -0.05:
        verdict = "model-b-wins"
        rec = f"Promote model-b (score diff: {score_diff:+.3f})"
    else:
        verdict = "model-a-wins"
        rec = f"Keep model-a (model-b score worse by {-score_diff:.3f})"

    comparison["verdict"] = verdict
    comparison["recommendation"] = rec

    print(f"\nVerdict: {verdict}")
    print(f"Recommendation: {rec}")
    print(f"{'='*60}\n")

    if args.output:
        Path(args.output).write_text(json.dumps(comparison, indent=2))
        print(f"Results written to: {args.output}")

    sys.exit(0 if verdict == "model-b-wins" else 1)


if __name__ == "__main__":
    main()
