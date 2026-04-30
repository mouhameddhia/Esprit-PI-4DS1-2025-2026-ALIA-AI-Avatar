"""LLM-as-judge runtime for evaluating pharmaceutical RAG answers."""

from __future__ import annotations

import json
from typing import Any
from urllib import request

from .prompt_template import build_llm_judge_prompt

DEFAULT_JUDGE_MODEL = "smollm2:135m"


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.0
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _verdict_from_score(overall_score: float) -> str:
    if overall_score >= 0.90:
        return "excellent"
    if overall_score >= 0.75:
        return "good"
    if overall_score >= 0.60:
        return "acceptable"
    return "poor"


def _normalize_result(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Judge model returned an empty or invalid JSON object.")

    required_keys = {
        "faithfulness",
        "answer_relevance",
        "context_utilization",
        "medical_safety",
        "clarity",
        "mode_alignment",
    }
    if not any(key in payload for key in required_keys):
        raise ValueError(f"Judge model JSON is missing expected scoring fields: {sorted(required_keys)}")

    metrics = {
        "faithfulness": _clamp_score(payload.get("faithfulness", 0.0)),
        "answer_relevance": _clamp_score(payload.get("answer_relevance", 0.0)),
        "context_utilization": _clamp_score(payload.get("context_utilization", 0.0)),
        "medical_safety": _clamp_score(payload.get("medical_safety", 0.0)),
        "clarity": _clamp_score(payload.get("clarity", 0.0)),
        "mode_alignment": _clamp_score(payload.get("mode_alignment", 0.0)),
    }

    # Enforce the project scoring rule regardless of model-reported overall_score.
    overall_score = _clamp_score(sum(metrics.values()) / len(metrics))

    issues_raw = payload.get("issues", [])
    if isinstance(issues_raw, list):
        issues = [str(item).strip() for item in issues_raw if str(item).strip()]
    else:
        issues = []

    if any(value < 0.5 for value in metrics.values()) and not issues:
        issues.append("One or more criteria scored below 0.5 without explicit rationale.")

    verdict_candidate = str(payload.get("verdict", "")).strip().lower()
    verdict = verdict_candidate if verdict_candidate in {"excellent", "good", "acceptable", "poor"} else _verdict_from_score(overall_score)

    return {
        **metrics,
        "overall_score": overall_score,
        "issues": issues,
        "verdict": verdict,
    }


def evaluate_answer_with_llm_judge(
    *,
    mode: str,
    question: str,
    answer: str,
    context: str,
    base_url: str = "http://localhost:11434",
    model: str = DEFAULT_JUDGE_MODEL,
    timeout_seconds: int = 90,
) -> dict[str, Any]:
    """Evaluate one answer with a strict medical LLM judge and return normalized JSON."""

    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"commercial", "training"}:
        raise ValueError("mode must be either 'commercial' or 'training'.")

    prompt = build_llm_judge_prompt(
        mode=normalized_mode,
        question=question,
        answer=answer,
        context=context,
    )

    def _generate_once(*, force_json_format: bool) -> dict[str, Any]:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": 4096,
            },
        }
        if force_json_format:
            payload["format"] = "json"

        req = request.Request(
            url=f"{base_url.rstrip('/')}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with request.urlopen(req, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")

        body = json.loads(raw)
        response_text = str(body.get("response", "")).strip()
        if not response_text:
            raise ValueError("Judge model returned an empty response.")
        parsed = json.loads(response_text)
        if not isinstance(parsed, dict):
            raise ValueError("Judge model did not return a JSON object.")
        return parsed

    first_error: Exception | None = None
    for force_json_format in (True, False):
        try:
            parsed = _generate_once(force_json_format=force_json_format)
            return _normalize_result(parsed)
        except Exception as exc:
            first_error = exc

    raise ValueError(f"Judge evaluation failed after retries: {first_error}")
