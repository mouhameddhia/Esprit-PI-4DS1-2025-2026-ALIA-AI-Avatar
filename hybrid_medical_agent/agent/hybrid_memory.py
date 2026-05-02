"""Hybrid memory for RAG chat: rolling summary + recent window.

Memory JSON schema:
{
  "summary": "string",
  "history": [
    {"user": "...", "assistant": "..."}
  ]
}
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from urllib import request

MAX_HISTORY = 5
MEMORY_FILE = Path(__file__).resolve().parents[1] / "storage" / "hybrid_memory.json"


def _default_memory() -> dict[str, Any]:
    return {"summary": "", "history": []}


def _normalize_text(value: str) -> str:
    return " ".join(str(value).strip().split())


def _sanitize_history(history: Any) -> list[dict[str, str]]:
    if not isinstance(history, list):
        return []

    cleaned: list[dict[str, str]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        user_text = _normalize_text(item.get("user", ""))
        assistant_text = _normalize_text(item.get("assistant", ""))
        if not user_text and not assistant_text:
            continue
        cleaned.append({"user": user_text, "assistant": assistant_text})
    return cleaned


def load_memory() -> dict[str, Any]:
    """Load memory from disk (or initialize default structure)."""

    try:
        if not MEMORY_FILE.exists():
            return _default_memory()

        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _default_memory()

        summary = _normalize_text(data.get("summary", ""))
        history = _sanitize_history(data.get("history", []))
        return {"summary": summary, "history": history}
    except Exception:
        return _default_memory()


def save_memory(memory: dict[str, Any]) -> None:
    """Persist memory to disk using the canonical JSON structure."""

    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": _normalize_text(memory.get("summary", "")),
        "history": _sanitize_history(memory.get("history", [])),
    }
    MEMORY_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_summary_input(old_messages: list[dict[str, str]], existing_summary: str) -> str:
    lines: list[str] = []
    for turn in old_messages:
        lines.append(f"User: {turn.get('user', '')}")
        lines.append(f"Assistant: {turn.get('assistant', '')}")
    transcript = "\n".join(lines).strip()

    return (
        "You are a memory compression assistant for a RAG chatbot.\n"
        "Create an updated concise conversation summary.\n"
        "Rules:\n"
        "- Keep only essential facts, decisions, constraints, and context.\n"
        "- Remove repetition and filler.\n"
        "- Output plain text, maximum 5 to 8 lines.\n"
        "- Do not invent information.\n\n"
        f"Existing summary:\n{existing_summary or '(none)'}\n\n"
        f"Older conversation to compress:\n{transcript or '(none)'}\n\n"
        "Return only the new summary text."
    )


def _summarize_with_ollama(prompt: str) -> str:
    model = os.getenv("MEMORY_SUMMARY_MODEL", "llama3:8b")
    base_url = os.getenv("MEMORY_SUMMARY_BASE_URL", "http://localhost:11434").rstrip("/")
    timeout_seconds = int(os.getenv("MEMORY_SUMMARY_TIMEOUT_SECONDS", "60"))

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 4096,
        },
    }

    req = request.Request(
        url=f"{base_url}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_seconds) as response:
        raw = response.read().decode("utf-8")
    body = json.loads(raw)
    return _normalize_text(body.get("response", ""))


def _fallback_summary(old_messages: list[dict[str, str]], existing_summary: str) -> str:
    """Deterministic fallback when LLM summary is unavailable."""

    fragments: list[str] = []
    if existing_summary:
        fragments.append(existing_summary)

    for turn in old_messages[-6:]:
        user_text = _normalize_text(turn.get("user", ""))
        assistant_text = _normalize_text(turn.get("assistant", ""))
        if user_text:
            fragments.append(f"User asked: {user_text}")
        if assistant_text:
            fragments.append(f"Assistant answered: {assistant_text}")

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for item in fragments:
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        unique.append(item)

    return "\n".join(unique[-8:])


def summarize_history(old_messages: list[dict[str, str]], existing_summary: str) -> str:
    """Create an updated concise summary from old messages + existing summary."""

    sanitized_old = _sanitize_history(old_messages)
    if not sanitized_old and not existing_summary:
        return ""

    prompt = _build_summary_input(sanitized_old, _normalize_text(existing_summary))
    try:
        summary = _summarize_with_ollama(prompt)
    except Exception:
        summary = ""

    if not summary:
        summary = _fallback_summary(sanitized_old, _normalize_text(existing_summary))

    # Enforce concise line count.
    lines = [line.strip() for line in re.split(r"\n+", summary) if line.strip()]
    if len(lines) > 8:
        lines = lines[:8]
    return "\n".join(lines)


def update_memory(user_input: str, assistant_output: str) -> dict[str, Any]:
    """Append one turn and summarize only when history exceeds MAX_HISTORY."""

    memory = load_memory()
    history = _sanitize_history(memory.get("history", []))

    user_text = _normalize_text(user_input)
    assistant_text = _normalize_text(assistant_output)
    if not user_text and not assistant_text:
        return memory

    new_turn = {"user": user_text, "assistant": assistant_text}

    # Prevent duplicate consecutive turns.
    if history:
        last_turn = history[-1]
        if (
            _normalize_text(last_turn.get("user", "")) == user_text
            and _normalize_text(last_turn.get("assistant", "")) == assistant_text
        ):
            return {"summary": memory.get("summary", ""), "history": history}

    history.append(new_turn)

    summary = _normalize_text(memory.get("summary", ""))
    if len(history) > MAX_HISTORY:
        old_messages = history[:-MAX_HISTORY]
        summary = summarize_history(old_messages, summary)
        history = history[-MAX_HISTORY:]

    updated = {"summary": summary, "history": history}
    save_memory(updated)
    return updated


def build_prompt(new_user_input: str) -> str:
    """Build LLM prompt context from persisted summary + recent conversation."""

    memory = load_memory()
    summary = memory.get("summary", "") or "(none)"
    history = _sanitize_history(memory.get("history", []))

    lines = [f"Summary: {summary}", "Recent conversation:"]
    if not history:
        lines.append("(empty)")
    else:
        for turn in history:
            lines.append(f"User: {turn.get('user', '')}")
            lines.append(f"Assistant: {turn.get('assistant', '')}")

    lines.append(f"User: {_normalize_text(new_user_input)}")
    return "\n".join(lines)
