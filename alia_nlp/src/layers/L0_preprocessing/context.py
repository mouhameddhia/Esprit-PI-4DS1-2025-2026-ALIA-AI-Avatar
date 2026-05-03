"""Build the context string injected into LLM prompts."""

from typing import Any, Dict, List


_HISTORY_WINDOW = 6


def build_history_text(history: List[Dict[str, Any]]) -> str:
    tail = history[-_HISTORY_WINDOW:]
    lines = [
        f"{m.get('role', 'unknown')}: {m.get('content', '')}"
        for m in tail
    ]
    return "\n".join(lines) if lines else "(none)"


def build_user_prompt(user_text: str, mode: str, history: List[Dict[str, Any]]) -> str:
    return (
        f"Mode: {mode}\n"
        f"Recent conversation:\n{build_history_text(history)}\n\n"
        f"Current user message:\n{user_text}"
    )
