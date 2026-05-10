"""
Chat business-logic helpers.

Extracted from routes/chat.py so that route handlers stay thin.
All functions here are pure utilities — no FastAPI routing, no DB writes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import HTTPException, status
from groq import RateLimitError as GroqRateLimitError
from motor.motor_asyncio import AsyncIOMotorDatabase

from .. import config
from ..utils.groq_client import get_groq_client
from ..utils.nlp import analyze_message_nlp as _fallback_nlp
from ..utils.hybrid_adapter import get_hybrid_adapter_runtime, infer_hybrid_adapter
from alia_nlp.src.layers.L6_scoring.explainability import build as _build_explainability

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: dict[str, str] = {
    "physician_portal": (
        "You are ALIA, an AI pharmaceutical representative for Laboratoires Vital (ALIA). "
        "You help healthcare professionals with product information, clinical data summaries, "
        "dosing guidelines, and educational content. Be clear, accurate, and compliant. "
        "Respond in a natural, conversational way, as if speaking with a colleague. Avoid sounding scripted, generic, or like a document extract. "
        "Use the retrieved product/document context to answer directly and succinctly. Lead with the answer in plain language, then add only the most relevant details. "
        "Keep replies short to medium length unless the user asks for more depth. Only use bullets when they genuinely improve clarity. "
        "Avoid phrases like 'according to the document' or 'based on the provided excerpt' unless the user explicitly asks for sourcing. "
        "Do not provide medical advice for individual patients. Do not diagnose. "
        "If you are uncertain, say so and suggest consulting official labeling or a medical specialist."
    ),
    "medrep_training": (
        "You are simulating a physician in a training scenario for medical representatives. "
        "Respond realistically to the rep's messages. You may ask challenging questions, "
        "object to claims, or request evidence. Keep responses concise and professional."
    ),
}

_CLARIFICATION_OPTIONS: dict[str, list[str]] = {
    "physician_portal": [
        "Dosing and administration guidance",
        "Safety profile, contraindications, or interactions",
        "Efficacy, indications, or study evidence",
        "Something else (please specify)",
    ],
    "medrep_training": [
        "Role-play objection handling",
        "Product messaging and positioning",
        "Visit structure and call planning",
        "Something else (please specify)",
    ],
}

# ---------------------------------------------------------------------------
# Mode helpers
# ---------------------------------------------------------------------------

def ensure_mode(mode: str) -> str:
    """Raise 400 if mode is not a recognised chat mode."""
    if mode not in SYSTEM_PROMPTS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid mode")
    return mode


# ---------------------------------------------------------------------------
# Session ownership
# ---------------------------------------------------------------------------

async def get_owned_session(
    db: AsyncIOMotorDatabase,
    session_id: str,
    user_email: str,
) -> tuple[dict, ObjectId]:
    """
    Fetch a conversation by ID and verify the requesting user owns it.

    Returns (doc, oid) on success.
    Raises 400 for a malformed ID, 404 if not found or not owned.
    """
    try:
        oid = ObjectId(session_id)
    except InvalidId as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session id",
        ) from exc

    doc = await db.conversations.find_one({"_id": oid})
    if not doc or doc.get("user_email") != user_email:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    return doc, oid


# ---------------------------------------------------------------------------
# NLP analysis
# ---------------------------------------------------------------------------

def analyze_message_nlp(
    user_text: str,
    history: list[dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    """
    Run NLP analysis on a user message.

    Uses the hybrid adapter when enabled (ALIA_USE_HYBRID_ADAPTER=1); falls
    back to the rule/keyword-based analyser otherwise.
    """
    if not config.USE_HYBRID_ADAPTER:
        return _fallback_nlp(user_text=user_text, history=history, mode=mode)

    runtime = get_hybrid_adapter_runtime()
    if runtime is None:
        logger.info("Hybrid adapter unavailable; using fallback NLP analyser")
        return _fallback_nlp(user_text=user_text, history=history, mode=mode)

    hybrid_messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPTS[mode]}
    ]
    for entry in history:
        if not isinstance(entry, dict):
            continue
        role = str(entry.get("role", "")).strip()
        content = str(entry.get("content", "")).strip()
        if role in {"system", "user", "assistant"} and content:
            hybrid_messages.append({"role": role, "content": content})
    hybrid_messages.append({"role": "user", "content": user_text})

    prediction = infer_hybrid_adapter(
        messages=hybrid_messages,
        runtime=runtime,
        max_new_tokens=config.HYBRID_MAX_NEW_TOKENS,
        hybrid_with_baseline=True,
    )
    if not prediction:
        logger.info("Hybrid adapter inference failed; using fallback NLP analyser")
        return _fallback_nlp(user_text=user_text, history=history, mode=mode)

    fallback = _fallback_nlp(user_text=user_text, history=history, mode=mode)
    analysis: dict[str, Any] = dict(fallback) if isinstance(fallback, dict) else {}
    analysis.update({
        "intent":              prediction.get("intent",              analysis.get("intent", "other")),
        "needs_clarification": prediction.get("needs_clarification", analysis.get("needs_clarification", False)),
        "safety_flags":        prediction.get("safety_flags",        analysis.get("safety_flags", [])),
        "secondary_tags":      prediction.get("secondary_tags",      analysis.get("secondary_tags", [])),
        "entity_map":          prediction.get("entity_map",          analysis.get("entity_map", {})),
        "confidence":          prediction.get("confidence",          analysis.get("confidence", 0.0)),
    })
    analysis["explainability"] = _build_explainability(
        user_text=user_text,
        intent=analysis.get("intent", "other"),
        entity_map=analysis.get("entity_map", {}),
        secondary_tags=analysis.get("secondary_tags", []),
        confidence=float(analysis.get("confidence") or 0.0),
        intent_source=analysis.get("intent_source", "unknown"),
    )
    return analysis


# ---------------------------------------------------------------------------
# Clarification reply
# ---------------------------------------------------------------------------

def build_clarification_reply(mode: str, nlp_analysis: dict[str, Any]) -> str:
    """Build a structured clarification prompt for the user."""
    options = _CLARIFICATION_OPTIONS.get(mode, _CLARIFICATION_OPTIONS["physician_portal"])
    topics = nlp_analysis.get("topics") or []
    entities = nlp_analysis.get("entities") or []

    focus_terms: list[str] = []
    for value in [*entities, *topics]:
        if isinstance(value, str) and value.strip():
            focus_terms.append(value.strip())

    unique_focus: list[str] = []
    seen: set[str] = set()
    for value in focus_terms:
        key = value.lower()
        if key not in seen:
            unique_focus.append(value)
            seen.add(key)
        if len(unique_focus) >= 2:
            break

    focus = f" around {', '.join(unique_focus)}" if unique_focus else ""
    return (
        f"I want to make sure I answer the right thing. "
        f"Could you clarify what you need{focus}?\n"
        f"1. {options[0]}\n"
        f"2. {options[1]}\n"
        f"3. {options[2]}\n"
        f"4. {options[3]}"
    )


# ---------------------------------------------------------------------------
# LLM completion
# ---------------------------------------------------------------------------

async def chat_completion(messages: list[dict[str, Any]]) -> str:
    """Call the Groq LLM and return the assistant reply text."""
    client = get_groq_client()
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Language model client is not configured (missing GROQ_API_KEY)",
        )

    def _call() -> str:
        try:
            completion = client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.7,
                max_tokens=1024,
            )
        except GroqRateLimitError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Language model temporarily unavailable (rate limit). Please try again in a few minutes.",
            ) from exc
        choice = completion.choices[0].message
        if not choice or not choice.content:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Empty response from language model",
            )
        return choice.content.strip()

    return await asyncio.to_thread(_call)
