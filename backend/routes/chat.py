"""Chat routes — send message and finalize session."""

import logging
from datetime import datetime
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel, Field

from ..dependencies import get_database, get_current_user, get_rag_pipeline
from ..models.user import UserInDB
from ..utils.chat_helpers import (
    SYSTEM_PROMPTS,
    analyze_message_nlp,
    build_clarification_reply,
    chat_completion,
    ensure_mode,
    get_owned_session,
)
from ..utils.summary import generate_summary_with_caching
from ..utils.nlp_evaluator import evaluate_conversation

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class SendMessageRequest(BaseModel):
    session_id: Optional[str] = None
    content: str = Field(..., min_length=1, max_length=16000)
    mode: Literal["physician_portal", "medrep_training"] = "physician_portal"


class SendMessageResponse(BaseModel):
    session_id: str
    reply: str


class FinalizeResponse(BaseModel):
    session_id: str
    summary: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/message", response_model=SendMessageResponse)
async def send_message(
    body: SendMessageRequest,
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
    rag=Depends(get_rag_pipeline),
):
    now = datetime.utcnow()
    reopened_closed_session = False

    if body.session_id:
        doc, oid = await get_owned_session(db, body.session_id, current_user.email)
        if doc.get("status") == "closed":
            reopened_closed_session = True
        mode = ensure_mode(doc["mode"])
        messages = list(doc.get("messages") or [])
        prior_nlp_events = list(doc.get("nlp_events") or [])
    else:
        doc = None
        oid = None
        mode = ensure_mode(body.mode)
        messages = []
        prior_nlp_events = []

    user_entry: dict[str, Any] = {"role": "user", "content": body.content.strip(), "at": now}

    nlp_analysis = analyze_message_nlp(
        user_text=user_entry["content"],
        history=messages,
        mode=mode,
    )

    needs_clarification = bool(nlp_analysis.get("needs_clarification", False))
    prior_clarification_requested = bool(
        prior_nlp_events and prior_nlp_events[-1].get("clarification_requested")
    )

    nlp_event: dict[str, Any] = {
        "at": now,
        "mode": mode,
        "message": user_entry["content"],
        "intent": nlp_analysis.get("intent", "other"),
        "language": nlp_analysis.get("language", "unknown"),
        "secondary_tags": nlp_analysis.get("secondary_tags", []),
        "entities": nlp_analysis.get("entities", []),
        "entity_map": nlp_analysis.get("entity_map", {}),
        "topics": nlp_analysis.get("topics", []),
        "objections": nlp_analysis.get("objections", []),
        "action_items": nlp_analysis.get("action_items", []),
        "safety_flags": nlp_analysis.get("safety_flags", []),
        "rewritten_query": nlp_analysis.get("rewritten_query", user_entry["content"]),
        "confidence": nlp_analysis.get("confidence", 0.0),
        "taxonomy_version": nlp_analysis.get("taxonomy_version"),
        "explainability": nlp_analysis.get("explainability", {}),
        "clarification_requested": needs_clarification,
        "clarification_follow_up": prior_clarification_requested,
        "clarification_resolved": prior_clarification_requested and not needs_clarification,
    }

    if needs_clarification:
        reply_text = build_clarification_reply(mode=mode, nlp_analysis=nlp_analysis)
    else:
        retrieval_query = nlp_analysis.get("rewritten_query") or user_entry["content"]
        groq_messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPTS[mode]}]

        safety_flags = nlp_analysis.get("safety_flags", [])
        if safety_flags:
            groq_messages.append({
                "role": "system",
                "content": (
                    "Safety note: The user request may include clinical-risk patterns "
                    f"({', '.join(safety_flags)}). Provide general educational information only, "
                    "avoid patient-specific medical advice, and recommend official labeling or "
                    "specialist consultation when needed."
                ),
            })

        nlp_topics = nlp_analysis.get("topics", [])
        nlp_entities = nlp_analysis.get("entities", [])
        if nlp_topics or nlp_entities:
            groq_messages.append({
                "role": "system",
                "content": (
                    f"NLP intent: {nlp_analysis.get('intent', 'other')}\n"
                    f"NLP entities: {', '.join(nlp_entities) if nlp_entities else 'none'}\n"
                    f"NLP topics: {', '.join(nlp_topics) if nlp_topics else 'none'}"
                ),
            })

        try:
            context = await rag.get_context(query=retrieval_query, top_k=3, min_score=0.3)
            if context:
                groq_messages.append({"role": "system", "content": context})
                logger.info("RAG context added to message")
        except Exception as exc:
            logger.warning("Failed to retrieve RAG context: %s", exc)

        for m in messages:
            groq_messages.append({"role": m["role"], "content": m["content"]})
        groq_messages.append({"role": "user", "content": user_entry["content"]})

        reply_text = await chat_completion(groq_messages)

    asst_time = datetime.utcnow()
    asst_entry: dict[str, Any] = {"role": "assistant", "content": reply_text, "at": asst_time}

    messages.append(user_entry)
    messages.append(asst_entry)

    if doc is None:
        result = await db.conversations.insert_one({
            "user_email": current_user.email,
            "mode": mode,
            "messages": messages,
            "nlp_events": [nlp_event],
            "summary": None,
            "summary_created_at": None,
            "summary_method": None,
            "summary_triggered_by": None,
            "rolling_summaries": [],
            "topics": [],
            "objections": [],
            "action_items": [],
            "status": "open",
            "created_at": now,
            "updated_at": asst_time,
        })
        session_id = str(result.inserted_id)
    else:
        update_set: dict[str, Any] = {"messages": messages, "updated_at": asst_time}
        if reopened_closed_session:
            update_set.update({
                "status": "open",
                "summary": None, "summary_created_at": None,
                "summary_method": None, "summary_triggered_by": None,
                "rolling_summaries": [],
                "nlp_evaluation": None, "competency_level": None,
                "evaluation_score": None, "evaluation_dimensions": {},
                "evaluation_strengths": [], "evaluation_gaps": [],
                "evaluation_notes": [], "evaluation_completed_at": None,
            })
        await db.conversations.update_one(
            {"_id": oid},
            {"$set": update_set, "$push": {"nlp_events": nlp_event}},
        )
        session_id = str(oid)

    return SendMessageResponse(session_id=session_id, reply=reply_text)


@router.post("/sessions/{session_id}/finalize", response_model=FinalizeResponse)
async def finalize_session(
    session_id: str,
    force_regenerate: bool = False,
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    doc, oid = await get_owned_session(db, session_id, current_user.email)

    if doc.get("summary") and not force_regenerate:
        return FinalizeResponse(session_id=session_id, summary=doc["summary"])

    msgs = doc.get("messages") or []
    summary, metadata, rolling_summaries = await generate_summary_with_caching(
        session_id=session_id,
        messages=msgs,
        force_regenerate=force_regenerate,
    )

    nlp_evaluation = evaluate_conversation(
        messages=msgs,
        summary=summary,
        metadata=metadata,
        nlp_events=list(doc.get("nlp_events") or []),
    )

    now = datetime.utcnow()
    await db.conversations.update_one(
        {"_id": oid},
        {"$set": {
            "summary": summary,
            "summary_created_at": now,
            "summary_method": "manual",
            "summary_triggered_by": current_user.email,
            "rolling_summaries": rolling_summaries,
            "competency_level": nlp_evaluation.get("level"),
            "evaluation_score": nlp_evaluation.get("score"),
            "evaluation_dimensions": nlp_evaluation.get("dimensions", {}),
            "evaluation_strengths": nlp_evaluation.get("strengths", []),
            "evaluation_gaps": nlp_evaluation.get("gaps", []),
            "evaluation_notes": nlp_evaluation.get("notes", []),
            "evaluation_completed_at": nlp_evaluation.get("evaluated_at"),
            "topics": metadata.get("topics", []),
            "objections": metadata.get("objections", []),
            "action_items": metadata.get("action_items", []),
            "status": "closed",
            "updated_at": now,
        }},
    )

    return FinalizeResponse(session_id=session_id, summary=summary)
