import logging
from datetime import datetime
from typing import Any, List, Literal, Optional

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel, Field

import asyncio

from .. import config
from ..dependencies import get_database, get_current_user, get_rep_scoring_service, get_rag_pipeline
from ..models.conversation import ConversationResponse, SessionListItem
from ..models.user import UserInDB
from ..utils.groq_client import get_groq_client
from ..utils.summary import generate_summary_with_caching
from ..utils.nlp import SUPPORTED_INTENTS, analyze_message_nlp as fallback_analyze_message_nlp
from ..utils.hybrid_adapter import get_hybrid_adapter_runtime, infer_hybrid_adapter
from ..utils.nlp_evaluator import evaluate_conversation
from ..services import RepScoringService
from NLP.pipeline.nlp import _build_explainability

logger = logging.getLogger(__name__)

router = APIRouter()

SYSTEM_PROMPTS = {
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

_CLARIFICATION_OPTIONS = {
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


def _use_hybrid_adapter() -> bool:
    return config.USE_HYBRID_ADAPTER


def _analyze_message_nlp(user_text: str, history: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    if not _use_hybrid_adapter():
        return fallback_analyze_message_nlp(user_text=user_text, history=history, mode=mode)

    runtime = get_hybrid_adapter_runtime()
    if runtime is None:
        logger.info("Hybrid adapter disabled or unavailable; using fallback NLP analyzer")
        return fallback_analyze_message_nlp(user_text=user_text, history=history, mode=mode)

    hybrid_messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPTS[mode]}]
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
        logger.info("Hybrid adapter inference failed; using fallback NLP analyzer")
        return fallback_analyze_message_nlp(user_text=user_text, history=history, mode=mode)

    fallback_analysis = fallback_analyze_message_nlp(user_text=user_text, history=history, mode=mode)
    analysis = dict(fallback_analysis) if isinstance(fallback_analysis, dict) else {}
    analysis.update({
        "intent": prediction.get("intent", analysis.get("intent", "other")),
        "needs_clarification": prediction.get("needs_clarification", analysis.get("needs_clarification", False)),
        "safety_flags": prediction.get("safety_flags", analysis.get("safety_flags", [])),
        "secondary_tags": prediction.get("secondary_tags", analysis.get("secondary_tags", [])),
        "entity_map": prediction.get("entity_map", analysis.get("entity_map", {})),
        "confidence": prediction.get("confidence", analysis.get("confidence", 0.0)),
    })
    analysis["explainability"] = _build_explainability(
        user_text=user_text,
        intent=analysis.get("intent", "other"),
        entity_map=analysis.get("entity_map", {}),
        secondary_tags=analysis.get("secondary_tags", []),
        confidence=float(analysis.get("confidence", 0.0) or 0.0),
    )
    return analysis


def _build_clarification_reply(mode: str, nlp_analysis: dict[str, Any]) -> str:
    options = _CLARIFICATION_OPTIONS.get(mode, _CLARIFICATION_OPTIONS["physician_portal"])
    topics = nlp_analysis.get("topics") or []
    entities = nlp_analysis.get("entities") or []

    focus_terms: list[str] = []
    for value in entities:
        if isinstance(value, str) and value.strip():
            focus_terms.append(value.strip())
    for value in topics:
        if isinstance(value, str) and value.strip():
            focus_terms.append(value.strip())

    focus = ""
    if focus_terms:
        unique_focus = []
        seen_focus = set()
        for value in focus_terms:
            value_key = value.lower()
            if value_key not in seen_focus:
                unique_focus.append(value)
                seen_focus.add(value_key)
            if len(unique_focus) >= 2:
                break
        focus = f" around {', '.join(unique_focus)}"

    return (
        "I want to make sure I answer the right thing. "
        f"Could you clarify what you need{focus}?\n"
        f"1. {options[0]}\n"
        f"2. {options[1]}\n"
        f"3. {options[2]}\n"
        f"4. {options[3]}"
    )


async def _chat_completion(messages: list[dict[str, Any]]) -> str:
    client = get_groq_client()

    def _call() -> str:
        completion = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.7,
            max_tokens=1024,
        )
        choice = completion.choices[0].message
        if not choice or not choice.content:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Empty response from language model",
            )
        return choice.content.strip()

    return await asyncio.to_thread(_call)


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


class NLPDebugRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=16000)
    mode: Literal["physician_portal", "medrep_training"] = "physician_portal"
    session_id: Optional[str] = None


class NLPDebugResponse(BaseModel):
    analysis: dict


class RepScoreRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=16000)


class RepScoreResponse(BaseModel):
    clarity_score: float
    persuasion_score: float
    confidence_score: float
    model_source: str


def _ensure_mode(mode: str) -> str:
    if mode not in SYSTEM_PROMPTS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid mode")
    return mode


async def _get_owned_session(
    db: AsyncIOMotorDatabase, session_id: str, email: str
):
    try:
        oid = ObjectId(session_id)
    except InvalidId as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid session id") from exc
    doc = await db.conversations.find_one({"_id": oid})
    if not doc or doc.get("user_email") != email:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return doc, oid


@router.post("/message", response_model=SendMessageResponse)
async def send_message(
    body: SendMessageRequest,
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
    rag = Depends(get_rag_pipeline),
):
    now = datetime.utcnow()
    reopened_closed_session = False

    if body.session_id:
        doc, oid = await _get_owned_session(db, body.session_id, current_user.email)
        if doc.get("status") == "closed":
            reopened_closed_session = True
        mode = _ensure_mode(doc["mode"])
        messages = list(doc.get("messages") or [])
        prior_nlp_events = list(doc.get("nlp_events") or [])
    else:
        doc = None
        oid = None
        mode = _ensure_mode(body.mode)
        messages = []
        prior_nlp_events = []

    user_entry = {"role": "user", "content": body.content.strip(), "at": now}
    nlp_analysis = _analyze_message_nlp(
        user_text=user_entry["content"],
        history=messages,
        mode=mode,
    )

    action_items = list(nlp_analysis.get("action_items", []))
    needs_clarification = "needs_intent_clarification" in action_items
    prior_clarification_requested = bool(
        prior_nlp_events and prior_nlp_events[-1].get("clarification_requested")
    )
    clarification_resolved = prior_clarification_requested and not needs_clarification

    nlp_event = {
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
        "action_items": action_items,
        "safety_flags": nlp_analysis.get("safety_flags", []),
        "rewritten_query": nlp_analysis.get("rewritten_query", user_entry["content"]),
        "confidence": nlp_analysis.get("confidence", 0.0),
        "taxonomy_version": nlp_analysis.get("taxonomy_version"),
        "explainability": nlp_analysis.get("explainability", {}),
        "clarification_requested": needs_clarification,
        "clarification_follow_up": prior_clarification_requested,
        "clarification_resolved": clarification_resolved,
    }

    if needs_clarification:
        reply_text = _build_clarification_reply(mode=mode, nlp_analysis=nlp_analysis)
    else:
        retrieval_query = nlp_analysis.get("rewritten_query") or user_entry["content"]

        groq_messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPTS[mode]}]

        safety_flags = nlp_analysis.get("safety_flags", [])
        if safety_flags:
            groq_messages.append(
                {
                    "role": "system",
                    "content": (
                        "Safety note: The user request may include clinical-risk patterns "
                        f"({', '.join(safety_flags)}). Provide general educational information only, "
                        "avoid patient-specific medical advice, and recommend official labeling or specialist consultation when needed."
                    ),
                }
            )

        nlp_topics = nlp_analysis.get("topics", [])
        nlp_entities = nlp_analysis.get("entities", [])
        if nlp_topics or nlp_entities:
            groq_messages.append(
                {
                    "role": "system",
                    "content": (
                        f"NLP intent: {nlp_analysis.get('intent', 'other')}\n"
                        f"NLP entities: {', '.join(nlp_entities) if nlp_entities else 'none'}\n"
                        f"NLP topics: {', '.join(nlp_topics) if nlp_topics else 'none'}"
                    ),
                }
            )

        context = ""
        try:
            context = await rag.get_context(
                query=retrieval_query,
                top_k=3,
                min_score=0.3,
            )

            if context:
                groq_messages.append({"role": "system", "content": context})
                logger.info("RAG context added to message")
        except Exception as e:
            logger.warning(f"Failed to retrieve RAG context: {e}")

        # Add conversation history
        for m in messages:
            groq_messages.append({"role": m["role"], "content": m["content"]})
        groq_messages.append({"role": "user", "content": user_entry["content"]})

        reply_text = await _chat_completion(groq_messages)
    asst_time = datetime.utcnow()
    asst_entry = {"role": "assistant", "content": reply_text, "at": asst_time}

    messages.append(user_entry)
    messages.append(asst_entry)

    if doc is None:
        insert_doc = {
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
        }
        result = await db.conversations.insert_one(insert_doc)
        session_id = str(result.inserted_id)
    else:
        update_set = {"messages": messages, "updated_at": asst_time}
        if reopened_closed_session:
            # Reopening invalidates previous rollups/evaluation so finalize can rebuild them.
            update_set.update(
                {
                    "status": "open",
                    "summary": None,
                    "summary_created_at": None,
                    "summary_method": None,
                    "summary_triggered_by": None,
                    "rolling_summaries": [],
                    "nlp_evaluation": None,
                    "competency_level": None,
                    "evaluation_score": None,
                    "evaluation_dimensions": {},
                    "evaluation_strengths": [],
                    "evaluation_gaps": [],
                    "evaluation_notes": [],
                    "evaluation_completed_at": None,
                }
            )
        await db.conversations.update_one(
            {"_id": oid},
            {
                "$set": update_set,
                "$push": {"nlp_events": nlp_event},
            },
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
    doc, oid = await _get_owned_session(db, session_id, current_user.email)
    
    # If already summarized and not forcing regeneration, return cached summary
    if doc.get("summary") and not force_regenerate:
        return FinalizeResponse(session_id=session_id, summary=doc["summary"])

    msgs = doc.get("messages") or []
    
    # Generate summary with caching, error handling, and incremental summaries
    summary, metadata, rolling_summaries = await generate_summary_with_caching(
        session_id=session_id,
        messages=msgs,
        force_regenerate=force_regenerate,
    )

    nlp_events = list(doc.get("nlp_events") or [])
    nlp_evaluation = evaluate_conversation(
        messages=msgs,
        summary=summary,
        metadata=metadata,
        nlp_events=nlp_events,
    )
    
    now = datetime.utcnow()
    await db.conversations.update_one(
        {"_id": oid},
        {
            "$set": {
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
            }
        },
    )
    
    return FinalizeResponse(session_id=session_id, summary=summary)


@router.get("/sessions", response_model=List[SessionListItem])
async def list_sessions(
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
    limit: int = 50,
    intent: Optional[str] = None,
    has_safety_flags: Optional[bool] = None,
):
    limit = min(max(limit, 1), 100)

    query: dict[str, Any] = {"user_email": current_user.email}

    filters = []
    if intent:
        if intent not in SUPPORTED_INTENTS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Invalid intent filter. Supported intents: "
                    f"{', '.join(sorted(SUPPORTED_INTENTS))}"
                ),
            )
        filters.append({"nlp_events": {"$elemMatch": {"intent": intent}}})

    if has_safety_flags is True:
        filters.append({"nlp_events": {"$elemMatch": {"safety_flags.0": {"$exists": True}}}})
    elif has_safety_flags is False:
        filters.append({"$nor": [{"nlp_events": {"$elemMatch": {"safety_flags.0": {"$exists": True}}}}]})

    if filters:
        query["$and"] = filters

    cursor = (
        db.conversations.find(query)
        .sort("updated_at", -1)
        .limit(limit)
    )
    items: list[SessionListItem] = []
    async for doc in cursor:
        summary = doc.get("summary")
        msgs = doc.get("messages") or []
        nlp_events = doc.get("nlp_events") or []
        last_nlp = nlp_events[-1] if nlp_events else {}
        if summary:
            preview = summary[:220].replace("\n", " ")
        elif msgs:
            preview = (msgs[-1].get("content") or "")[:220]
        else:
            preview = ""
        items.append(
            SessionListItem(
                id=str(doc["_id"]),
                mode=doc.get("mode", ""),
                created_at=doc["created_at"],
                updated_at=doc["updated_at"],
                summary=summary,
                summary_created_at=doc.get("summary_created_at"),
                rolling_summaries=doc.get("rolling_summaries", []),
                status=doc.get("status", "open"),
                preview=preview,
                nlp_event_count=len(nlp_events),
                last_intent=last_nlp.get("intent"),
                last_confidence=last_nlp.get("confidence"),
                last_safety_flags=last_nlp.get("safety_flags", []),
                last_level=doc.get("competency_level"),
                last_score=doc.get("evaluation_score"),
            )
        )
    return items


@router.get("/sessions/{session_id}", response_model=ConversationResponse)
async def get_session(
    session_id: str,
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    doc, _ = await _get_owned_session(db, session_id, current_user.email)
    return ConversationResponse(**doc)


@router.post("/nlp-debug", response_model=NLPDebugResponse)
async def nlp_debug(
    body: NLPDebugRequest,
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    mode = _ensure_mode(body.mode)
    history = []

    if body.session_id:
        doc, _ = await _get_owned_session(db, body.session_id, current_user.email)
        history = list(doc.get("messages") or [])

    analysis = _analyze_message_nlp(
        user_text=body.content.strip(),
        history=history,
        mode=mode,
    )
    return NLPDebugResponse(analysis=analysis)


@router.post("/rep-score", response_model=RepScoreResponse)
async def rep_score(
    body: RepScoreRequest,
    current_user: UserInDB = Depends(get_current_user),
    rep_scoring_service: RepScoringService = Depends(get_rep_scoring_service),
):
    _ = current_user
    result = rep_scoring_service.score_response(body.content)
    return RepScoreResponse(**result)
