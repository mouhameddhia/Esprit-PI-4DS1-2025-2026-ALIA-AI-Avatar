"""Session listing and retrieval routes."""

import logging
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from ..dependencies import get_database, get_current_user
from ..models.conversation import ConversationResponse, SessionListItem
from ..models.user import UserInDB
from ..utils.chat_helpers import get_owned_session
from ..utils.nlp import SUPPORTED_INTENTS

logger = logging.getLogger(__name__)

router = APIRouter()


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
                    "Invalid intent filter. Supported: "
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

    cursor = db.conversations.find(query).sort("updated_at", -1).limit(limit)

    items: list[SessionListItem] = []
    async for doc in cursor:
        nlp_events = doc.get("nlp_events") or []
        last_nlp = nlp_events[-1] if nlp_events else {}
        msgs = doc.get("messages") or []
        summary = doc.get("summary")

        if summary:
            preview = summary[:220].replace("\n", " ")
        elif msgs:
            preview = (msgs[-1].get("content") or "")[:220]
        else:
            preview = ""

        items.append(SessionListItem(
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
        ))

    return items


@router.get("/sessions/{session_id}", response_model=ConversationResponse)
async def get_session(
    session_id: str,
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    doc, _ = await get_owned_session(db, session_id, current_user.email)
    return ConversationResponse(**doc)
