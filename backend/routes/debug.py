"""Debug and scoring endpoints (NLP inspection, rep scoring)."""

import logging
from typing import Literal, Optional

from fastapi import APIRouter, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel, Field

from ..dependencies import get_database, get_current_user, get_rep_scoring_service
from ..models.user import UserInDB
from ..services import RepScoringService
from ..utils.chat_helpers import analyze_message_nlp, ensure_mode, get_owned_session

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/nlp-debug", response_model=NLPDebugResponse)
async def nlp_debug(
    body: NLPDebugRequest,
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    mode = ensure_mode(body.mode)
    history = []

    if body.session_id:
        doc, _ = await get_owned_session(db, body.session_id, current_user.email)
        history = list(doc.get("messages") or [])

    analysis = analyze_message_nlp(
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
    result = rep_scoring_service.score_response(body.content)
    return RepScoreResponse(**result)
