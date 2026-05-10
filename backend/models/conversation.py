from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any, Literal
from datetime import datetime
from bson import ObjectId


def _objectid_to_str(v: Any) -> Any:
    if isinstance(v, ObjectId):
        return str(v)
    return v


# ── Sub-document models ───────────────────────────────────────────────────────

class MessageEntry(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    at: datetime


class RollingSummaryEntry(BaseModel):
    summary: str
    generated_at: datetime
    message_count: int


class NLPEventEntry(BaseModel):
    at: datetime
    mode: str
    message: str
    intent: str
    confidence: float
    language: Optional[str] = None
    secondary_tags: List[str] = []
    entities: List[str] = []
    entity_map: dict[str, List[str]] = {}
    topics: List[str] = []
    objections: List[str] = []
    action_items: List[str] = []
    safety_flags: List[str] = []
    rewritten_query: str = ""
    taxonomy_version: Optional[str] = None
    explainability: dict[str, Any] = Field(default_factory=dict)
    affect: dict[str, Any] = Field(default_factory=dict)
    audio_affect: dict[str, Any] = Field(default_factory=dict)
    clarification_requested: bool = False
    clarification_follow_up: bool = False
    clarification_resolved: bool = False


# ── Shared base ───────────────────────────────────────────────────────────────

class _ConversationBase(BaseModel):
    """Fields common to both physician and MedRep conversations."""
    id: str = Field(..., alias="_id")
    user_email: str
    mode: str
    status: str
    created_at: datetime
    updated_at: datetime

    messages: List[MessageEntry] = []
    nlp_events: List[NLPEventEntry] = []

    summary: Optional[str] = None
    summary_created_at: Optional[datetime] = None
    summary_method: Optional[Literal["auto", "manual"]] = None
    summary_triggered_by: Optional[str] = None
    rolling_summaries: List[RollingSummaryEntry] = []

    topics: List[str] = []
    action_items: List[str] = []

    model_config = {"populate_by_name": True}

    @field_validator("id", mode="before")
    @classmethod
    def _id_from_objectid(cls, v: Any) -> Any:
        return _objectid_to_str(v)


# ── Physician Portal conversation ─────────────────────────────────────────────

class PhysicianConversationResponse(_ConversationBase):
    """
    Physician conversations store clinical Q&A.
    No sales competency evaluation — physicians are not trainees.
    """
    # Physicians may have informational action items but not sales objections.
    # objections field intentionally absent.
    pass


# ── MedRep Training conversation ──────────────────────────────────────────────

class MedRepConversationResponse(_ConversationBase):
    """
    MedRep training conversations include sales simulation.
    Adds competency evaluation written by finalize_session.
    """
    objections: List[str] = []

    # Competency evaluation — populated by POST /chat/sessions/{id}/finalize
    competency_level: Optional[str] = None          # Debutant | Junior | Confirme | Expert
    evaluation_score: Optional[float] = None         # 0.0–10.0
    evaluation_dimensions: dict[str, Any] = {}      # 10 sales dimensions, each 0–10
    evaluation_strengths: List[str] = []             # dimension names scoring ≥ 8
    evaluation_gaps: List[str] = []                  # dimension names scoring < 7
    evaluation_notes: List[str] = []                 # coaching notes
    evaluation_affect_summary: dict[str, Any] = {}  # avg confidence, frustration ratio, etc.
    evaluation_completed_at: Optional[datetime] = None


# ── Generic fallback (used by routes that don't know the mode yet) ────────────

class ConversationResponse(_ConversationBase):
    """
    Generic response model used by the sessions list endpoint.
    Includes all optional fields from both modes; absent fields return None/[].
    """
    objections: List[str] = []
    competency_level: Optional[str] = None
    evaluation_score: Optional[float] = None
    evaluation_dimensions: dict = {}
    evaluation_strengths: List[str] = []
    evaluation_gaps: List[str] = []
    evaluation_notes: List[str] = []
    evaluation_affect_summary: dict = {}
    evaluation_completed_at: Optional[datetime] = None


# ── Session list item ─────────────────────────────────────────────────────────

class SessionListItem(BaseModel):
    id: str
    mode: str
    created_at: datetime
    updated_at: datetime
    summary: Optional[str] = None
    summary_created_at: Optional[datetime] = None
    rolling_summaries: List[RollingSummaryEntry] = []
    status: str
    preview: str
    nlp_event_count: int = 0
    last_intent: Optional[str] = None
    last_confidence: Optional[float] = None
    last_safety_flags: List[str] = []
    # MedRep-only — None for physician sessions
    last_level: Optional[str] = None
    last_score: Optional[float] = None
