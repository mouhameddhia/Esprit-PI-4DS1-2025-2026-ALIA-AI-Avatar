from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any, Literal
from datetime import datetime
from bson import ObjectId


def _objectid_to_str(v: Any) -> Any:
    if isinstance(v, ObjectId):
        return str(v)
    return v


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
    secondary_tags: List[str] = []
    entities: List[str] = []
    entity_map: dict[str, List[str]] = {}
    topics: List[str] = []
    objections: List[str] = []
    action_items: List[str] = []
    safety_flags: List[str] = []
    rewritten_query: str
    confidence: float
    taxonomy_version: Optional[str] = None
    explainability: dict[str, Any] = Field(default_factory=dict)


class ConversationResponse(BaseModel):
    id: str = Field(..., alias="_id")
    user_email: str
    mode: str
    messages: List[MessageEntry] = []
    summary: Optional[str] = None
    summary_created_at: Optional[datetime] = None
    summary_method: Optional[Literal["auto", "manual"]] = None
    summary_triggered_by: Optional[str] = None
    rolling_summaries: List[RollingSummaryEntry] = []
    nlp_events: List[NLPEventEntry] = []
    competency_level: Optional[str] = None
    evaluation_score: Optional[float] = None
    evaluation_dimensions: dict = {}
    evaluation_strengths: List[str] = []
    evaluation_gaps: List[str] = []
    evaluation_notes: List[str] = []
    evaluation_completed_at: Optional[datetime] = None
    topics: List[str] = []
    objections: List[str] = []
    action_items: List[str] = []
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = {"populate_by_name": True}

    @field_validator("id", mode="before")
    @classmethod
    def _id_from_objectid(cls, v: Any) -> Any:
        return _objectid_to_str(v)


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
    last_level: Optional[str] = None
    last_score: Optional[float] = None
