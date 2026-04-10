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


class ConversationResponse(BaseModel):
    id: str = Field(..., alias="_id")
    user_email: str
    mode: str
    messages: List[MessageEntry] = []
    summary: Optional[str] = None
    summary_created_at: Optional[datetime] = None
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
    status: str
    preview: str
