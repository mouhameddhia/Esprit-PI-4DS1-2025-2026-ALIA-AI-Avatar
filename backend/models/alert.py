from typing import Optional, Any
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, model_validator

from .common import coerce_mongo_id


class AlertSeverity(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class AlertStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"


class AlertBase(BaseModel):
    user_message: str
    detected_intent: str
    missing_entity: Optional[str] = None
    missing_info: Optional[str] = None
    severity: AlertSeverity = AlertSeverity.MEDIUM
    status: AlertStatus = AlertStatus.OPEN
    source_conversation_id: Optional[str] = None


class AlertCreate(AlertBase):
    pass


class AlertInDB(AlertBase):
    id: str
    timestamp: datetime
    resolved_at: Optional[datetime] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_id(cls, data: Any) -> Any:
        return coerce_mongo_id(data)


class AlertResponse(AlertBase):
    id: str
    timestamp: datetime
    resolved_at: Optional[datetime] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_id(cls, data: Any) -> Any:
        return coerce_mongo_id(data)
