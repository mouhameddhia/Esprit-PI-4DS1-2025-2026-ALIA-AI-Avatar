from typing import Optional, Any
from datetime import datetime

from pydantic import BaseModel, model_validator

from .common import coerce_mongo_id


class UserBase(BaseModel):
    email: str
    name: str
    role: str  # "admin" | "medrep" | "physician"


class UserCreate(UserBase):
    password: str


class UserInDB(UserBase):
    id: str
    hashed_password: str
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="before")
    @classmethod
    def _coerce_id(cls, data: Any) -> Any:
        return coerce_mongo_id(data)


class UserResponse(UserBase):
    id: str
    created_at: datetime

    @model_validator(mode="before")
    @classmethod
    def _coerce_id(cls, data: Any) -> Any:
        return coerce_mongo_id(data)


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None
