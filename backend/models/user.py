from pydantic import BaseModel, Field, field_validator
from typing import Optional, Any
from datetime import datetime
from bson import ObjectId


def _objectid_to_str(v: Any) -> Any:
    if isinstance(v, ObjectId):
        return str(v)
    return v

class UserBase(BaseModel):
    email: str
    name: str
    role: str  # "medrep" or "physician"

class UserCreate(UserBase):
    password: str

class UserInDB(UserBase):
    id: str = Field(..., alias="_id")
    hashed_password: str
    created_at: datetime
    updated_at: datetime

    model_config = {
        "populate_by_name": True,
    }

    @field_validator("id", mode="before")
    @classmethod
    def _id_from_objectid(cls, v: Any) -> Any:
        return _objectid_to_str(v)

class UserResponse(UserBase):
    id: str = Field(..., alias="_id")
    created_at: datetime

    model_config = {
        "populate_by_name": True,
    }

    @field_validator("id", mode="before")
    @classmethod
    def _id_from_objectid(cls, v: Any) -> Any:
        return _objectid_to_str(v)

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None
