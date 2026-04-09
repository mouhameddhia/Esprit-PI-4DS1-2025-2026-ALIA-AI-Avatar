from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

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

class UserResponse(UserBase):
    id: str = Field(..., alias="_id")
    created_at: datetime

    model_config = {
        "populate_by_name": True,
    }

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None
