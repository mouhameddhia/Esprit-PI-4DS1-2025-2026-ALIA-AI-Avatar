from typing import Optional, List, Any
from datetime import datetime

from pydantic import BaseModel, model_validator

from .common import coerce_mongo_id


class ProductBase(BaseModel):
    name: str
    description: str
    category: str
    indications: List[str]
    contraindications: List[str]
    dosage: str


class ProductCreate(ProductBase):
    pass


class ProductUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    indications: Optional[List[str]] = None
    contraindications: Optional[List[str]] = None
    dosage: Optional[str] = None


class ProductInDB(ProductBase):
    id: str
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="before")
    @classmethod
    def _coerce_id(cls, data: Any) -> Any:
        return coerce_mongo_id(data)


class ProductResponse(ProductBase):
    id: str
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="before")
    @classmethod
    def _coerce_id(cls, data: Any) -> Any:
        return coerce_mongo_id(data)
