from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class ProductBase(BaseModel):
    name: str
    description: str
    category: str
    indications: List[str]
    contraindications: List[str]
    dosage: str

class ProductCreate(ProductBase):
    pass

class ProductInDB(ProductBase):
    id: str
    created_at: datetime
    updated_at: datetime

class ProductResponse(ProductBase):
    id: str
    created_at: datetime</content>
<parameter name="filePath">c:\Users\moham\Desktop\alia-web-main\backend\models\product.py