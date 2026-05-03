"""Products CRUD — admin only."""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from ..dependencies import get_database, require_roles
from ..models.product import ProductCreate, ProductUpdate, ProductResponse
from ..models.user import UserInDB
from ..utils.mongo import object_id

router = APIRouter(prefix="/products", tags=["products"])

_admin = Depends(require_roles("admin"))


@router.get("", response_model=List[ProductResponse])
async def list_products(
    search: Optional[str] = Query(None, description="Search in name / description / category"),
    category: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    query: dict = {}
    if search:
        query["$or"] = [
            {"name": {"$regex": search, "$options": "i"}},
            {"description": {"$regex": search, "$options": "i"}},
            {"category": {"$regex": search, "$options": "i"}},
        ]
    if category:
        query["category"] = {"$regex": f"^{category}$", "$options": "i"}

    cursor = db.products.find(query).sort("created_at", -1).skip(skip).limit(limit)
    return [ProductResponse(**p) for p in await cursor.to_list(length=limit)]


@router.post("", response_model=ProductResponse, status_code=status.HTTP_201_CREATED)
async def create_product(
    payload: ProductCreate,
    current_user: UserInDB = _admin,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    now = datetime.utcnow()
    result = await db.products.insert_one({**payload.model_dump(), "created_at": now, "updated_at": now})
    created = await db.products.find_one({"_id": result.inserted_id})
    return ProductResponse(**created)


@router.put("/{product_id}", response_model=ProductResponse)
async def update_product(
    product_id: str,
    payload: ProductUpdate,
    current_user: UserInDB = _admin,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    oid = object_id(product_id, "product")
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields to update")

    updates["updated_at"] = datetime.utcnow()
    result = await db.products.update_one({"_id": oid}, {"$set": updates})
    if result.matched_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")

    return ProductResponse(**await db.products.find_one({"_id": oid}))


@router.delete("/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_product(
    product_id: str,
    current_user: UserInDB = _admin,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    oid = object_id(product_id, "product")
    result = await db.products.delete_one({"_id": oid})
    if result.deleted_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")
