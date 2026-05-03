"""Users management — admin only."""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel

from ..dependencies import get_database, require_roles
from ..models.user import UserInDB, UserResponse
from ..utils.mongo import object_id

router = APIRouter(prefix="/users", tags=["users-admin"])

_admin = Depends(require_roles("admin"))

VALID_ROLES = {"admin", "medrep", "physician"}


class UpdateRoleRequest(BaseModel):
    role: str


@router.get("", response_model=List[UserResponse])
async def list_users(
    search: Optional[str] = Query(None, description="Search by name or email"),
    role: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    current_user: UserInDB = _admin,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    query: dict = {}
    if search:
        query["$or"] = [
            {"name": {"$regex": search, "$options": "i"}},
            {"email": {"$regex": search, "$options": "i"}},
        ]
    if role:
        query["role"] = role.lower()

    cursor = db.users.find(query, {"hashed_password": 0}).sort("created_at", -1).skip(skip).limit(limit)
    return [UserResponse(**u) for u in await cursor.to_list(length=limit)]


@router.put("/{user_id}", response_model=UserResponse)
async def update_user_role(
    user_id: str,
    payload: UpdateRoleRequest,
    current_user: UserInDB = _admin,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    new_role = payload.role.lower().strip()
    if new_role not in VALID_ROLES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Must be one of: {', '.join(sorted(VALID_ROLES))}",
        )

    oid = object_id(user_id, "user")
    result = await db.users.update_one(
        {"_id": oid},
        {"$set": {"role": new_role, "updated_at": datetime.utcnow()}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return UserResponse(**await db.users.find_one({"_id": oid}))


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: str,
    current_user: UserInDB = _admin,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    if current_user.id == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot delete your own account",
        )

    oid = object_id(user_id, "user")
    result = await db.users.delete_one({"_id": oid})
    if result.deleted_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
