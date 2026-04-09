from motor.motor_asyncio import AsyncIOMotorDatabase
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .utils.auth import verify_token
from .models.user import UserInDB

security = HTTPBearer()

def get_database() -> AsyncIOMotorDatabase:
    from .main import db
    return db

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: AsyncIOMotorDatabase = Depends(get_database)) -> UserInDB:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    token = credentials.credentials
    email = verify_token(token, credentials_exception)
    user = await db.users.find_one({"email": email})
    if user is None:
        raise credentials_exception
    return UserInDB(**user)


def require_roles(*allowed_roles: str):
    async def role_checker(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Operation not permitted for your role",
            )
        return current_user
    return role_checker
