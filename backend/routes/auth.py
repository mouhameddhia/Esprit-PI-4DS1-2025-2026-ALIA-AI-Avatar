from fastapi import APIRouter, Depends, HTTPException, Header, status
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from authlib.integrations.requests_client import OAuth2Session
import os
from motor.motor_asyncio import AsyncIOMotorDatabase
from ..dependencies import get_database, get_current_user, require_roles
from ..models.user import UserInDB, UserResponse, Token, UserCreate
from ..utils.auth import create_access_token, verify_password, get_password_hash
from pydantic import BaseModel
from jwt import PyJWKClient
import jwt
from datetime import datetime

router = APIRouter()

def get_auth0_config():
    domain = os.getenv("AUTH0_DOMAIN")
    client_id = os.getenv("AUTH0_CLIENT_ID")
    client_secret = os.getenv("AUTH0_CLIENT_SECRET")
    if not domain:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Auth0 domain not configured")
    if not client_id:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Auth0 client id not configured")

    return {
        "domain": domain,
        "issuer": f"https://{domain}/",
        "authorize_url": f"https://{domain}/authorize",
        "token_url": f"https://{domain}/oauth/token",
        "userinfo_url": f"https://{domain}/userinfo",
        "client_id": client_id,
        "client_secret": client_secret,
    }

@router.get("/login")
async def login():
    config = get_auth0_config()
    auth0_oauth = OAuth2Session(
        client_id=config["client_id"],
        client_secret=config["client_secret"],
        redirect_uri="http://localhost:8000/auth/callback",
        scope="openid profile email",
    )
    authorize_url = config["authorize_url"]
    authorization_url, state = auth0_oauth.create_authorization_url(authorize_url)
    return RedirectResponse(authorization_url)

@router.get("/callback", response_model=Token)
async def auth_callback(code: str, state: str, db: AsyncIOMotorDatabase = Depends(get_database)):
    config = get_auth0_config()
    auth0_oauth = OAuth2Session(
        client_id=config["client_id"],
        client_secret=config["client_secret"],
        redirect_uri="http://localhost:8000/auth/callback",
        scope="openid profile email",
    )
    token_url = config["token_url"]
    userinfo_url = config["userinfo_url"]
    token = auth0_oauth.fetch_token(token_url, code=code)
    user_info_response = auth0_oauth.get(userinfo_url)
    user_info = user_info_response.json()

    user = await db.users.find_one({"email": user_info["email"]})
    if not user:
        user_data = {
            "email": user_info["email"],
            "name": user_info.get("name", ""),
            "role": "medrep",
            "hashed_password": "",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        result = await db.users.insert_one(user_data)
        user = await db.users.find_one({"_id": result.inserted_id})

    access_token = create_access_token(data={"sub": user["email"]})
    return Token(access_token=access_token, token_type="bearer")

class Auth0SyncRequest(BaseModel):
    role: str
    name: str | None = None


def verify_auth0_id_token(id_token: str):
    config = get_auth0_config()
    issuer = config["issuer"]
    jwks_url = f"{issuer}.well-known/jwks.json"
    try:
        jwk_client = PyJWKClient(jwks_url)
        signing_key = jwk_client.get_signing_key_from_jwt(id_token)
        payload = jwt.decode(
            id_token,
            signing_key.key,
            algorithms=["RS256"],
            audience=config["client_id"],
            issuer=issuer,
        )
        return payload
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid Auth0 token: {exc}")


@router.post("/auth0-sync", response_model=UserResponse)
async def auth0_sync_profile(
    sync_data: Auth0SyncRequest,
    authorization: str | None = Header(None),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Auth0 ID token")

    id_token = authorization.split(" ", 1)[1]
    payload = verify_auth0_id_token(id_token)
    email = payload.get("email")
    if not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Auth0 token missing email")

    name = sync_data.name or payload.get("name") or ""
    role = sync_data.role.lower().replace(" ", "")

    user = await db.users.find_one({"email": email})
    now = datetime.utcnow()
    if not user:
        user_data = {
            "email": email,
            "name": name,
            "role": role,
            "hashed_password": "",
            "created_at": now,
            "updated_at": now,
        }
        result = await db.users.insert_one(user_data)
        user = await db.users.find_one({"_id": result.inserted_id})
    else:
        await db.users.update_one(
            {"email": email},
            {"$set": {"name": name, "role": role, "updated_at": now}},
        )
        user = await db.users.find_one({"email": email})

    return UserResponse(**user)

@router.post("/login/jwt", response_model=Token)
async def login_jwt(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncIOMotorDatabase = Depends(get_database)):
    user = await db.users.find_one({"email": form_data.username})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user["email"]})
    return Token(access_token=access_token, token_type="bearer")

@router.post("/signup", response_model=Token)
async def signup(user: UserCreate, db: AsyncIOMotorDatabase = Depends(get_database)):
    # Check if user already exists
    existing_user = await db.users.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Hash the password
    hashed_password = get_password_hash(user.password)
    
    # Create user
    user_data = {
        "email": user.email,
        "name": user.name,
        "role": user.role.lower().replace(" ", ""),
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    result = await db.users.insert_one(user_data)
    user_doc = await db.users.find_one({"_id": result.inserted_id})
    
    # Create access token
    access_token = create_access_token(data={"sub": user.email})
    return Token(access_token=access_token, token_type="bearer")

@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: UserInDB = Depends(get_current_user)):
    return current_user

@router.get("/protected/medrep")
async def medrep_area(current_user: UserInDB = Depends(require_roles("medrep"))):
    return {
        "message": "MedRep access granted",
        "email": current_user.email,
        "role": current_user.role,
    }
