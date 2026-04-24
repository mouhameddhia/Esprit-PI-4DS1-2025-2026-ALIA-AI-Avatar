import base64
import json
import secrets

from fastapi import APIRouter, Depends, HTTPException, Header, Query, status
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from authlib.integrations.requests_client import OAuth2Session
from motor.motor_asyncio import AsyncIOMotorDatabase
from .. import config
from ..dependencies import get_database, get_current_user, require_roles
from ..models.user import UserInDB, UserResponse, Token, UserCreate
from ..utils.auth import create_access_token, verify_password, get_password_hash
from pydantic import BaseModel
from jwt import PyJWKClient
import jwt
from datetime import datetime

_ALLOWED_ROLES = {"medrep", "physician"}

router = APIRouter()

def get_auth0_config():
    if not config.AUTH0_DOMAIN:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Auth0 domain not configured")
    if not config.AUTH0_CLIENT_ID:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Auth0 client id not configured")

    return {
        "domain": config.AUTH0_DOMAIN,
        "issuer": f"https://{config.AUTH0_DOMAIN}/",
        "authorize_url": f"https://{config.AUTH0_DOMAIN}/authorize",
        "token_url": f"https://{config.AUTH0_DOMAIN}/oauth/token",
        "userinfo_url": f"https://{config.AUTH0_DOMAIN}/userinfo",
        "callback_url": config.AUTH0_CALLBACK_URL,
        "client_id": config.AUTH0_CLIENT_ID,
        "client_secret": config.AUTH0_CLIENT_SECRET,
    }

@router.get("/login")
async def login(role: str = Query("medrep")):
    role = role.lower().replace(" ", "")
    if role not in _ALLOWED_ROLES:
        role = "medrep"

    auth0_cfg = get_auth0_config()

    # Encode role + CSRF nonce in state so the callback can read the role
    # without a second round-trip to the frontend.
    state_payload = {"nonce": secrets.token_urlsafe(16), "role": role}
    state = base64.urlsafe_b64encode(json.dumps(state_payload).encode()).decode()

    auth0_oauth = OAuth2Session(
        client_id=auth0_cfg["client_id"],
        client_secret=auth0_cfg["client_secret"],
        redirect_uri=auth0_cfg["callback_url"],
        scope="openid profile email",
    )
    authorization_url, _ = auth0_oauth.create_authorization_url(
        auth0_cfg["authorize_url"],
        state=state,
    )
    return RedirectResponse(authorization_url)

@router.get("/callback", response_model=Token)
async def auth_callback(code: str, state: str, db: AsyncIOMotorDatabase = Depends(get_database)):
    # Decode role from state (falls back to "medrep" if state is not our format)
    role = "medrep"
    try:
        state_payload = json.loads(base64.urlsafe_b64decode(state.encode()).decode())
        decoded_role = state_payload.get("role", "medrep").lower().replace(" ", "")
        if decoded_role in _ALLOWED_ROLES:
            role = decoded_role
    except Exception:
        pass

    auth0_cfg = get_auth0_config()
    auth0_oauth = OAuth2Session(
        client_id=auth0_cfg["client_id"],
        client_secret=auth0_cfg["client_secret"],
        redirect_uri=auth0_cfg["callback_url"],
        scope="openid profile email",
    )
    auth0_oauth.fetch_token(auth0_cfg["token_url"], code=code)
    user_info = auth0_oauth.get(auth0_cfg["userinfo_url"]).json()

    email = user_info["email"]
    name = user_info.get("name", "")
    now = datetime.utcnow()

    user = await db.users.find_one({"email": email})
    if not user:
        result = await db.users.insert_one({
            "email": email,
            "name": name,
            "role": role,
            "hashed_password": "",
            "created_at": now,
            "updated_at": now,
        })
        user = await db.users.find_one({"_id": result.inserted_id})
    else:
        await db.users.update_one(
            {"email": email},
            {"$set": {"role": role, "name": name, "updated_at": now}},
        )
        user = await db.users.find_one({"email": email})

    access_token = create_access_token(data={"sub": user["email"]})
    return Token(access_token=access_token, token_type="bearer")

class Auth0SyncRequest(BaseModel):
    role: str
    name: str | None = None


class Auth0SyncResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse


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


def verify_auth0_token(auth_token: str):
    """
    Accept either an Auth0 ID token (preferred) or an Auth0 access token.
    If JWT verification fails, fall back to /userinfo lookup.
    """
    try:
        return verify_auth0_id_token(auth_token)
    except HTTPException:
        pass

    config = get_auth0_config()
    try:
        auth0_oauth = OAuth2Session(token={"access_token": auth_token, "token_type": "Bearer"})
        user_info_response = auth0_oauth.get(config["userinfo_url"])
        if user_info_response.status_code != 200:
            raise ValueError(f"userinfo request failed with status {user_info_response.status_code}")
        payload = user_info_response.json()
        if not payload.get("email"):
            raise ValueError("Auth0 token missing email")
        return payload
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid Auth0 token: {exc}",
        )


@router.post("/auth0-sync", response_model=Auth0SyncResponse)
async def auth0_sync_profile(
    sync_data: Auth0SyncRequest,
    authorization: str | None = Header(None),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Auth0 ID token")

    auth_token = authorization.split(" ", 1)[1]
    payload = verify_auth0_token(auth_token)
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

    access_token = create_access_token(data={"sub": user["email"]})
    return Auth0SyncResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(**user),
    )

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
