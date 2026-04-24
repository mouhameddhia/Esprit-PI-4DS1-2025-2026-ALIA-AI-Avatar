from motor.motor_asyncio import AsyncIOMotorDatabase
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .utils.auth import verify_token
from .models.user import UserInDB

security = HTTPBearer()

def get_database() -> AsyncIOMotorDatabase:
    from .main import db
    return db

def get_vector_client():
    """Get the vector database client."""
    from .main import vector_client
    return vector_client

def get_embedding_encoder():
    """Get the embedding encoder."""
    from .main import embedding_encoder
    return embedding_encoder

def get_product_indexer():
    """Get the product indexer."""
    from .main import product_indexer
    return product_indexer

def get_conversation_embedder():
    """Get the conversation embedder."""
    from .main import conversation_embedder
    return conversation_embedder

def get_knowledge_document_indexer():
    """Get the knowledge document indexer."""
    from .main import knowledge_document_indexer
    return knowledge_document_indexer


def get_rag_pipeline():
    """Get the RAG pipeline singleton."""
    from .main import rag_pipeline
    return rag_pipeline

def get_rep_scoring_service():
    """Get the representative response scoring service singleton."""
    from .services import get_rep_scoring_service as _get_service
    return _get_service()

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
