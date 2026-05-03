"""
Centralized application configuration.

All environment variables are read here once. Everywhere else in the codebase
imports from this module instead of calling os.getenv() directly, so defaults
and types are defined in a single place.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load the .env file that lives next to this module (backend/.env).
# Calling load_dotenv() multiple times is safe — it is a no-op if the env is
# already populated, unless override=True is passed.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

# ---------------------------------------------------------------------------
# LLM (Groq)
# ---------------------------------------------------------------------------
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ---------------------------------------------------------------------------
# Vector database (Pinecone)
# ---------------------------------------------------------------------------
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "alia-knowledge")
VECTOR_DB_TYPE: str = os.getenv("VECTOR_DB_TYPE", "pinecone")

# ---------------------------------------------------------------------------
# Databases
# ---------------------------------------------------------------------------
MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017/alia")
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "384"))

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------
_TRUTHY = {"1", "true", "yes", "on"}

USE_HYBRID_ADAPTER: bool = os.getenv("ALIA_USE_HYBRID_ADAPTER", "0").lower() in _TRUTHY
USE_ENTITY_EXTRACTOR_V2: bool = os.getenv("ALIA_USE_ENTITY_EXTRACTOR_V2", "0").lower() in _TRUTHY
HYBRID_MAX_NEW_TOKENS: int = int(os.getenv("ALIA_HYBRID_MAX_NEW_TOKENS", "220"))

# ---------------------------------------------------------------------------
# Rep scoring
# ---------------------------------------------------------------------------
REP_SCORER_ARTIFACT_DIR: str = os.getenv("REP_SCORER_ARTIFACT_DIR", "")

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
JWT_SECRET: str = os.getenv("JWT_SECRET", "your_secret_key")
# Secret key required to self-register as admin. Empty string = admin signup disabled.
ADMIN_SECRET_KEY: str = os.getenv("ADMIN_SECRET_KEY", "")
AUTH0_DOMAIN: str = os.getenv("AUTH0_DOMAIN", "")
AUTH0_CLIENT_ID: str = os.getenv("AUTH0_CLIENT_ID", "")
AUTH0_CLIENT_SECRET: str = os.getenv("AUTH0_CLIENT_SECRET", "")
AUTH0_CALLBACK_URL: str = os.getenv("AUTH0_CALLBACK_URL", "http://localhost:8000/auth/callback")

# ---------------------------------------------------------------------------
# Background task scheduler
# ---------------------------------------------------------------------------
INACTIVITY_THRESHOLD_MINUTES: int = int(os.getenv("INACTIVITY_THRESHOLD_MINUTES", "60"))
SHADOW_LOG_LOOKBACK_DAYS: int = int(os.getenv("SHADOW_LOG_LOOKBACK_DAYS", "7"))
SHADOW_LOG_MAX_ROWS: int = int(os.getenv("SHADOW_LOG_MAX_ROWS", "500"))
SHADOW_MAX_DIVERGENCE: float = float(os.getenv("SHADOW_MAX_DIVERGENCE", "0.08"))
SHADOW_JOB_HOUR_UTC: int = int(os.getenv("SHADOW_JOB_HOUR_UTC", "2"))
SHADOW_JOB_MINUTE_UTC: int = int(os.getenv("SHADOW_JOB_MINUTE_UTC", "0"))
