"""Shared Groq client factory for backend services."""

import logging

from fastapi import HTTPException, status

from .. import config

logger = logging.getLogger(__name__)


def get_groq_client():
    """
    Return an authenticated Groq client.

    Raises HTTP 503 if the Groq SDK is not installed or GROQ_API_KEY is missing.
    This is the backend variant — use it in routes and utils that run inside a
    request context.  The NLP pipeline keeps its own silent-fallback variant
    because it must operate without FastAPI as a dependency.
    """
    try:
        from groq import Groq
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Groq SDK not installed. Run: pip install groq",
        ) from exc

    if not config.GROQ_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GROQ_API_KEY not configured",
        )

    return Groq(api_key=config.GROQ_API_KEY)
