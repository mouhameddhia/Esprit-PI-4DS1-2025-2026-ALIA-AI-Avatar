"""Groq client factory with lazy initialisation."""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_client = None


def get_groq_client():
    global _client
    if _client is not None:
        return _client
    try:
        from groq import Groq
    except ImportError:
        logger.error("groq package not installed — run: pip install groq")
        return None
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        logger.warning("GROQ_API_KEY not set")
        return None
    _client = Groq(api_key=api_key)
    return _client


def get_model() -> str:
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
