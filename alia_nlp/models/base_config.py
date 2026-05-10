"""Base LLM configuration."""

import os

MODEL_ID: str  = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TEMPERATURE: float = 0.1
MAX_TOKENS:  int   = 700
RESPONSE_FORMAT    = {"type": "json_object"}
