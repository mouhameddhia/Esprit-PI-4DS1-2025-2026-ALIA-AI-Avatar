"""LLM-based intent extraction using structured output (JSON mode).

Key improvement over the old pipeline: Groq's json_object response_format
guarantees valid JSON — no fragile text-to-JSON parser needed.
"""

import json
import logging
import os
from typing import Any, Dict

from alia_nlp.utils.groq_client import get_groq_client, get_model

logger = logging.getLogger(__name__)


def call_structured(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """
    Call Groq with response_format=json_object.
    Returns a guaranteed-valid dict (empty dict on any failure).
    """
    client = get_groq_client()
    if client is None:
        return {}

    try:
        completion = client.chat.completions.create(
            model=get_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            response_format={"type": "json_object"},  # no parser needed
            temperature=0.1,
            max_tokens=900,
        )
        raw = completion.choices[0].message.content or "{}"
        return json.loads(raw)
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return {}
