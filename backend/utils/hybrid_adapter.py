"""
Hybrid adapter inference bridge.

The hybrid adapter is an optional fine-tuned NLP model that augments the
keyword/rule-based fallback analyzer. It is disabled by default
(ALIA_USE_HYBRID_ADAPTER=0 in .env). When disabled, chat.py falls back to
utils/nlp.py automatically — these stubs ensure the import never fails.

To enable: set ALIA_USE_HYBRID_ADAPTER=1 and replace the stubs below with
real model loading / inference code.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def get_hybrid_adapter_runtime() -> Optional[Any]:
    """Return the loaded adapter runtime, or None if unavailable/disabled."""
    # Not yet implemented — returning None causes chat.py to use the fallback.
    return None


def infer_hybrid_adapter(
    messages: list[dict[str, str]],
    runtime: Any,
    max_new_tokens: int = 220,
    hybrid_with_baseline: bool = True,
) -> Optional[dict[str, Any]]:
    """Run inference through the hybrid adapter. Returns None if unavailable."""
    if runtime is None:
        return None
    # Placeholder — real inference would call runtime.generate(messages, ...) here.
    return None
