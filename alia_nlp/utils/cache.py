"""Simple in-process LRU cache for NLP results."""

import hashlib
import json
from functools import lru_cache
from typing import Any, Dict, Optional


def _cache_key(user_text: str, mode: str) -> str:
    payload = json.dumps({"t": user_text, "m": mode}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


# Module-level dict cache (process-local, evicted on restart)
_store: Dict[str, Any] = {}
_MAX_SIZE = 512


def get(user_text: str, mode: str) -> Optional[Dict[str, Any]]:
    return _store.get(_cache_key(user_text, mode))


def put(user_text: str, mode: str, result: Dict[str, Any]) -> None:
    if len(_store) >= _MAX_SIZE:
        # Evict oldest quarter
        keys = list(_store.keys())
        for k in keys[: _MAX_SIZE // 4]:
            del _store[k]
    _store[_cache_key(user_text, mode)] = result


def clear() -> None:
    _store.clear()
