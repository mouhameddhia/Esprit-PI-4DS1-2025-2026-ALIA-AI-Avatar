"""User-scoped memory namespace helpers.

This module creates isolated per-user memory directories under:
memory/users/<sanitized_user_id>/
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UserMemoryPaths:
    """Resolved memory paths for one authenticated user."""

    user_id: str
    base_dir: Path
    conversation_path: Path
    persistent_path: Path


class UserMemoryNamespace:
    """Resolve and create per-user memory storage directories safely."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self.users_root = workspace_root / "memory" / "users"

    @staticmethod
    def sanitize_user_id(user_id: str | None) -> str:
        """Return a filesystem-safe user identifier.

        Falls back to "anonymous" when missing and hashes unsafe values.
        """

        candidate = str(user_id or "anonymous").strip().lower()
        if not candidate:
            return "anonymous"

        safe = re.sub(r"[^a-z0-9_-]", "_", candidate)
        safe = re.sub(r"_+", "_", safe).strip("_")
        if safe in {"", ".", ".."}:
            return "anonymous"

        # Keep IDs compact and deterministic.
        if len(safe) > 64:
            digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:16]
            safe = f"user_{digest}"
        return safe

    def resolve(self, user_id: str | None) -> UserMemoryPaths:
        """Return isolated memory paths and ensure the folder exists."""

        safe_user_id = self.sanitize_user_id(user_id)
        base_dir = self.users_root / safe_user_id
        base_dir.mkdir(parents=True, exist_ok=True)
        return UserMemoryPaths(
            user_id=safe_user_id,
            base_dir=base_dir,
            conversation_path=base_dir / "conversation_memory.json",
            persistent_path=base_dir / "persistent_memory.json",
        )
