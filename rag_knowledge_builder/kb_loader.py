"""Knowledge-base loader utilities.

Centralized helper to resolve and discover JSON knowledge files.

Note: The knowledge base source was moved from the previous
`rag_knowledge_base/json` location into `rag_knowledge_builder/scripts`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List


def default_json_dirs() -> List[str]:
    """Return default relative folders that contain KB JSON files."""
    return ["rag_knowledge_builder/scripts"]


def resolve_json_paths(workspace_root: Path, additional: List[str] | None = None) -> List[Path]:
    """Resolve a list of relative folder strings to absolute Path objects."""
    folders = list(default_json_dirs())
    if additional:
        folders.extend(additional)
    return [workspace_root / f for f in folders]


def discover_json_files(workspace_root: Path, folders: List[str] | None = None) -> List[Path]:
    """Discover JSON files under the configured KB folders.

    Returns a sorted, de-duplicated list of Path objects.
    """
    bases = resolve_json_paths(workspace_root, additional=folders)
    files: List[Path] = []
    for base in bases:
        if not base.exists():
            continue
        if base.is_file() and base.suffix.lower() == ".json":
            files.append(base)
            continue
        for path in base.rglob("*.json"):
            files.append(path)
    return sorted(set(files))
