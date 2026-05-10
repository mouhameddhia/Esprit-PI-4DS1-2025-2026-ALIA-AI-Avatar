"""Batch pipeline — no latency constraint, used for evaluation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from alia_nlp.src.pipeline.online import analyze


def run_batch(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process a list of dataset rows and return NLP results."""
    results = []
    for row in rows:
        text = str(row.get("text", ""))
        mode = str(row.get("mode", "physician_portal"))
        result = analyze(text, mode=mode)
        results.append(result.to_dict())
    return results


def run_file(path: Path) -> List[Dict[str, Any]]:
    import json
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return run_batch(rows)
