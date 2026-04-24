#!/usr/bin/env python3
"""Build CI shadow-monitoring fixture from a labeled JSONL dataset.

Primary intent is taken from dataset labels. Shadow intent is predicted by the
current NLP pipeline. This provides a deterministic input for shadow_monitoring
in CI where production logs are unavailable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from NLP.pipeline.nlp import analyze_message_nlp


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_no} is not a JSON object")
            yield row


def _build_row(source: Dict[str, Any]) -> Dict[str, Any] | None:
    text = source.get("text")
    expected_intent = source.get("expected_intent")
    mode = source.get("mode", "physician_portal")

    if not isinstance(text, str) or not text.strip():
        return None
    if not isinstance(expected_intent, str) or not expected_intent.strip():
        return None
    if not isinstance(mode, str) or not mode.strip():
        mode = "physician_portal"

    prediction = analyze_message_nlp(user_text=text.strip(), history=[], mode=mode)

    return {
        "text": text.strip(),
        "mode": mode,
        "primary_intent": expected_intent.strip(),
        "shadow_intent": str(prediction.get("intent", "other")),
        "shadow_confidence": prediction.get("confidence"),
        "ground_truth_intent": expected_intent.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build CI shadow fixture from labeled dataset")
    parser.add_argument("dataset", help="Path to labeled JSONL dataset")
    parser.add_argument(
        "--output-jsonl",
        default=str(Path(__file__).resolve().parent / "results" / "ci_shadow_fixture.jsonl"),
        help="Output JSONL path",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return 1

    output_path = Path(args.output_jsonl).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as out:
        for row in _read_jsonl(dataset_path):
            built = _build_row(row)
            if built is None:
                skipped += 1
                continue
            out.write(json.dumps(built, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Rows written: {kept}")
    print(f"Rows skipped: {skipped}")
    print(f"Output written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
