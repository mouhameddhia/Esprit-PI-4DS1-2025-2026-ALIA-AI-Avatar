"""Evaluate runtime language detection quality on a labeled benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from NLP.pipeline.language import detect_language


SUPPORTED_LANGS = {"en", "fr", "ar", "unknown"}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def evaluate(rows: list[dict]) -> dict:
    total = 0
    correct = 0
    per_lang_totals = {lang: 0 for lang in SUPPORTED_LANGS}
    per_lang_correct = {lang: 0 for lang in SUPPORTED_LANGS}
    failures: list[dict] = []

    for idx, row in enumerate(rows, start=1):
        text = str(row.get("text", ""))
        expected = str(row.get("expected_language", "unknown")).strip().lower()
        if expected not in SUPPORTED_LANGS:
            expected = "unknown"

        predicted = detect_language(text)
        total += 1
        per_lang_totals[expected] += 1

        is_correct = predicted == expected
        if is_correct:
            correct += 1
            per_lang_correct[expected] += 1
        else:
            failures.append(
                {
                    "row": idx,
                    "text": text,
                    "expected": expected,
                    "predicted": predicted,
                }
            )

    accuracy = (correct / total) if total else 0.0
    per_lang_accuracy = {
        lang: (per_lang_correct[lang] / per_lang_totals[lang]) if per_lang_totals[lang] else None
        for lang in sorted(SUPPORTED_LANGS)
    }

    return {
        "size": total,
        "accuracy": round(accuracy, 4),
        "per_language_accuracy": per_lang_accuracy,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run language detection quality gate")
    parser.add_argument("dataset", type=Path, help="Path to language detection JSONL dataset")
    parser.add_argument("--min-accuracy", type=float, default=0.9, help="Minimum required overall accuracy")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("NLP/evaluation/results/eval_language_detection_latest.json"),
        help="Output JSON path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.dataset)
    result = evaluate(rows)

    gate_passed = result["accuracy"] >= args.min_accuracy

    payload = {
        "dataset": str(args.dataset),
        "result": result,
        "thresholds": {"min_accuracy": args.min_accuracy},
        "passed": gate_passed,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps(payload, indent=2))
    if not gate_passed:
        print(
            f"Language detection gate failed: accuracy={result['accuracy']:.4f} < min_accuracy={args.min_accuracy:.4f}"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
