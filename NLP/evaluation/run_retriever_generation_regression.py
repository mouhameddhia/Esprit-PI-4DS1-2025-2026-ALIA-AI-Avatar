"""Static regression checks for retriever and generation safety-critical wiring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_CHECKS = [
    {
        "name": "reranker_integration",
        "file": Path("backend/utils/rag_pipeline.py"),
        "needle": "rerank_candidates(",
    },
    {
        "name": "retrieval_non_verbatim_guardrail",
        "file": Path("backend/utils/rag_pipeline.py"),
        "needle": "Do not copy the source text verbatim unless the user explicitly asks for it",
    },
    {
        "name": "physician_patient_advice_guardrail",
        "file": Path("backend/routes/chat.py"),
        "needle": "Do not provide medical advice for individual patients",
    },
    {
        "name": "clarification_prompt_prefix",
        "file": Path("backend/routes/chat.py"),
        "needle": "I want to make sure I answer the right thing",
    },
    {
        "name": "clarification_option_mode_switch",
        "file": Path("backend/routes/chat.py"),
        "needle": "_CLARIFICATION_OPTIONS",
    },
    {
        "name": "nlp_event_language_logging",
        "file": Path("backend/routes/chat.py"),
        "needle": '"language": nlp_analysis.get("language", "unknown")',
    },
]


def run_checks(repo_root: Path) -> dict:
    failures: list[dict] = []
    passes: list[str] = []

    for check in REQUIRED_CHECKS:
        target = repo_root / check["file"]
        if not target.exists():
            failures.append(
                {
                    "check": check["name"],
                    "file": str(check["file"]),
                    "reason": "file_missing",
                }
            )
            continue

        content = target.read_text(encoding="utf-8")
        if check["needle"] not in content:
            failures.append(
                {
                    "check": check["name"],
                    "file": str(check["file"]),
                    "reason": "needle_not_found",
                    "needle": check["needle"],
                }
            )
        else:
            passes.append(check["name"])

    return {
        "passed_checks": passes,
        "failed_checks": failures,
        "total": len(REQUIRED_CHECKS),
        "passed": len(failures) == 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retriever+generation regression checks")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root path",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("NLP/evaluation/results/eval_retriever_generation_regression_latest.json"),
        help="Output JSON path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_checks(args.repo_root)

    payload = {
        "result": result,
        "passed": result["passed"],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps(payload, indent=2))
    if not result["passed"]:
        print("Retriever+generation regression checks failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
