"""CLI runner for the Hybrid Medical Agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE_ROOT = SCRIPT_PATH.parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from hybrid_medical_agent.agent.controller import HybridMedicalController


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run hybrid medical orchestration agent")
    parser.add_argument("--question", default=None, help="Single question to answer")
    parser.add_argument("--training", action="store_true", help="Start one training turn (doctor -> rep)")
    parser.add_argument("--rep-answer", default=None, help="Representative answer used with --training")
    parser.add_argument("--drug", default=None, help="Optional drug for training turn")
    parser.add_argument("--topic", default=None, help="Optional topic for training turn")
    parser.add_argument("--llm-model", default="llama3:8b", help="Local Ollama model")
    parser.add_argument("--llm-base-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--user-id", default="demo_user", help="Authenticated user ID for per-user memory namespace")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    controller = HybridMedicalController(
        workspace_root=WORKSPACE_ROOT,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
    )

    if args.training:
        doctor_question, expected_answer, source = controller.start_training_turn(
            drug_name=args.drug,
            topic=args.topic,
            user_id=args.user_id,
        )
        if not args.rep_answer:
            print(
                json.dumps(
                    {
                        "mode": "training",
                        "doctor_question": doctor_question,
                        "expected_source": source,
                        "note": "Provide --rep-answer to evaluate the representative reply.",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return

        evaluation = controller.evaluate_training_answer(
            doctor_question,
            args.rep_answer,
            user_id=args.user_id,
        )
        print(
            json.dumps(
                {
                    "mode": "training",
                    "doctor_question": evaluation.doctor_question,
                    "rep_answer": evaluation.rep_answer,
                    "expected_answer": evaluation.expected_answer,
                    "feedback": evaluation.feedback,
                    "source": evaluation.source,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if not args.question:
        parser.error("Use --question for QA mode or --training for training mode.")

    result = controller.handle_query(args.question, user_id=args.user_id)
    print(
        json.dumps(
            {
                "question": args.question,
                "answer": result.answer,
                "source": result.source,
                "drug_name": result.drug_name,
                "topic": result.topic,
                "confidence": result.confidence,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
