#!/usr/bin/env python3
"""Package fine-tune candidate predictions/eval into CI artifact and PR comment markdown."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _format_percent(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value) * 100:.2f}%"
    return "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description="Package fine-tune candidate results")
    parser.add_argument("--predictions-jsonl", required=True)
    parser.add_argument("--evaluation-json", required=True)
    parser.add_argument("--output-dir", default="NLP/evaluation/results")
    parser.add_argument("--prefix", default="ci_finetune_candidate")
    args = parser.parse_args()

    predictions_path = Path(args.predictions_jsonl).expanduser().resolve()
    evaluation_path = Path(args.evaluation_json).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()

    if not predictions_path.exists():
        print(f"Predictions file not found: {predictions_path}")
        return 1
    if not evaluation_path.exists():
        print(f"Evaluation file not found: {evaluation_path}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    eval_artifact = _read_json(evaluation_path)
    result = eval_artifact.get("result") if isinstance(eval_artifact.get("result"), dict) else {}
    thresholds = eval_artifact.get("thresholds") if isinstance(eval_artifact.get("thresholds"), dict) else {}

    artifact_predictions = out_dir / f"{args.prefix}_predictions.jsonl"
    artifact_eval = out_dir / f"{args.prefix}_eval.json"
    artifact_comment = out_dir / f"{args.prefix}_pr_comment.md"

    shutil.copyfile(predictions_path, artifact_predictions)
    shutil.copyfile(evaluation_path, artifact_eval)

    comment = f"""# Fine-tune Candidate Evaluation\n\n"""
    comment += f"Generated at: {datetime.now(timezone.utc).isoformat()}\n\n"
    comment += "## Summary\n"
    comment += f"- Reference rows: {result.get('reference_rows', 'n/a')}\n"
    comment += f"- Matched predictions: {result.get('matched_predictions', 'n/a')}\n"
    comment += f"- Coverage: {_format_percent(result.get('coverage'))}\n"
    comment += f"- Intent accuracy: {_format_percent(result.get('intent_accuracy'))}\n"
    comment += f"- Clarification recall: {_format_percent(result.get('clarification_recall'))}\n"
    comment += f"- Entity recall: {_format_percent(result.get('entity_recall'))}\n\n"
    comment += "## Thresholds\n"
    comment += f"- min intent accuracy: {thresholds.get('min_intent_accuracy', 'n/a')}\n"
    comment += f"- min clarification recall: {thresholds.get('min_clarification_recall', 'n/a')}\n"
    comment += f"- min entity recall: {thresholds.get('min_entity_recall', 'n/a')}\n"
    comment += f"- min coverage: {thresholds.get('min_coverage', 'n/a')}\n\n"
    comment += f"## Gate Result\n\n**{str(eval_artifact.get('quality_gate', 'unknown')).upper()}**\n"

    if isinstance(result.get("missing_predictions"), list) and result.get("missing_predictions"):
        comment += "\n## Missing Predictions (first 10)\n"
        for item in result.get("missing_predictions", [])[:10]:
            if isinstance(item, dict):
                comment += f"- {item.get('mode', 'n/a')} :: {item.get('text', 'n/a')}\n"

    artifact_comment.write_text(comment, encoding="utf-8")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "predictions_jsonl": str(artifact_predictions),
        "evaluation_json": str(artifact_eval),
        "pr_comment_md": str(artifact_comment),
        "quality_gate": eval_artifact.get("quality_gate", "unknown"),
        "result": result,
    }
    (out_dir / f"{args.prefix}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote artifacts to: {out_dir}")
    print(f"PR comment: {artifact_comment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
