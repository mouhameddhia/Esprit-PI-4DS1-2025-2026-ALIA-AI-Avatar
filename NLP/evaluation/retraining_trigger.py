#!/usr/bin/env python3
"""Phase 4 retraining trigger analyzer.

Builds a single decision artifact from evaluation outputs and determines
whether a retraining cycle should be started.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


DEFAULT_POLICY: Dict[str, float] = {
    "min_intent_accuracy": 0.90,
    "min_safety_recall": 0.95,
    "min_entity_map_recall": 0.50,
    "min_clear_intent_accuracy": 0.85,
    "min_rerank_mrr": 0.85,
    "min_language_accuracy": 0.90,
    "max_shadow_divergence": 0.15,
}


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _get_metric(payload: Dict[str, Any], key: str) -> float | None:
    if not isinstance(payload, dict):
        return None

    result = payload.get("result")
    if isinstance(result, dict) and isinstance(result.get(key), (int, float)):
        return float(result[key])

    if isinstance(payload.get(key), (int, float)):
        return float(payload[key])

    return None


def _passed(payload: Dict[str, Any]) -> bool | None:
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("quality_gate"), str):
        return payload["quality_gate"].lower() == "pass"
    if isinstance(payload.get("passed"), bool):
        return payload["passed"]
    return None


def _threshold_check(
    name: str,
    value: float | None,
    comparator: str,
    threshold: float,
    reasons: list[str],
    checks: Dict[str, Any],
) -> None:
    if value is None:
        checks[name] = {"value": None, "threshold": threshold, "ok": True, "note": "missing_metric"}
        return

    if comparator == "min":
        ok = value >= threshold
    else:
        ok = value <= threshold

    checks[name] = {"value": round(value, 4), "threshold": threshold, "ok": ok}
    if not ok:
        if comparator == "min":
            reasons.append(f"{name} below threshold ({value:.4f} < {threshold:.4f})")
        else:
            reasons.append(f"{name} above threshold ({value:.4f} > {threshold:.4f})")


def build_decision(
    eval_artifact: Dict[str, Any],
    shadow_artifact: Dict[str, Any],
    clarification_artifact: Dict[str, Any],
    rerank_artifact: Dict[str, Any],
    language_artifact: Dict[str, Any],
    policy: Dict[str, float],
) -> Dict[str, Any]:
    reasons: list[str] = []
    checks: Dict[str, Any] = {}

    # Gate-level checks first.
    for label, artifact in [
        ("eval_gate", eval_artifact),
        ("shadow_gate", shadow_artifact),
        ("clarification_gate", clarification_artifact),
        ("rerank_gate", rerank_artifact),
        ("language_gate", language_artifact),
    ]:
        status = _passed(artifact)
        ok = True if status is None else bool(status)
        checks[label] = {"value": status, "ok": ok}
        if not ok:
            reasons.append(f"{label} failed")

    # Metric-level checks.
    _threshold_check(
        "intent_accuracy",
        _get_metric(eval_artifact, "intent_accuracy"),
        "min",
        policy["min_intent_accuracy"],
        reasons,
        checks,
    )
    _threshold_check(
        "safety_recall",
        _get_metric(eval_artifact, "safety_recall"),
        "min",
        policy["min_safety_recall"],
        reasons,
        checks,
    )
    _threshold_check(
        "entity_map_recall",
        _get_metric(eval_artifact, "entity_map_recall"),
        "min",
        policy["min_entity_map_recall"],
        reasons,
        checks,
    )
    _threshold_check(
        "shadow_divergence",
        _get_metric(shadow_artifact, "divergence_rate"),
        "max",
        policy["max_shadow_divergence"],
        reasons,
        checks,
    )
    _threshold_check(
        "clear_intent_accuracy",
        _get_metric(clarification_artifact, "clear_intent_accuracy"),
        "min",
        policy["min_clear_intent_accuracy"],
        reasons,
        checks,
    )

    rerank_mrr = None
    if isinstance(rerank_artifact.get("result"), dict):
        rerank_mrr = (
            (rerank_artifact["result"].get("reranked") or {}).get("mrr")
            if isinstance(rerank_artifact["result"].get("reranked"), dict)
            else None
        )
    if not isinstance(rerank_mrr, (int, float)):
        rerank_mrr = _get_metric(rerank_artifact, "mrr")

    _threshold_check(
        "rerank_mrr",
        float(rerank_mrr) if isinstance(rerank_mrr, (int, float)) else None,
        "min",
        policy["min_rerank_mrr"],
        reasons,
        checks,
    )

    language_accuracy = _get_metric(language_artifact, "accuracy")
    _threshold_check(
        "language_accuracy",
        language_accuracy,
        "min",
        policy["min_language_accuracy"],
        reasons,
        checks,
    )

    retraining_required = len(reasons) > 0
    severity = "none"
    if retraining_required and len(reasons) <= 2:
        severity = "watch"
    elif retraining_required:
        severity = "trigger"

    if retraining_required:
        recommendation = {
            "action": "start_retraining_cycle",
            "steps": [
                "Refresh hard-negative and ambiguous datasets from latest logs",
                "Run intent/entity/rerank training jobs",
                "Evaluate candidate offline and in shadow mode",
                "Promote only if promotion gate passes",
            ],
        }
    else:
        recommendation = {
            "action": "no_retraining_needed",
            "steps": ["Continue monitoring with current production model"]
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy_version": "phase4_v1",
        "policy": policy,
        "retraining_required": retraining_required,
        "severity": severity,
        "reasons": reasons,
        "checks": checks,
        "recommendation": recommendation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate retraining trigger decision artifact")
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--shadow-json", required=True)
    parser.add_argument("--clarification-json", required=True)
    parser.add_argument("--rerank-json", required=True)
    parser.add_argument("--language-json", required=True)
    parser.add_argument("--policy-json", help="Optional JSON file overriding default thresholds")
    parser.add_argument("--output-json", default="NLP/evaluation/results/ci_retraining_trigger.json")
    parser.add_argument("--fail-on-trigger", action="store_true")
    args = parser.parse_args()

    policy = dict(DEFAULT_POLICY)
    if args.policy_json:
        policy_payload = _load_json(Path(args.policy_json).expanduser().resolve())
        if isinstance(policy_payload, dict):
            for key, value in policy_payload.items():
                if key in policy and isinstance(value, (int, float)):
                    policy[key] = float(value)

    decision = build_decision(
        eval_artifact=_load_json(Path(args.eval_json).expanduser().resolve()),
        shadow_artifact=_load_json(Path(args.shadow_json).expanduser().resolve()),
        clarification_artifact=_load_json(Path(args.clarification_json).expanduser().resolve()),
        rerank_artifact=_load_json(Path(args.rerank_json).expanduser().resolve()),
        language_artifact=_load_json(Path(args.language_json).expanduser().resolve()),
        policy=policy,
    )

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Retraining required: {decision['retraining_required']}")
    print(f"Severity: {decision['severity']}")
    if decision["reasons"]:
        print("Reasons:")
        for reason in decision["reasons"]:
            print(f"- {reason}")
    print(f"Artifact saved to: {output_path}")

    if args.fail_on_trigger and decision["retraining_required"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
