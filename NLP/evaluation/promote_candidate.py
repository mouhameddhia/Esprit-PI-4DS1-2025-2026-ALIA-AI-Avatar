#!/usr/bin/env python3
"""Promotion gate for candidate NLP rollout decisions.

Compares candidate evaluation metrics against a baseline and checks shadow
monitoring divergence before promotion.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_metric(artifact: Dict[str, Any], metric_name: str) -> float:
    result = artifact.get("result")
    if isinstance(result, dict):
        value = result.get(metric_name)
        if isinstance(value, (int, float)):
            return float(value)

    # Backward compatibility for flattened artifacts.
    value = artifact.get(metric_name)
    if isinstance(value, (int, float)):
        return float(value)

    return 0.0


def _evaluate_gate(
    baseline_eval: Dict[str, Any],
    candidate_eval: Dict[str, Any],
    shadow_eval: Dict[str, Any],
    min_intent_delta: float,
    min_safety_delta: float,
    min_entity_delta: float,
    max_shadow_divergence: float,
) -> Tuple[bool, Dict[str, Any]]:
    baseline_intent = _extract_metric(baseline_eval, "intent_accuracy")
    baseline_safety = _extract_metric(baseline_eval, "safety_recall")
    baseline_entity = _extract_metric(baseline_eval, "entity_map_recall")

    candidate_intent = _extract_metric(candidate_eval, "intent_accuracy")
    candidate_safety = _extract_metric(candidate_eval, "safety_recall")
    candidate_entity = _extract_metric(candidate_eval, "entity_map_recall")

    intent_delta = round(candidate_intent - baseline_intent, 4)
    safety_delta = round(candidate_safety - baseline_safety, 4)
    entity_delta = round(candidate_entity - baseline_entity, 4)

    shadow_result = shadow_eval.get("result", {}) if isinstance(shadow_eval.get("result"), dict) else {}
    shadow_divergence = float(shadow_result.get("divergence_rate", shadow_eval.get("divergence_rate", 1.0)))
    shadow_gate = str(shadow_eval.get("quality_gate", "fail")).lower()

    checks = {
        "candidate_eval_gate": str(candidate_eval.get("quality_gate", "fail")).lower() == "pass",
        "shadow_eval_gate": shadow_gate == "pass",
        "intent_delta": intent_delta >= min_intent_delta,
        "safety_delta": safety_delta >= min_safety_delta,
        "entity_delta": entity_delta >= min_entity_delta,
        "shadow_divergence": shadow_divergence <= max_shadow_divergence,
    }

    reasons = []
    if not checks["candidate_eval_gate"]:
        reasons.append("candidate_eval quality_gate is not pass")
    if not checks["shadow_eval_gate"]:
        reasons.append("shadow_eval quality_gate is not pass")
    if not checks["intent_delta"]:
        reasons.append(f"intent_delta {intent_delta:+.4f} < required {min_intent_delta:+.4f}")
    if not checks["safety_delta"]:
        reasons.append(f"safety_delta {safety_delta:+.4f} < required {min_safety_delta:+.4f}")
    if not checks["entity_delta"]:
        reasons.append(f"entity_delta {entity_delta:+.4f} < required {min_entity_delta:+.4f}")
    if not checks["shadow_divergence"]:
        reasons.append(
            f"shadow_divergence {shadow_divergence:.4f} > max {max_shadow_divergence:.4f}"
        )

    passed = all(checks.values())
    summary = {
        "checks": checks,
        "metrics": {
            "baseline": {
                "intent_accuracy": baseline_intent,
                "safety_recall": baseline_safety,
                "entity_map_recall": baseline_entity,
            },
            "candidate": {
                "intent_accuracy": candidate_intent,
                "safety_recall": candidate_safety,
                "entity_map_recall": candidate_entity,
            },
            "delta": {
                "intent_accuracy": intent_delta,
                "safety_recall": safety_delta,
                "entity_map_recall": entity_delta,
            },
            "shadow": {
                "divergence_rate": round(shadow_divergence, 4),
                "quality_gate": shadow_gate,
            },
        },
        "reasons": reasons,
    }
    return passed, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Promotion gate for candidate NLP rollout")
    parser.add_argument("--baseline-eval", required=True, help="Path to baseline eval JSON artifact")
    parser.add_argument("--candidate-eval", required=True, help="Path to candidate eval JSON artifact")
    parser.add_argument("--shadow-eval", required=True, help="Path to shadow eval JSON artifact")
    parser.add_argument("--min-intent-delta", type=float, default=0.0)
    parser.add_argument("--min-safety-delta", type=float, default=0.0)
    parser.add_argument("--min-entity-delta", type=float, default=0.0)
    parser.add_argument("--max-shadow-divergence", type=float, default=0.15)
    parser.add_argument(
        "--output-json",
        default=str(Path(__file__).resolve().parent / "results" / "promotion_decision_latest.json"),
        help="Output path for promotion decision artifact",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline_eval).expanduser().resolve()
    candidate_path = Path(args.candidate_eval).expanduser().resolve()
    shadow_path = Path(args.shadow_eval).expanduser().resolve()

    for path in (baseline_path, candidate_path, shadow_path):
        if not path.exists():
            print(f"Missing input: {path}")
            return 1

    baseline_eval = _load_json(baseline_path)
    candidate_eval = _load_json(candidate_path)
    shadow_eval = _load_json(shadow_path)

    passed, summary = _evaluate_gate(
        baseline_eval=baseline_eval,
        candidate_eval=candidate_eval,
        shadow_eval=shadow_eval,
        min_intent_delta=args.min_intent_delta,
        min_safety_delta=args.min_safety_delta,
        min_entity_delta=args.min_entity_delta,
        max_shadow_divergence=args.max_shadow_divergence,
    )

    decision = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "baseline_eval": str(baseline_path),
            "candidate_eval": str(candidate_path),
            "shadow_eval": str(shadow_path),
        },
        "thresholds": {
            "min_intent_delta": args.min_intent_delta,
            "min_safety_delta": args.min_safety_delta,
            "min_entity_delta": args.min_entity_delta,
            "max_shadow_divergence": args.max_shadow_divergence,
        },
        "promotion_gate": "pass" if passed else "fail",
        "summary": summary,
    }

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Promotion gate: {decision['promotion_gate'].upper()}")
    print(f"Intent delta: {summary['metrics']['delta']['intent_accuracy']:+.4f}")
    print(f"Safety delta: {summary['metrics']['delta']['safety_recall']:+.4f}")
    print(f"Entity delta: {summary['metrics']['delta']['entity_map_recall']:+.4f}")
    print(f"Shadow divergence: {summary['metrics']['shadow']['divergence_rate']:.4f}")
    print(f"Artifact saved to: {output_path}")

    if not passed:
        print("Promotion blocked:")
        for reason in summary.get("reasons", []):
            print(f"- {reason}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
