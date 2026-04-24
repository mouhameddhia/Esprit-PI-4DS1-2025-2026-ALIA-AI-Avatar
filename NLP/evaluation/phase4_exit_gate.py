#!/usr/bin/env python3
"""Phase 4 exit gate: aggregate required artifacts and enforce completion criteria."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _is_quality_pass(payload: Dict[str, Any]) -> bool:
    qg = payload.get("quality_gate")
    if isinstance(qg, str):
        return qg.lower() == "pass"

    passed = payload.get("passed")
    if isinstance(passed, bool):
        return passed

    result = payload.get("result")
    if isinstance(result, dict) and isinstance(result.get("passed"), bool):
        return bool(result.get("passed"))

    return False


def _has_ratchet_health(payload: Dict[str, Any]) -> bool:
    if str(payload.get("quality_gate", "")).lower() != "pass":
        return False
    reason = str(payload.get("reason", ""))
    if reason == "quality_gate_failed_reset_streak":
        return False
    return bool(reason)


def _check_exists(path: Path) -> Tuple[bool, str]:
    return (path.exists(), "exists" if path.exists() else "missing")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 4 exit gate")
    parser.add_argument("--results-dir", default="NLP/evaluation/results")
    parser.add_argument("--output-json", default="NLP/evaluation/results/ci_phase4_exit_gate.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()

    required_files = {
        "ci_eval": results_dir / "ci_eval.json",
        "ci_shadow_eval": results_dir / "ci_shadow_eval.json",
        "ci_clarification_eval": results_dir / "ci_clarification_eval.json",
        "ci_retrieval_rerank_eval": results_dir / "ci_retrieval_rerank_eval.json",
        "ci_language_eval": results_dir / "ci_language_eval.json",
        "ci_retriever_generation_regression": results_dir / "ci_retriever_generation_regression.json",
        "ci_retrieval_review_gate": results_dir / "ci_retrieval_review_gate.json",
        "ci_retraining_trigger": results_dir / "ci_retraining_trigger.json",
        "ci_version_snapshot": results_dir / "ci_version_snapshot.json",
        "ci_trained_intent_shadow_eval": results_dir / "ci_trained_intent_shadow_eval.json",
        "ci_trained_entity_shadow_eval": results_dir / "ci_trained_entity_shadow_eval.json",
        "ci_trained_rerank_shadow_eval": results_dir / "ci_trained_rerank_shadow_eval.json",
        "ci_intent_shadow_thresholds_next": results_dir / "ci_intent_shadow_thresholds_next.json",
        "ci_entity_shadow_thresholds_next": results_dir / "ci_entity_shadow_thresholds_next.json",
    }

    checks: List[Dict[str, Any]] = []
    failed: List[str] = []

    loaded: Dict[str, Dict[str, Any]] = {}
    for name, path in required_files.items():
        ok, reason = _check_exists(path)
        checks.append({"name": f"artifact:{name}", "ok": ok, "detail": reason, "path": str(path)})
        if not ok:
            failed.append(f"artifact:{name}")
            continue
        loaded[name] = _read_json(path)

    def add_check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})
        if not ok:
            failed.append(name)

    if "ci_eval" in loaded:
        add_check("gate:ci_eval", _is_quality_pass(loaded["ci_eval"]), "quality_gate")
    if "ci_shadow_eval" in loaded:
        add_check("gate:ci_shadow_eval", _is_quality_pass(loaded["ci_shadow_eval"]), "quality_gate")
    if "ci_clarification_eval" in loaded:
        add_check("gate:ci_clarification_eval", _is_quality_pass(loaded["ci_clarification_eval"]), "quality_gate")
    if "ci_retrieval_rerank_eval" in loaded:
        add_check("gate:ci_retrieval_rerank_eval", _is_quality_pass(loaded["ci_retrieval_rerank_eval"]), "quality_gate")
    if "ci_language_eval" in loaded:
        add_check("gate:ci_language_eval", _is_quality_pass(loaded["ci_language_eval"]), "quality_gate")
    if "ci_retriever_generation_regression" in loaded:
        add_check(
            "gate:ci_retriever_generation_regression",
            _is_quality_pass(loaded["ci_retriever_generation_regression"]),
            "passed/result.passed",
        )
    if "ci_retrieval_review_gate" in loaded:
        add_check("gate:ci_retrieval_review_gate", _is_quality_pass(loaded["ci_retrieval_review_gate"]), "quality_gate")

    if "ci_trained_intent_shadow_eval" in loaded:
        add_check("gate:ci_trained_intent_shadow_eval", _is_quality_pass(loaded["ci_trained_intent_shadow_eval"]), "quality_gate")
    if "ci_trained_entity_shadow_eval" in loaded:
        add_check("gate:ci_trained_entity_shadow_eval", _is_quality_pass(loaded["ci_trained_entity_shadow_eval"]), "quality_gate")
    if "ci_trained_rerank_shadow_eval" in loaded:
        add_check("gate:ci_trained_rerank_shadow_eval", _is_quality_pass(loaded["ci_trained_rerank_shadow_eval"]), "quality_gate")

    if "ci_retraining_trigger" in loaded:
        retraining_required = loaded["ci_retraining_trigger"].get("retraining_required")
        add_check(
            "policy:retraining_not_required",
            retraining_required is False,
            f"retraining_required={retraining_required}",
        )

    if "ci_version_snapshot" in loaded:
        assets = loaded["ci_version_snapshot"].get("assets")
        all_exist = True
        asset_count = 0
        if isinstance(assets, dict):
            for _, meta in assets.items():
                asset_count += 1
                if not (isinstance(meta, dict) and meta.get("exists") is True):
                    all_exist = False
        else:
            all_exist = False
        add_check("policy:version_assets_exist", all_exist and asset_count > 0, f"asset_count={asset_count}")

    if "ci_intent_shadow_thresholds_next" in loaded:
        add_check(
            "ratchet:intent_health",
            _has_ratchet_health(loaded["ci_intent_shadow_thresholds_next"]),
            str(loaded["ci_intent_shadow_thresholds_next"].get("reason", "")),
        )
    if "ci_entity_shadow_thresholds_next" in loaded:
        add_check(
            "ratchet:entity_health",
            _has_ratchet_health(loaded["ci_entity_shadow_thresholds_next"]),
            str(loaded["ci_entity_shadow_thresholds_next"].get("reason", "")),
        )

    passed_checks = [item["name"] for item in checks if item.get("ok") is True]
    failed_checks = [item["name"] for item in checks if item.get("ok") is not True]

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy_version": "phase4_exit_v1",
        "summary": {
            "total_checks": len(checks),
            "passed_checks": len(passed_checks),
            "failed_checks": len(failed_checks),
            "pass_ratio": round((len(passed_checks) / len(checks)) if checks else 0.0, 4),
        },
        "quality_gate": "pass" if not failed else "fail",
        "failed_checks": failed_checks,
        "checks": checks,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Phase 4 checks: {len(passed_checks)}/{len(checks)} passed")
    print(f"Artifact saved to: {output_path}")
    if failed_checks:
        print("QUALITY GATE: FAIL")
        for item in failed_checks:
            print(f"- {item}")
        return 2

    print("QUALITY GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
