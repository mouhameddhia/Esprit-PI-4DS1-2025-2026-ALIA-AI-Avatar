#!/usr/bin/env python3
"""Compute next-step intent shadow thresholds from latest holdout results."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, min_v: float, max_v: float) -> float:
    return max(min_v, min(max_v, value))


def _metric_or_default(obj: Dict[str, Any], key: str, default: float) -> float:
    val = obj.get(key)
    return float(val) if isinstance(val, (int, float)) else float(default)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ratchet intent shadow thresholds after passing runs")
    parser.add_argument(
        "--config-json",
        default="NLP/evaluation/intent_shadow_thresholds.json",
        help="Threshold config with policy/state",
    )
    parser.add_argument(
        "--eval-json",
        default="NLP/evaluation/results/ci_trained_intent_shadow_eval.json",
        help="Latest trained-intent shadow eval artifact",
    )
    parser.add_argument(
        "--output-json",
        default="NLP/evaluation/results/ci_intent_shadow_thresholds_next.json",
        help="Proposed next thresholds artifact",
    )
    parser.add_argument(
        "--write-back-config",
        action="store_true",
        help="Write updated thresholds/state back to --config-json",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config_json).expanduser().resolve()
    eval_path = Path(args.eval_json).expanduser().resolve()
    out_path = Path(args.output_json).expanduser().resolve()

    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        return 1
    if not eval_path.exists():
        print(f"Eval artifact not found: {eval_path}")
        return 1

    cfg = _read_json(cfg_path)
    ev = _read_json(eval_path)

    thresholds = cfg.get("thresholds") if isinstance(cfg.get("thresholds"), dict) else {}
    policy = cfg.get("ratchet_policy") if isinstance(cfg.get("ratchet_policy"), dict) else {}
    state = cfg.get("state") if isinstance(cfg.get("state"), dict) else {}

    before = {
        "min_shadow_accuracy": _metric_or_default(thresholds, "min_shadow_accuracy", 0.60),
        "max_divergence": _metric_or_default(thresholds, "max_divergence", 0.45),
        "max_train_test_gap": _metric_or_default(thresholds, "max_train_test_gap", 0.40),
    }
    after = deepcopy(before)

    required_passes = int(policy.get("required_consecutive_passes", 2) or 2)
    min_buffer = policy.get("min_buffer") if isinstance(policy.get("min_buffer"), dict) else {}
    step = policy.get("step") if isinstance(policy.get("step"), dict) else {}
    bounds = policy.get("bounds") if isinstance(policy.get("bounds"), dict) else {}

    step_shadow = _metric_or_default(step, "shadow_accuracy", 0.02)
    step_div = _metric_or_default(step, "divergence", 0.02)
    step_gap = _metric_or_default(step, "train_test_gap", 0.02)

    buffer_shadow = _metric_or_default(min_buffer, "shadow_accuracy", 0.02)
    buffer_div = _metric_or_default(min_buffer, "divergence", 0.02)
    buffer_gap = _metric_or_default(min_buffer, "train_test_gap", 0.02)

    max_shadow = _metric_or_default(bounds, "max_shadow_accuracy", 0.90)
    min_div = _metric_or_default(bounds, "min_divergence", 0.10)
    min_gap = _metric_or_default(bounds, "min_train_test_gap", 0.10)

    result = ev.get("result") if isinstance(ev.get("result"), dict) else {}
    generalization = ev.get("generalization") if isinstance(ev.get("generalization"), dict) else {}

    obs_shadow = _metric_or_default(result, "shadow_intent_accuracy", 0.0)
    obs_div = _metric_or_default(result, "divergence_rate", 1.0)
    obs_gap = _metric_or_default(generalization, "train_test_gap", 1.0)

    quality_gate = str(ev.get("quality_gate", "fail")).lower()
    prev_passes = int(state.get("consecutive_passes", 0) or 0)
    consecutive_passes = (prev_passes + 1) if quality_gate == "pass" else 0

    tightened = {}
    reason = "no_change"

    if quality_gate == "pass" and consecutive_passes >= required_passes:
        # Increase minimum shadow accuracy only when current metric has headroom.
        candidate_shadow = _clamp(before["min_shadow_accuracy"] + step_shadow, 0.0, max_shadow)
        if obs_shadow >= (candidate_shadow + buffer_shadow) and candidate_shadow > before["min_shadow_accuracy"]:
            after["min_shadow_accuracy"] = _round4(candidate_shadow)

        # Decrease max divergence only when current metric is comfortably lower.
        candidate_div = _clamp(before["max_divergence"] - step_div, min_div, 1.0)
        if obs_div <= (candidate_div - buffer_div) and candidate_div < before["max_divergence"]:
            after["max_divergence"] = _round4(candidate_div)

        # Decrease max train-test gap only when current metric is comfortably lower.
        candidate_gap = _clamp(before["max_train_test_gap"] - step_gap, min_gap, 1.0)
        if obs_gap <= (candidate_gap - buffer_gap) and candidate_gap < before["max_train_test_gap"]:
            after["max_train_test_gap"] = _round4(candidate_gap)

        tightened = {
            k: {"from": before[k], "to": after[k]}
            for k in before
            if after[k] != before[k]
        }

        if tightened:
            reason = "tightened_after_consistent_passes"
            # Reset streak after ratchet to require another stable period before next tighten.
            consecutive_passes = 0
        else:
            reason = "hold_thresholds_insufficient_headroom"
    elif quality_gate != "pass":
        reason = "quality_gate_failed_reset_streak"

    now = datetime.now(timezone.utc).isoformat()

    next_cfg = deepcopy(cfg)
    next_cfg["thresholds"] = after
    next_cfg["state"] = {
        "consecutive_passes": consecutive_passes,
        "last_updated_at": now,
        "last_quality_gate": quality_gate,
    }

    report = {
        "generated_at": now,
        "config_path": str(cfg_path),
        "eval_path": str(eval_path),
        "quality_gate": quality_gate,
        "reason": reason,
        "observed": {
            "shadow_accuracy": _round4(obs_shadow),
            "divergence": _round4(obs_div),
            "train_test_gap": _round4(obs_gap),
        },
        "before_thresholds": before,
        "after_thresholds": after,
        "tightened": tightened,
        "state": {
            "required_consecutive_passes": required_passes,
            "consecutive_passes": consecutive_passes,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.write_back_config:
        cfg_path.write_text(json.dumps(next_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Quality gate: {quality_gate}")
    print(f"Reason: {reason}")
    print(f"Consecutive passes: {consecutive_passes}/{required_passes}")
    if tightened:
        print("Thresholds tightened:")
        for name, change in tightened.items():
            print(f"- {name}: {change['from']:.4f} -> {change['to']:.4f}")
    else:
        print("Thresholds unchanged")
    print(f"Artifact saved to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
