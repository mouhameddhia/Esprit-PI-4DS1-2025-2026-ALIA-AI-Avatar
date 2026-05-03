"""Evaluation drift monitoring — compare current run against the v6 baseline."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Baseline established on v6 dataset
_BASELINE: Dict[str, float] = {
    "intent_accuracy":       0.9592,
    "safety_recall":         1.0,
    "secondary_tags_recall": 0.7475,
    "entity_map_recall":     0.5625,
}

_DRIFT_THRESHOLD = -3.0  # percent


def compare(eval_result: Dict, baseline: Optional[Dict[str, float]] = None) -> Dict:
    baseline = baseline or _BASELINE
    comparison: Dict = {}
    for metric, base_val in baseline.items():
        current = eval_result.get(metric, 0.0)
        drift_pct = ((current - base_val) / base_val * 100) if base_val else 0.0
        comparison[metric] = {
            "baseline": base_val,
            "current": current,
            "drift_pct": round(drift_pct, 2),
            "status": "DRIFT" if drift_pct < _DRIFT_THRESHOLD else "OK",
        }
    return comparison


def archive(dataset_name: str, eval_json_path: str,
            output_dir: str = "alia_nlp/evaluation/results/archive") -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(eval_json_path) as f:
        data = json.load(f)
    data["archived_at"] = datetime.now().isoformat()
    data["dataset_version"] = dataset_name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir) / f"eval_{dataset_name}_{stamp}.json"
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Archived: {out}")
    for k, v in compare(data.get("result", {})).items():
        symbol = "!" if v["status"] == "DRIFT" else "✓"
        print(f"  {symbol} {k}: {v['current']:.2%} (drift {v['drift_pct']:+.1f}%)")
