#!/usr/bin/env python3
"""Archive and analyze NLP evaluation drift over time."""

import json
import os
from datetime import datetime
from pathlib import Path

def archive_evaluation(dataset_name: str, eval_json_path: str, output_dir: str = "NLP/evaluation/results/archive"):
    """Archive an evaluation result with timestamp.
    
    Args:
        dataset_name: Name/version of dataset evaluated (e.g., 'v4', 'public_supplement')
        eval_json_path: Path to evaluation result JSON
        output_dir: Directory to store archived results
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        with open(eval_json_path, "r") as f:
            eval_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {eval_json_path} not found")
        return
    
    timestamp = datetime.now().isoformat()
    archive_name = f"eval_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    archive_path = Path(output_dir) / archive_name
    
    # Add metadata
    eval_data["archived_at"] = timestamp
    eval_data["dataset_version"] = dataset_name
    
    with open(archive_path, "w") as f:
        json.dump(eval_data, f, indent=2)
    
    print(f"✓ Archived: {archive_path}")
    print(f"  Intent accuracy: {eval_data.get('intent_accuracy', 'N/A')}")
    print(f"  Safety recall: {eval_data.get('safety_recall', 'N/A')}")
    print(f"  Quality gate: {eval_data.get('quality_gate', 'N/A')}")

def compare_to_baseline(eval_data: dict, baseline_metrics: dict = None):
    """Compare current evaluation to v4 baseline.
    
    Args:
        eval_data: Current evaluation result
        baseline_metrics: v4 baseline metrics (defaults to known values)
    
    Returns:
        dict: Comparison with drift percentages
    """
    if baseline_metrics is None:
        baseline_metrics = {
            "intent_accuracy": 0.9592,
            "safety_recall": 1.0,
            "secondary_tags_recall": 0.7475,
            "entity_map_recall": 0.5625,
        }
    
    comparison = {}
    for metric, baseline_value in baseline_metrics.items():
        current_value = eval_data.get(metric, 0.0)
        drift_pct = ((current_value - baseline_value) / baseline_value * 100) if baseline_value > 0 else 0
        comparison[metric] = {
            "baseline": baseline_value,
            "current": current_value,
            "drift_pct": drift_pct,
            "status": "⚠ DRIFT" if drift_pct < -3 else "✓ OK"
        }
    
    return comparison

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python drift_monitoring.py <dataset_name> <eval_json_path>")
        print("Example: python drift_monitoring.py v4 NLP/evaluation/results/eval_v4_latest.json")
        sys.exit(1)
    
    dataset = sys.argv[1]
    eval_file = sys.argv[2]
    
    archive_evaluation(dataset, eval_file)
    
    # Show comparison
    with open(eval_file) as f:
        data = json.load(f)
    
    comparison = compare_to_baseline(data)
    print("\nDrift Analysis (vs v4 baseline):")
    for metric, values in comparison.items():
        print(f"  {metric}:")
        print(f"    Baseline: {values['baseline']:.2%}")
        print(f"    Current:  {values['current']:.2%}")
        print(f"    Drift:    {values['drift_pct']:+.1f}% {values['status']}")
