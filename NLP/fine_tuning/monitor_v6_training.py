#!/usr/bin/env python3
"""
Monitor and evaluate v6 multi-epoch adapter as it completes.

Usage:
    python NLP/fine_tuning/monitor_v6_training.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path


def run_v6_evaluation():
    """Run full v6 evaluation pipeline once adapter exists."""
    adapter_dir = Path("NLP/fine_tuning/models/intent_qlora_v6_1p5b_multiepoch")
    
    if not adapter_dir.exists():
        print(f"[Monitor] Adapter dir not ready: {adapter_dir}")
        return False
    
    print("\n[Monitor] ✓ Adapter ready, starting evaluation pipeline...")
    
    # Generate predictions on v2
    print("\n[Monitor] Generating predictions on v2 with v6 adapter...")
    cmd_v2_gen = [
        sys.executable, "-X", "utf8",
        "NLP/fine_tuning/generate_candidate_predictions.py",
        "--reference-openai-jsonl", "NLP/fine_tuning/data/nlp_sft_v2_test_openai_messages.jsonl",
        "--output-jsonl", "NLP/fine_tuning/data/candidate_predictions_qlora_v6_1p5b_multiepoch_on_v2.jsonl",
        "--mode", "local_adapter",
        "--adapter-dir", str(adapter_dir),
        "--max-new-tokens", "220",
        "--taxonomy-json", "NLP/taxonomy/nlp_taxonomy.json",
        "--hybrid-with-baseline",
    ]
    
    result = subprocess.run(cmd_v2_gen, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[Monitor] V2 generation failed:\n{result.stderr}")
        return False
    print(result.stdout)
    
    # Evaluate v2
    print("\n[Monitor] Evaluating v2 predictions...")
    cmd_v2_eval = [
        sys.executable, "-X", "utf8",
        "NLP/fine_tuning/evaluate_candidate_outputs.py",
        "--reference-openai-jsonl", "NLP/fine_tuning/data/nlp_sft_v2_test_openai_messages.jsonl",
        "--candidate-predictions-jsonl", "NLP/fine_tuning/data/candidate_predictions_qlora_v6_1p5b_multiepoch_on_v2.jsonl",
        "--output-json", "NLP/evaluation/results/ci_finetune_candidate_eval_qlora_v6_1p5b_multiepoch_on_v2.json",
    ]
    
    result = subprocess.run(cmd_v2_eval, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[Monitor] V2 eval failed:\n{result.stderr}")
        return False
    print(result.stdout)
    
    # Package v2
    print("\n[Monitor] Packaging v2 results...")
    cmd_v2_pkg = [
        sys.executable, "-X", "utf8",
        "NLP/fine_tuning/package_candidate_eval.py",
        "--predictions-jsonl", "NLP/fine_tuning/data/candidate_predictions_qlora_v6_1p5b_multiepoch_on_v2.jsonl",
        "--evaluation-json", "NLP/evaluation/results/ci_finetune_candidate_eval_qlora_v6_1p5b_multiepoch_on_v2.json",
        "--output-dir", "NLP/evaluation/results",
        "--prefix", "ci_finetune_candidate_qlora_v6_1p5b_multiepoch_on_v2",
    ]
    
    result = subprocess.run(cmd_v2_pkg, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[Monitor] V2 pkg failed:\n{result.stderr}")
        return False
    print(result.stdout)
    
    # Similarly for v3
    print("\n[Monitor] Generating predictions on v3 with v6 adapter...")
    cmd_v3_gen = [
        sys.executable, "-X", "utf8",
        "NLP/fine_tuning/generate_candidate_predictions.py",
        "--reference-openai-jsonl", "NLP/fine_tuning/data/nlp_sft_v3_entity_test_openai_messages.jsonl",
        "--output-jsonl", "NLP/fine_tuning/data/candidate_predictions_qlora_v6_1p5b_multiepoch_on_v3.jsonl",
        "--mode", "local_adapter",
        "--adapter-dir", str(adapter_dir),
        "--max-new-tokens", "220",
        "--taxonomy-json", "NLP/taxonomy/nlp_taxonomy.json",
        "--hybrid-with-baseline",
    ]
    
    result = subprocess.run(cmd_v3_gen, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[Monitor] V3 generation failed:\n{result.stderr}")
        return False
    print(result.stdout)
    
    print("\n[Monitor] Evaluating v3 predictions...")
    cmd_v3_eval = [
        sys.executable, "-X", "utf8",
        "NLP/fine_tuning/evaluate_candidate_outputs.py",
        "--reference-openai-jsonl", "NLP/fine_tuning/data/nlp_sft_v3_entity_test_openai_messages.jsonl",
        "--candidate-predictions-jsonl", "NLP/fine_tuning/data/candidate_predictions_qlora_v6_1p5b_multiepoch_on_v3.jsonl",
        "--output-json", "NLP/evaluation/results/ci_finetune_candidate_eval_qlora_v6_1p5b_multiepoch_on_v3.json",
    ]
    
    result = subprocess.run(cmd_v3_eval, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[Monitor] V3 eval failed:\n{result.stderr}")
        return False
    print(result.stdout)
    
    print("\n[Monitor] Packaging v3 results...")
    cmd_v3_pkg = [
        sys.executable, "-X", "utf8",
        "NLP/fine_tuning/package_candidate_eval.py",
        "--predictions-jsonl", "NLP/fine_tuning/data/candidate_predictions_qlora_v6_1p5b_multiepoch_on_v3.jsonl",
        "--evaluation-json", "NLP/evaluation/results/ci_finetune_candidate_eval_qlora_v6_1p5b_multiepoch_on_v3.json",
        "--output-dir", "NLP/evaluation/results",
        "--prefix", "ci_finetune_candidate_qlora_v6_1p5b_multiepoch_on_v3",
    ]
    
    result = subprocess.run(cmd_v3_pkg, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[Monitor] V3 pkg failed:\n{result.stderr}")
        return False
    print(result.stdout)
    
    # Parse and summarize results
    eval_v2_path = Path("NLP/evaluation/results/ci_finetune_candidate_eval_qlora_v6_1p5b_multiepoch_on_v2.json")
    eval_v3_path = Path("NLP/evaluation/results/ci_finetune_candidate_eval_qlora_v6_1p5b_multiepoch_on_v3.json")
    
    if eval_v2_path.exists() and eval_v3_path.exists():
        with open(eval_v2_path) as f:
            eval_v2 = json.load(f)
        with open(eval_v3_path) as f:
            eval_v3 = json.load(f)
        
        print("\n" + "="*60)
        print("V6 MULTI-EPOCH EVALUATION RESULTS")
        print("="*60)
        
        print("\nV2 Test Set:")
        print(f"  Intent:         {eval_v2['result'].get('intent_accuracy', 0):.2%}")
        print(f"  Clarification:  {eval_v2['result'].get('clarification_recall', 0):.2%}")
        print(f"  Entity:         {eval_v2['result'].get('entity_map_recall', 0):.2%}")
        print(f"  Coverage:       {eval_v2['result'].get('coverage', 0):.2%}")
        print(f"  Gate:           {eval_v2.get('quality_gate', 'UNKNOWN')}")
        
        print("\nV3 Test Set:")
        print(f"  Intent:         {eval_v3['result'].get('intent_accuracy', 0):.2%}")
        print(f"  Clarification:  {eval_v3['result'].get('clarification_recall', 0):.2%}")
        print(f"  Entity:         {eval_v3['result'].get('entity_map_recall', 0):.2%}")
        print(f"  Coverage:       {eval_v3['result'].get('coverage', 0):.2%}")
        print(f"  Gate:           {eval_v3.get('quality_gate', 'UNKNOWN')}")
        
        print("\n" + "="*60)
    
    return True


if __name__ == "__main__":
    print("[Monitor] Starting v6 adapter monitor...")
    print("[Monitor] This script will wait for training to complete and then run full evaluation.")
    
    # Poll for adapter directory
    adapter_dir = Path("NLP/fine_tuning/models/intent_qlora_v6_1p5b_multiepoch")
    max_wait = 3600  # 1 hour max wait
    elapsed = 0
    
    while not adapter_dir.exists() and elapsed < max_wait:
        print(f"[Monitor] Waiting for adapter... ({elapsed}s elapsed)")
        time.sleep(30)
        elapsed += 30
    
    if adapter_dir.exists():
        print("[Monitor] Training complete, running evaluation...")
        success = run_v6_evaluation()
        if success:
            print("\n[Monitor] ✓ V6 evaluation complete!")
            sys.exit(0)
        else:
            print("\n[Monitor] ✗ V6 evaluation failed")
            sys.exit(1)
    else:
        print(f"\n[Monitor] ✗ Adapter dir did not appear within {max_wait}s")
        sys.exit(1)
