"""
XAI Agent - Real-World Test Case and Usage Example
"""

import json
from xai_agent.pipeline import analyze_system


# ============================================================================
# REAL JSON INPUT - Example pharmaceutical Q&A case
# ============================================================================

TEST_INPUT = {
    "query": "What is the composition of BACTOL products?",
    "retrieved_docs": [
        {
            "source": "BACTOL Product Range",
            "content": "BACTOL Professional Soap (90g): Benzalkonium chloride 0.18g. BACTOL Disinfectant Spray (125ml): Alcohol 95° at 65%. BACTOL Disinfectant Gel (125ml): Alcohol 70%, Benzalkonium chloride 0.2%.",
            "similarity_score": 0.95,
            "source_type": "product_monograph"
        },
        {
            "source": "BACTOL Product Details",
            "content": "BACTOL is an antiseptic / disinfectant for external use only. It is available as a professional soap, a disinfectant spray, and a disinfectant gel. The spray and gel are no-rinse formulas.",
            "similarity_score": 0.89,
            "source_type": "product_documentation"
        }
    ],
    "answer": "BACTOL is available in three product forms. BACTOL Professional Soap (90g) contains benzalkonium chloride 0.18g. BACTOL Disinfectant Spray (125ml) contains alcohol 95° at 65%. BACTOL Disinfectant Gel (125ml) contains alcohol 70% and benzalkonium chloride 0.2%.",
    "judge_output": {
        "faithfulness": 0.92,
        "answer_relevance": 0.89,
        "context_utilization": 0.85,
        "medical_safety": 0.88,
        "clarity": 0.86,
        "mode_alignment": 0.84,
        "overall_score": 0.87,
        "verdict": "good"
    },
    "mode": "commercial",
    "training_scores": {
        "correctness": 0.90,
        "completeness": 0.85,
        "safety": 0.88,
        "clarity": 0.82
    }
}


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main():
    """Run analysis on test case."""
    
    print("=" * 80)
    print("XAI AGENT - REAL JSON I/O DEMONSTRATION")
    print("=" * 80)
    
    print("\n📥 INPUT JSON:")
    print("-" * 80)
    print(json.dumps(TEST_INPUT, indent=2))
    
    # Convert to JSON string for pipeline
    input_json = json.dumps(TEST_INPUT)
    
    print("\n⏳ ANALYZING...")
    print("-" * 80)
    
    # Run analysis
    result_json = analyze_system(input_json)
    result = json.loads(result_json)
    
    # Display results
    print("\n📤 OUTPUT - FINAL VERDICT:")
    print("-" * 80)
    print(json.dumps(result["final_verdict"], indent=2))
    
    print("\n📊 SUMMARY REPORT:")
    print("-" * 80)
    print(json.dumps(result["summary_report"], indent=2))
    
    print("\n🔍 DETAILED ANALYSES:")
    print("-" * 80)
    
    print("\n1️⃣  RETRIEVAL ANALYSIS:")
    retrieval = result["retrieval_analysis"]
    print(f"   Quality: {retrieval.get('quality')}")
    print(f"   Risk Level: {retrieval.get('risk_level')}")
    print(f"   Avg Similarity: {retrieval.get('metrics', {}).get('avg_similarity')}")
    print(f"   Documents Analyzed: {retrieval.get('count')}")
    if retrieval.get("issues"):
        print(f"   Issues: {', '.join(retrieval.get('issues', [])[:2])}")
    
    print("\n2️⃣  GENERATION AUDIT:")
    generation = result["generation_audit"]
    print(f"   Total Claims: {generation.get('total_claims')}")
    print(f"   Verified Claims: {generation.get('verified_claims')}")
    print(f"   Unsupported Claims: {generation.get('unsupported_claims')}")
    print(f"   Consistency Score: {generation.get('consistency_score'):.3f}")
    print(f"   Consistency Level: {generation.get('consistency_level')}")
    print(f"   Hallucinations Detected: {generation.get('hallucinations_detected')}")
    
    print("\n3️⃣  JUDGE ANALYSIS:")
    judge = result["judge_analysis"]
    print(f"   Overall Score: {judge.get('overall_score'):.2f}")
    print(f"   Verdict: {judge.get('verdict')}")
    print(f"   Alignment Score: {judge.get('alignment_score'):.3f}")
    print(f"   Confidence: {judge.get('confidence')}")
    print(f"   Consistency: {judge.get('consistency', {}).get('status')}")
    
    print("\n4️⃣  SAFETY AUDIT:")
    safety = result["safety_audit"]
    print(f"   Risk Level: {safety.get('risk_level')}")
    print(f"   Risk Score: {safety.get('risk_score'):.3f}")
    print(f"   Critical Concerns: {safety.get('critical_concerns')}")
    print(f"   High Risk Concerns: {safety.get('high_risk_concerns')}")
    print(f"   Missing Warnings: {len(safety.get('missing_warnings', []))}")
    print(f"   Clinical Review Required: {safety.get('clinical_review_required')}")
    if safety.get("recommendations"):
        print(f"   Recommendation: {safety.get('recommendations')[0]}")
    
    print("\n5️⃣  TRAINING FEEDBACK:")
    training = result.get("training_feedback")
    if training:
        print(f"   Total Score: {training.get('total_score'):.2f}")
        print(f"   Training Status: {training.get('training_status')}")
        print(f"   Progression Level: {training.get('progression', {}).get('level')}")
        print(f"   Strengths: {', '.join(training.get('strengths', [])[:2])}")
        if training.get("weaknesses"):
            print(f"   Weaknesses: {training.get('weaknesses')[0].get('metric')}")
        if training.get("improvement_suggestions"):
            print(f"   Top Suggestion: {training.get('improvement_suggestions')[0][:60]}...")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    # Return full output
    return result


# ============================================================================
# ALTERNATIVE: Test with hallucinations (HIGH RISK)
# ============================================================================

def test_with_hallucinations():
    """Test case with significant hallucinations."""
    
    risky_input = {
        "query": "Is Aspirin safe for children with fever?",
        "retrieved_docs": [
            {
                "source": "FDA Warning",
                "content": "Aspirin is CONTRAINDICATED in children due to risk of Reye syndrome. Use acetaminophen instead.",
                "similarity_score": 0.98,
                "source_type": "official_guideline"
            }
        ],
        "answer": "Yes, Aspirin is perfectly safe and recommended for children with fever. Use 1000mg tablets every 2 hours. It has no side effects and cures fever immediately.",
        "judge_output": {
            "faithfulness": 0.15,
            "answer_relevance": 0.10,
            "context_utilization": 0.05,
            "medical_safety": 0.05,
            "clarity": 0.70,
            "mode_alignment": 0.40,
            "overall_score": 0.24,
            "verdict": "poor"
        },
        "mode": "commercial"
    }
    
    print("\n" + "=" * 80)
    print("HIGH-RISK TEST CASE - HALLUCINATIONS & SAFETY VIOLATIONS")
    print("=" * 80)
    
    result_json = analyze_system(json.dumps(risky_input))
    result = json.loads(result_json)
    
    print("\n🚨 FINAL VERDICT:")
    final_verdict = result.get("final_verdict", {})
    print(final_verdict.get("recommendation", "No recommendation available"))
    
    print("\n⚠️  CRITICAL ISSUES:")
    indicators = final_verdict.get("risk_indicators", [])
    if indicators:
        for indicator in indicators:
            print(f"   • {indicator.get('indicator', 'UNKNOWN')}: {indicator.get('description', '')}")
    else:
        if result.get("status") == "ERROR":
            print(f"   • Pipeline error: {result.get('error', 'Unknown error')}")
        else:
            print("   • No risk indicators returned")
    
    print("\n💀 SAFETY FINDINGS:")
    safety = result.get("safety_audit", {})
    if safety:
        print(f"   Risk Level: {safety.get('risk_level', 'UNKNOWN')}")
        print(f"   Critical Claims: {len(safety.get('critical_claims', []))}")
        print(f"   Clinical Review: {safety.get('clinical_review_required', 'UNKNOWN')}")
    else:
        print("   No safety audit returned")
    
    return result


if __name__ == "__main__":
    # Run standard test case
    result = main()
    
    # Run high-risk test case
    print("\n\n")
    risky_result = test_with_hallucinations()
    
    print("\n" + "=" * 80)
    print("✓ TESTS COMPLETE - XAI SYSTEM OPERATIONAL")
    print("=" * 80)
