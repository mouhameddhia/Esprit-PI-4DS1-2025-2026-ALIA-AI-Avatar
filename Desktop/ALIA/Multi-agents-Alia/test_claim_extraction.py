#!/usr/bin/env python
"""Debug script to test claim extraction and validation."""

import sys
from pathlib import Path

# Add the hybrid_medical_agent to path
sys.path.insert(0, str(Path(__file__).parent))

from hybrid_medical_agent.agent.controller import HybridMedicalController

def test_claim_extraction():
    """Test the claim extraction logic."""
    controller = HybridMedicalController(
        workspace_root=Path(__file__).parent.parent.parent.parent.parent,
    )
    
    test_answer = "BACTOL is excellent for treating viral infections and is completely safe for newborns"
    
    print("="*80)
    print("TESTING CLAIM EXTRACTION")
    print("="*80)
    print(f"\nTest answer: {test_answer}\n")
    
    claims = controller._extract_claims_from_answer(test_answer, "en")
    
    print(f"Detected {len(claims)} claims:")
    for i, (claim_type, claim_text) in enumerate(claims, 1):
        print(f"  {i}. Type: {claim_type}")
        print(f"     Text: '{claim_text}'")
    
    print("\n" + "="*80)
    print("Testing against BACTOL KB:")
    print("="*80)
    
    # Get BACTOL payload
    payload = controller.json_retriever.get_payload("BACTOL")
    if payload:
        print("\n✓ BACTOL payload found")
        
        # Extract ground truth
        ground_truth = {
            "indications": controller._extract_field(payload, "indications"),
            "composition": controller._extract_field(payload, "composition"),
            "dosage": controller._extract_field(payload, "dosage"),
            "warnings": controller._extract_field(payload, "warnings"),
            "side_effects": controller._extract_field(payload, "side_effects"),
            "benefits": controller._extract_field(payload, "benefits"),
        }
        
        print("\nGround truth fields:")
        for field, value in ground_truth.items():
            if value:
                print(f"\n  {field.upper()}:")
                print(f"    {value[:150]}...")
        
        # Check each claim
        print("\n" + "-"*80)
        print("Checking claims against ground truth:")
        print("-"*80)
        
        for claim_type, claim_text in claims:
            mismatch = controller._check_claim_against_ground_truth(
                claim_type=claim_type,
                claim_text=claim_text,
                ground_truth=ground_truth,
            )
            
            if mismatch:
                print(f"\n⚠️  MISMATCH DETECTED:")
                print(f"    Claim type: {claim_type}")
                print(f"    Rep claim: '{claim_text}'")
                print(f"    Ground truth: '{ground_truth.get(claim_type, 'N/A')[:100]}'")
            else:
                print(f"\n✓ Claim OK: {claim_type} - '{claim_text}'")
    else:
        print("\n✗ BACTOL payload not found")

if __name__ == "__main__":
    test_claim_extraction()
