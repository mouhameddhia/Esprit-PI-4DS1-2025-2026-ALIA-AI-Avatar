#!/usr/bin/env python
"""Quick test of the answer validation logic."""

import sys
from pathlib import Path

# Add the hybrid_medical_agent to path
sys.path.insert(0, str(Path(__file__).parent))

from hybrid_medical_agent.agent.controller import HybridMedicalController
from hybrid_medical_agent.agent.competency_framework import CompetencyLevel

def test_validation():
    """Test the answer validation against KB."""
    workspace_root = Path(__file__).parent.parent.parent.parent.parent
    print(f"Workspace root: {workspace_root}")
    
    # Initialize controller
    controller = HybridMedicalController(
        workspace_root=workspace_root,
        competency_level=CompetencyLevel.JUNIOR,
    )
    
    # Test cases with different answers
    test_cases = [
        {
            "answer": "HYDRA cures cancer and treats infections.",
            "drug": "HYDRA",
            "level": 1,
            "language": "en",
            "context": "JSON knowledge about HYDRA",
        },
        {
            "answer": "BACTOL is used for treating bacterial infections.",
            "drug": "BACTOL",
            "level": 3,
            "language": "en",
            "context": "JSON knowledge about BACTOL",
        },
        {
            "answer": "Hello, I want to discuss COSMOPHARMA.",
            "drug": "COSMOPHARMA",
            "level": 2,
            "language": "en",
            "context": "JSON knowledge about COSMOPHARMA",
        },
    ]
    
    print("\n" + "="*60)
    print("TESTING ANSWER VALIDATION LOGIC")
    print("="*60 + "\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['answer']}")
        print(f"  Drug: {test['drug']}")
        print(f"  Level: {test['level']}")
        print(f"  Language: {test['language']}")
        
        objection = controller._validate_rep_answer_against_kb(
            rep_answer=test['answer'],
            resolved_drug=test['drug'],
            alia_level=test['level'],
            language=test['language'],
            knowledge_context=test['context'],
        )
        
        if objection:
            print(f"  ✓ Objection generated:\n{objection[:100]}...")
        else:
            print(f"  ✓ No objection (answer acceptable)")
        print()
    
    print("="*60)
    print("✓ Validation logic test complete")
    print("="*60)

if __name__ == "__main__":
    test_validation()
