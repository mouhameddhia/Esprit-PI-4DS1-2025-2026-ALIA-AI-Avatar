#!/usr/bin/env python
"""Test training mode with answer validation and objections."""

import sys
from pathlib import Path

# Add the hybrid_medical_agent to path
sys.path.insert(0, str(Path(__file__).parent))

from hybrid_medical_agent.agent.controller import HybridMedicalController
from hybrid_medical_agent.agent.competency_framework import CompetencyLevel

def test_training_mode_with_validation():
    """Test full training mode flow with answer validation."""
    workspace_root = Path(__file__).parent.parent.parent.parent.parent
    print(f"Workspace root: {workspace_root}\n")
    
    # Initialize controller
    controller = HybridMedicalController(
        workspace_root=workspace_root,
        competency_level=CompetencyLevel.JUNIOR,
    )
    
    print("="*80)
    print("TRAINING MODE WITH ANSWER VALIDATION TEST")
    print("="*80 + "\n")
    
    # Simulate a training conversation
    conversation_history = []
    conversation_state = {}
    
    # User 1: Greeting
    print("USER: Hello, I'm a medical rep")
    print("-" * 80)
    
    response = controller.handle_query(
        question="Hello, I'm a medical rep",
        mode="training",
        user_id="test_user_1",
        messages=conversation_history,
        conversation_state=conversation_state,
    )
    
    print(f"ASSISTANT: {response.answer[:200]}...")
    if response.follow_up:
        print(f"FOLLOW-UP (Objection): {response.follow_up[:200]}...")
    print(f"Source: {response.source}, Topic: {response.topic}\n")
    
    # Add to history
    conversation_history.append({"role": "user", "content": "Hello, I'm a medical rep"})
    conversation_history.append({"role": "assistant", "content": response.answer})
    
    # User 2: Product introduction
    print("USER: I want to discuss BACTOL")
    print("-" * 80)
    
    response = controller.handle_query(
        question="I want to discuss BACTOL",
        mode="training",
        user_id="test_user_1",
        messages=conversation_history,
        conversation_state=conversation_state,
    )
    
    print(f"ASSISTANT: {response.answer[:200]}...")
    if response.follow_up:
        print(f"FOLLOW-UP: {response.follow_up[:200]}...")
    print(f"Source: {response.source}, Topic: {response.topic}\n")
    
    conversation_history.append({"role": "user", "content": "I want to discuss BACTOL"})
    conversation_history.append({"role": "assistant", "content": response.answer})
    
    # User 3: Factual claim (intentionally wrong to trigger objection)
    print("USER: BACTOL cures all viral infections and is safe for newborns")
    print("-" * 80)
    
    response = controller.handle_query(
        question="BACTOL cures all viral infections and is safe for newborns",
        mode="training",
        user_id="test_user_1",
        messages=conversation_history,
        conversation_state=conversation_state,
    )
    
    print(f"ASSISTANT: {response.answer[:300]}...")
    if response.follow_up:
        print(f"\n⚠️  FOLLOW-UP (OBJECTION):\n{response.follow_up}\n")
    else:
        print("\n(No objection generated)")
    print(f"Source: {response.source}, Topic: {response.topic}\n")
    
    print("="*80)
    print("✓ Test complete - check if objection was generated for wrong answer")
    print("="*80)

if __name__ == "__main__":
    try:
        test_training_mode_with_validation()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
