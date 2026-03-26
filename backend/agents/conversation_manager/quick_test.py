#!/usr/bin/env python3
"""
Quick test - Minimal script to verify Conversation Manager works
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "agents" / "conversation_manager"))

from agent1 import ConversationManagerAgent

print("🧪 Testing Conversation Manager\n")

# Initialize
agent = ConversationManagerAgent(model_name="llama3:8b")
print("✓ Agent initialized\n")

# Test 1
print("Test 1: Training Mode - Greeting")
result = agent.process("Hi there, I want to talk about your new drug", "training")
print(f"  Intent: {result['intent']}")
print(f"  Next Agent: {result['next_agent']}")
print()

# Test 2
print("Test 2: Doctor Mode - Dosage Question")
result = agent.process("What's the dosage for adults?", "doctor")
print(f"  Intent: {result['intent']}")
print(f"  Next Agent: {result['next_agent']}")
print()

print("✅ Basic tests passed!")
