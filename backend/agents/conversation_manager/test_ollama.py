#!/usr/bin/env python3
"""
Ollama Diagnostics - Check if Ollama is working correctly
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("🔍 OLLAMA DIAGNOSTICS")
print("=" * 70)

# Test 1: Can we import Ollama client?
print("\n[1] Checking Ollama Python library...")
try:
    from ollama import Client
    print("    ✅ Ollama library imported successfully")
except ImportError as e:
    print(f"    ❌ Failed to import Ollama: {e}")
    print("    Fix: pip install ollama")
    sys.exit(1)

# Test 2: Can we connect to Ollama server?
print("\n[2] Attempting to connect to Ollama server...")
try:
    client = Client(host="http://localhost:11434")
    print("    ✅ Connected to Ollama at localhost:11434")
except Exception as e:
    print(f"    ❌ Failed to connect: {e}")
    print("    Fix: Start Ollama with: ollama serve")
    sys.exit(1)

# Test 3: Can we list models?
print("\n[3] Checking available models...")
try:
    response = client.list()
    models = response.get("models", [])
    
    if not models:
        print("    ❌ No models found!")
        print("    Fix: Download a model with: ollama pull llama3:8b")
        sys.exit(1)
    
    print(f"    ✅ Found {len(models)} model(s):")
    for model in models:
        name = model.get("name", "unknown")
        print(f"       • {name}")
    
except Exception as e:
    print(f"    ❌ Failed to list models: {e}")
    sys.exit(1)

# Test 4: Is llama3:8b available?
print("\n[4] Checking for llama3:8b model...")
model_available = any("llama3:8b" in m.get("name", "") for m in models)

if model_available:
    print("    ✅ llama3:8b is installed")
else:
    print("    ⚠️  llama3:8b not found")
    print("    Fix: ollama pull llama3:8b")
    model_to_test = models[0].get("name", "llama3:8b") if models else "llama3:8b"
    print(f"    Will test with: {model_to_test}")

# Test 5: Can we get a response from LLM?
print("\n[5] Testing LLM response (this may take 10-30 seconds)...\n")
try:
    model_to_use = "llama3:8b" if model_available else models[0].get("name")
    print(f"    Testing with model: {model_to_use}")
    print("    Sending: 'Hello, are you working?'\n")
    
    start_time = time.time()
    
    response = client.chat(
        model=model_to_use,
        messages=[{"role": "user", "content": "Hello, are you working? Answer in one word."}],
        stream=False,
        options={
            "temperature": 0.1,
            "num_predict": 50
        }
    )
    
    elapsed = time.time() - start_time
    content = response.get("message", {}).get("content", "")
    
    print(f"    Response: '{content}'")
    print(f"    ✅ LLM responded in {elapsed:.2f} seconds")
    
except Exception as e:
    print(f"    ❌ LLM call failed: {e}")
    print("    Make sure Ollama is running and the model is downloaded")
    sys.exit(1)

# Test 6: Test Conversation Manager
print("\n[6] Testing Conversation Manager Agent...\n")
try:
    from agent1 import ConversationManagerAgent
    
    print("    Initializing agent...")
    agent = ConversationManagerAgent(model_name=model_to_use)
    
    print("    Processing: 'Tell me about your new medication'")
    start_time = time.time()
    
    result = agent.process(
        user_input="Tell me about your new medication",
        system_mode="training"
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n    ✅ Agent processed in {elapsed:.2f} seconds")
    print(f"    Intent: {result.get('intent')}")
    print(f"    Entities: {result.get('entities')}")
    print(f"    Next Agent: {result.get('next_agent')}")
    
except Exception as e:
    print(f"    ❌ Agent test failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED - OLLAMA IS WORKING!")
print("=" * 70)
print("\n📊 System Status:")
print(f"   • Ollama Server: ✅ Running")
print(f"   • Models Available: ✅ {len(models)} model(s)")
print(f"   • LLM Response: ✅ Working")
print(f"   • Conversation Manager: ✅ Working")
print("\n🚀 You can now run:")
print("   python backend/main.py --demo")
print("   python backend/main.py --interactive --mode training")
print("=" * 70)
