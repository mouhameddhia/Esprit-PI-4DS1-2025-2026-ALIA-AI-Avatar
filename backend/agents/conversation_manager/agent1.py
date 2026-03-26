# agent1.py
"""
Agent 1 - Conversation Manager

Clean, focused implementation.

Responsibilities:
- Intent detection
- Entity extraction (drug names)
- Dialogue state management
- Routing to next agent

NOT responsible for:
- Mode detection (handled by authentication)
- Content generation
- Knowledge retrieval
- Scoring or evaluation
"""

import json
from typing import Dict, Any, Optional
from ollama import Client

# Handle both relative imports (when used as module) and direct imports (when run as script)
try:
    from .prompts import build_intent_prompt
    from .schemas import (
        SystemMode,
        TrainingIntent,
        DoctorIntent,
        AgentResponse,
        ConversationState
    )
except ImportError:
    from prompts import build_intent_prompt
    from schemas import (
        SystemMode,
        TrainingIntent,
        DoctorIntent,
        AgentResponse,
        ConversationState
    )


class ConversationManagerAgent:
    """
    Agent 1 - Conversation Manager
    
    Receives:
    - user_input: str
    - system_mode: str ("training" or "doctor")
    - conversation_state: Dict[str, Any]
    
    Returns:
    - intent: str
    - entities: List[str]
    - dialogue_state: str
    - next_agent: str
    """
    
    def __init__(self, model_name: str = "llama3:8b"):
        """Initialize the conversation manager with LLM client"""
        self.client = Client()
        self.model = model_name
        
        # LLM parameters optimized for classification stability
        self.llm_options = {
            "temperature": 0.1,   # Very low for consistent classification
            "top_p": 0.9,
            "num_predict": 300    # Enough tokens for JSON response
        }
    
    def process(
        self,
        user_input: str,
        system_mode: str,
        conversation_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main processing method
        
        Args:
            user_input: The user's message
            system_mode: "training" or "doctor"
            conversation_state: Current conversation context (optional)
        
        Returns:
            Dictionary containing intent, entities, dialogue_state, next_agent
        """
        
        # Validate system mode
        if system_mode not in [SystemMode.TRAINING, SystemMode.DOCTOR]:
            raise ValueError(f"Invalid system_mode: {system_mode}. Must be 'training' or 'doctor'")
        
        # Initialize conversation state if not provided
        if conversation_state is None:
            conversation_state = ConversationState().to_dict()
        
        # Step 1: Build classification prompt
        prompt = build_intent_prompt(
            user_input=user_input,
            system_mode=system_mode,
            conversation_state=conversation_state
        )
        
        # Step 2: Call LLM for classification
        llm_response = self._call_llm(prompt)
        
        # Step 3: Parse structured output
        result = self._parse_response(llm_response, system_mode)
        
        # Step 4: Determine next agent (rule-based routing)
        result["next_agent"] = self._determine_next_agent(
            result["intent"],
            system_mode
        )
        
        return result
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM with the classification prompt
        
        Args:
            prompt: Formatted prompt string
        
        Returns:
            Raw LLM response
        """
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options=self.llm_options
            )
            raw_content = response["message"]["content"]
            
            # Debug: Check if response is empty
            if not raw_content or raw_content.strip() == "":
                print(f"⚠️ LLM returned empty response")
                print(f"📝 Prompt length: {len(prompt)} chars")
                return '{"intent": "unclear", "entities": [], "dialogue_state": "unknown"}'
            
            return raw_content
            
        except Exception as e:
            print(f"⚠️ LLM call failed: {e}")
            return '{"intent": "unclear", "entities": [], "dialogue_state": "unknown"}'
    
    def _parse_response(self, llm_response: str, system_mode: str) -> Dict[str, Any]:
        """
        Parse LLM response into structured format
        
        Implements safety fallback if parsing fails
        
        Args:
            llm_response: Raw LLM output
            system_mode: Current system mode
        
        Returns:
            Parsed dictionary with intent, entities, dialogue_state
        """
        try:
            # Clean response (remove markdown code blocks if present)
            cleaned = llm_response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            # Extract JSON object if LLM added extra text
            # Find the first { and last } to extract just the JSON
            first_brace = cleaned.find('{')
            last_brace = cleaned.rfind('}')
            if first_brace != -1 and last_brace != -1:
                cleaned = cleaned[first_brace:last_brace + 1]
            
            # Parse JSON
            parsed = json.loads(cleaned)
            
            # Validate required fields
            if "intent" not in parsed:
                raise ValueError("Missing 'intent' field")
            
            # Ensure entities is a list
            if "entities" not in parsed:
                parsed["entities"] = []
            elif not isinstance(parsed["entities"], list):
                parsed["entities"] = [parsed["entities"]]
            
            # Ensure dialogue_state exists
            if "dialogue_state" not in parsed:
                parsed["dialogue_state"] = self._infer_dialogue_state(parsed["intent"])
            
            return {
                "intent": parsed["intent"],
                "entities": parsed["entities"],
                "dialogue_state": parsed["dialogue_state"]
            }
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"⚠️ Failed to parse LLM response: {e}")
            print(f"Raw response: {llm_response[:200]}")
            
            # Fallback response
            return {
                "intent": "unclear",
                "entities": [],
                "dialogue_state": "unknown"
            }
    
    def _infer_dialogue_state(self, intent: str) -> str:
        """Infer dialogue state from intent if not provided by LLM"""
        state_mapping = {
            "greeting": "greeting",
            "product_introduction": "introduction",
            "clinical_evidence": "discussion",
            "safety_information": "discussion",
            "objection_response": "objection",
            "closing": "closing",
            "efficacy_question": "discussion",
            "safety_question": "discussion",
            "dosage_question": "discussion",
            "mechanism_question": "discussion",
            "contraindication_question": "discussion",
        }
        return state_mapping.get(intent, "unknown")
    
    def _determine_next_agent(self, intent: str, system_mode: str) -> str:
        """
        Determine next agent based on intent and mode
        
        Rule-based routing (more stable than LLM-based)
        
        Args:
            intent: Classified intent
            system_mode: Current system mode
        
        Returns:
            Name of next agent to handle the request
        """
        
        # Handle unclear or off-topic cases
        if intent in ["unclear", "off_topic"]:
            return "fallback"
        
        # Route based on system mode
        if system_mode == SystemMode.TRAINING:
            # In training mode, route to doctor simulation
            return "doctor_simulation"
        
        elif system_mode == SystemMode.DOCTOR:
            # In doctor mode, route to knowledge retrieval
            return "knowledge_retrieval"
        
        # Default fallback
        return "fallback"



# ==================== Testing ====================

if __name__ == "__main__":
    """
    Local testing with various scenarios
    """
    
    print("=" * 60)
    print("🧪 Testing Agent 1 - Conversation Manager")
    print("=" * 60)
    
    agent = ConversationManagerAgent()
    
    # Test scenarios for TRAINING mode
    training_tests = [
        "Hello doctor, good morning!",
        "I'd like to introduce you to our new medication, Cardiomax",
        "Our clinical trials showed 85% efficacy in patients",
        "The safety profile is excellent with minimal side effects",
        "I understand your concerns about the cost, but...",
        "Thank you for your time. Can we schedule a follow-up?"
    ]
    
    # Test scenarios for DOCTOR mode
    doctor_tests = [
        "How effective is this drug compared to the standard treatment?",
        "What are the common side effects?",
        "What's the recommended dosage for elderly patients?",
        "How does this medication work?",
        "Are there any contraindications I should know about?",
        "Tell me about the latest research on this drug"
    ]
    
    print("\n📘 TRAINING MODE Tests:")
    print("-" * 60)
    
    for i, test_input in enumerate(training_tests, 1):
        print(f"\n{i}. Input: \"{test_input}\"")
        
        result = agent.process(
            user_input=test_input,
            system_mode="training",
            conversation_state={
                "stage": "discussion",
                "turn_number": i,
                "mentioned_topics": [],
                "last_intent": None
            }
        )
        
        print(f"   Intent: {result['intent']}")
        print(f"   Entities: {result['entities']}")
        print(f"   State: {result['dialogue_state']}")
        print(f"   Next Agent: {result['next_agent']}")
    
    print("\n\n📗 DOCTOR MODE Tests:")
    print("-" * 60)
    
    for i, test_input in enumerate(doctor_tests, 1):
        print(f"\n{i}. Input: \"{test_input}\"")
        
        result = agent.process(
            user_input=test_input,
            system_mode="doctor",
            conversation_state={
                "stage": "discussion",
                "turn_number": i,
                "mentioned_topics": [],
                "last_intent": None
            }
        )
        
        print(f"   Intent: {result['intent']}")
        print(f"   Entities: {result['entities']}")
        print(f"   State: {result['dialogue_state']}")
        print(f"   Next Agent: {result['next_agent']}")
    
    print("\n" + "=" * 60)
    print("✅ Testing complete!")
    print("=" * 60)
