# prompts.py
"""
Prompt templates for Agent 1 - Conversation Manager
"""

from typing import Dict, Any


def build_intent_prompt(
    user_input: str,
    system_mode: str,
    conversation_state: Dict[str, Any]
) -> str:
    """
    Build the classification prompt based on system mode
    
    Args:
        user_input: The user's message
        system_mode: "training" or "doctor"
        conversation_state: Current conversation context
    
    Returns:
        Formatted prompt string
    """
    
    if system_mode == "training":
        intents_section = _get_training_intents()
        next_agent = "knowledge_retrieval"
    else:
        intents_section = _get_doctor_intents()
        next_agent = "knowledge_retrieval"
    
    # Format conversation state
    state_info = _format_conversation_state(conversation_state)
    
    prompt = f"""Classify this medical conversation input.

Mode: {system_mode}
Context: Turn {conversation_state.get('turn_number', 1)}, Stage: {conversation_state.get('stage', 'unknown')}

Input: "{user_input}"

{intents_section}

Return ONLY valid JSON (no markdown):
{{
  "intent": "<classify from list above>",
  "entities": ["<drug names if any>"],
  "dialogue_state": "<greeting|introduction|discussion|objection|closing|unknown>",
  "next_agent": "{next_agent}"
}}"""
    
    return prompt


def _get_training_intents() -> str:
    """Return training mode intent descriptions"""
    return """TRAINING intents:
- greeting: Greets, starts conversation
- product_introduction: Presents pharmaceutical product
- clinical_evidence: Discusses clinical data, efficacy
- safety_information: Presents safety, side effects
- objection_response: Responds to concerns
- closing: Wraps up, asks for commitment
- off_topic: Unrelated topic
- unclear: Ambiguous input"""


def _get_doctor_intents() -> str:
    """Return doctor mode intent descriptions"""
    return """DOCTOR intents:
- efficacy_question: Effectiveness, outcomes
- safety_question: Side effects, safety
- dosage_question: Dosing, administration
- mechanism_question: How drug works
- contraindication_question: Contraindications, warnings
- general_question: General product questions
- off_topic: Unrelated topic
- unclear: Ambiguous input"""


def _format_conversation_state(state: Dict[str, Any]) -> str:
    """Format conversation state for the prompt"""
    if not state:
        return "No prior conversation context"
    
    stage = state.get("stage", "unknown")
    turn = state.get("turn_number", 1)
    topics = state.get("mentioned_topics", [])
    last_intent = state.get("last_intent", "none")
    
    topics_str = ", ".join(topics) if topics else "none"
    
    return f"""- Current stage: {stage}
- Turn number: {turn}
- Topics mentioned: {topics_str}
- Last intent: {last_intent}"""


def build_routing_prompt(intent: str, system_mode: str) -> str:
    """
    Optional: Build prompt for routing decision
    Currently routing is rule-based, but this allows LLM-based routing if needed
    """
    return f"""Based on the detected intent "{intent}" in {system_mode} mode,
determine which agent should handle the next step.

Available agents:
- doctor_simulation (for training greeting/closing flow)
- knowledge_retrieval (for training medical content and doctor mode)
- fallback (for unclear cases)

Return only the agent name."""
