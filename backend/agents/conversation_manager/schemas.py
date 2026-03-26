# schemas.py
"""
Data models and schemas for Agent 1 - Conversation Manager
"""

from typing import Dict, Any, List
from enum import Enum


class SystemMode(str, Enum):
    """System operation modes"""
    TRAINING = "training"
    DOCTOR = "doctor"


class TrainingIntent(str, Enum):
    """Intents available in training mode"""
    GREETING = "greeting"
    PRODUCT_INTRODUCTION = "product_introduction"
    CLINICAL_EVIDENCE = "clinical_evidence"
    SAFETY_INFORMATION = "safety_information"
    OBJECTION_RESPONSE = "objection_response"
    CLOSING = "closing"
    OFF_TOPIC = "off_topic"
    UNCLEAR = "unclear"


class DoctorIntent(str, Enum):
    """Intents available in doctor mode"""
    EFFICACY_QUESTION = "efficacy_question"
    SAFETY_QUESTION = "safety_question"
    DOSAGE_QUESTION = "dosage_question"
    MECHANISM_QUESTION = "mechanism_question"
    CONTRAINDICATION_QUESTION = "contraindication_question"
    GENERAL_QUESTION = "general_question"
    OFF_TOPIC = "off_topic"
    UNCLEAR = "unclear"


class DialogueState(str, Enum):
    """Possible dialogue states"""
    GREETING = "greeting"
    INTRODUCTION = "introduction"
    DISCUSSION = "discussion"
    OBJECTION = "objection"
    CLOSING = "closing"
    UNKNOWN = "unknown"


class AgentResponse:
    """Structure for Agent 1 response"""
    
    def __init__(
        self,
        intent: str,
        entities: List[str],
        dialogue_state: str,
        next_agent: str,
        confidence: float = 1.0
    ):
        self.intent = intent
        self.entities = entities
        self.dialogue_state = dialogue_state
        self.next_agent = next_agent
        self.confidence = confidence
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "entities": self.entities,
            "dialogue_state": self.dialogue_state,
            "next_agent": self.next_agent,
            "confidence": self.confidence
        }


class ConversationState:
    """Tracks the current conversation state"""
    
    def __init__(
        self,
        stage: str = "greeting",
        mentioned_topics: List[str] = None,
        turn_number: int = 1,
        last_intent: str = None
    ):
        self.stage = stage
        self.mentioned_topics = mentioned_topics or []
        self.turn_number = turn_number
        self.last_intent = last_intent
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "mentioned_topics": self.mentioned_topics,
            "turn_number": self.turn_number,
            "last_intent": self.last_intent
        }
    
    def update(self, intent: str, entities: List[str] = None):
        """Update conversation state based on new intent"""
        self.turn_number += 1
        self.last_intent = intent
        
        if entities:
            for entity in entities:
                if entity not in self.mentioned_topics:
                    self.mentioned_topics.append(entity)
        
        # Update stage based on intent
        if intent in ["greeting"]:
            self.stage = "greeting"
        elif intent in ["product_introduction"]:
            self.stage = "introduction"
        elif intent in ["objection_response"]:
            self.stage = "objection"
        elif intent in ["closing"]:
            self.stage = "closing"
        else:
            self.stage = "discussion"
