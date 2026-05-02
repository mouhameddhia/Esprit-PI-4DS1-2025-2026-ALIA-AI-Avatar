"""Local Ollama interface enforcing strict context-only doctor responses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import request

from .competency_framework import (
    CompetencyLevel,
    get_competency_system_prompt,
    get_competency_training_brief,
)

COMMERCIAL_DOCTOR_SYSTEM_PROMPT = """You are a professional medical information assistant.

RULES:
- Be precise and factual
- Do NOT ask questions
- Do NOT simulate conversation
- Only answer based on JSON + RAG
- If data is missing, say it is not available in knowledge base

Always base your answer ONLY on the provided context."""

TRAINING_DOCTOR_SYSTEM_PROMPT = """You are a senior medical doctor training a pharmaceutical representative.

CONVERSATION RULES (HIGHEST PRIORITY):
- NEVER ask the same question twice in the same conversation
- If product/context already mentioned before, continue from there
- Remember what was discussed and build on it
- Do NOT reset to onboarding - progress forward
- Continue conversation naturally without loops

INTERACTION RULES:
- Speak naturally like a real doctor in conversation
- Lead the conversation actively but progressively
- Gently correct or guide when needed
- Keep conversation alive and flowing
- Never hallucinate - use only JSON + RAG context
- If information is incomplete, give brief coaching hint then ask next best training question
- Do not block the conversation when product details are partial
- Respond normally to greetings, introductions, and small talk
- Keep answers concise, professional, and clinically oriented

Your goal is to train the medical representative like in a real clinical detailing session - continuous and natural, not loop-based.

Always base your answer ONLY on the provided context."""

DOCTOR_SYSTEM_PROMPT = COMMERCIAL_DOCTOR_SYSTEM_PROMPT


@dataclass
class LLMResponse:
    answer: str
    model: str


@dataclass
class IntentClassification:
    """Structured intent label predicted by the LLM router prompt."""

    label: str
    model: str


class DoctorLLM:
    """Minimal Ollama HTTP client for doctor-persona generation."""

    def __init__(
        self,
        model: str = "llama3:8b",
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 90,
        competency_level: CompetencyLevel | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.competency_level = competency_level or CompetencyLevel.JUNIOR

    @staticmethod
    def _format_history(conversation_history: list[dict[str, str]] | None) -> str:
        if not conversation_history:
            return "(empty)"

        lines: list[str] = []
        for message in conversation_history:
            role = message.get("role", "unknown")
            content = str(message.get("content", "")).strip()
            if content:
                lines.append(f"{role.upper()}: {content}")

        return "\n".join(lines) if lines else "(empty)"

    def _build_prompt(
        self,
        context: str,
        user_input: str,
        conversation_history: list[dict[str, str]] | None = None,
        fact_memory: str | None = None,
        conversation_summary: str | None = None,
    ) -> str:
        history_text = self._format_history(conversation_history)
        fact_section = fact_memory.strip() if fact_memory and fact_memory.strip() else "(empty)"
        summary_section = conversation_summary.strip() if conversation_summary and conversation_summary.strip() else "(empty)"

        # Use competency-level system prompt if available and append the
        # level-specific training brief (which contains the global generation
        # constraints such as JSON-first, RAG fallback, no-hallucination,
        # and level-adapted objection rules). This ensures all direct LLM
        # generations strictly follow those constraints.
        system_prompt = get_competency_system_prompt(self.competency_level)
        competency_brief = get_competency_training_brief(self.competency_level)

        return (
            f"SYSTEM:\n{system_prompt}\n\nGLOBAL_CONSTRAINTS:\n{competency_brief}\n\n"
            "FACT MEMORY (trusted user profile only; never use for medical claims):\n"
            f"{fact_section}\n\n"
            "CONVERSATION SUMMARY (compressed history, may be incomplete):\n"
            f"{summary_section}\n\n"
            "EVIDENCE CONTEXT (authoritative medical evidence; do not exceed it):\n"
            f"{context}\n\n"
            "CONVERSATION HISTORY (chronological):\n"
            f"{history_text}\n\n"
            "USER QUESTION:\n"
            f"{user_input}\n\n"
            "If the context does not contain the requested fact, answer exactly: "
            "Information not available in the provided data"
        )

    def generate(
        self,
        context: str,
        user_input: str,
        system_prompt: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        fact_memory: str | None = None,
        conversation_summary: str | None = None,
    ) -> LLMResponse:
        """Generate grounded answer from provided context only."""

        prompt = self._build_prompt(
            context=context,
            user_input=user_input,
            conversation_history=conversation_history,
            fact_memory=fact_memory,
            conversation_summary=conversation_summary,
        )
        if system_prompt:
            prompt = prompt.replace(DOCTOR_SYSTEM_PROMPT, system_prompt)

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": 8192,
            },
        }

        req = request.Request(
            url=f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            raw = response.read().decode("utf-8")
        body = json.loads(raw)
        answer = str(body.get("response", "")).strip()

        if not answer:
            answer = "Information not available in the provided data"

        return LLMResponse(answer=answer, model=self.model)

    def classify_intent(
        self,
        user_input: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> IntentClassification:
        """Classify user intent for controller routing decisions."""

        history_text = self._format_history(conversation_history)
        prompt = (
            "You are an intent classifier for a pharmaceutical assistant.\n"
            "Return STRICT JSON only with one key: label.\n"
            "Allowed labels:\n"
            "- medical\n"
            "- greeting\n"
            "- product_discussion\n"
            "- off_topic\n\n"
            "Classification rules:\n"
            "- greeting: hello/thanks/simple social opener without medical request\n"
            "- product_discussion: asks to compare/recommend/position products or drugs\n"
            "- medical: asks factual clinical/pharmaceutical content (indication, dosage, safety, etc.)\n"
            "- off_topic: unrelated to pharma/medical training/clinical use\n"
            "If uncertain between medical and product_discussion, choose medical.\n\n"
            f"Conversation history:\n{history_text}\n\n"
            f"User message:\n{user_input}\n"
        )

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
                "num_ctx": 4096,
            },
        }

        req = request.Request(
            url=f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            raw = response.read().decode("utf-8")

        body = json.loads(raw)
        response_text = str(body.get("response", "")).strip()
        label = "off_topic"

        try:
            parsed = json.loads(response_text)
            candidate = str(parsed.get("label", "")).strip().lower()
            if candidate in {"medical", "greeting", "product_discussion", "off_topic"}:
                label = candidate
        except json.JSONDecodeError:
            lowered = response_text.lower()
            for candidate in ("medical", "greeting", "product_discussion", "off_topic"):
                if candidate in lowered:
                    label = candidate
                    break

        return IntentClassification(label=label, model=self.model)
