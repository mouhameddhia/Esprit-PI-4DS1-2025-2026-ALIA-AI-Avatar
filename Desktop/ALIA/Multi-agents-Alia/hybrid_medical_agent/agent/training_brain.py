"""Dynamic LLM-driven training brain for the Hybrid Medical Agent."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib import request

from .competency_framework import coerce_competency_level, get_competency_training_brief

DEFAULT_BRAIN_ACTIONS = {"ask_question", "respond", "clarify", "challenge"}
DEFAULT_BRAIN_TOPICS = {
    "indications",
    "dosage",
    "composition",
    "warnings",
    "side_effects",
    "mechanism_of_action",
    "administration",
    "age",
    "safety",
    "patient_profile",
    "other",
}
FRENCH_TOKENS = {
    "bonjour",
    "salut",
    "merci",
    "comment",
    "pourquoi",
    "dosage",
    "effets secondaires",
    "medicament",
    "produit",
    "indications",
    "posologie",
    "contre-indications",
}


@dataclass
class TrainingBrainDecision:
    """Structured action returned by the training brain."""

    action: str
    message: str
    topic: str | None = None


class TrainingBrain:
    """LLM-powered decision engine for dynamic training conversations."""

    def __init__(self, model: str = "llama3:8b", base_url: str = "http://localhost:11434", timeout_seconds: int = 90) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _normalize_history(messages: list[dict[str, str]]) -> str:
        lines: list[str] = []
        for message in messages:
            role = message.get("role", "unknown")
            content = message.get("content", "").strip()
            if content:
                lines.append(f"{role.upper()}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _normalize_text(text: str) -> str:
        lowered = text.lower()
        cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
        return " ".join(cleaned.split())

    def _latest_assistant_message(self, messages: list[dict[str, str]]) -> str:
        for message in reversed(messages):
            if str(message.get("role", "")).lower() != "assistant":
                continue
            content = str(message.get("content", "")).strip()
            if content:
                return content
        return ""

    def _is_repetitive(self, candidate: str, messages: list[dict[str, str]]) -> bool:
        if not candidate:
            return False
        latest = self._latest_assistant_message(messages)
        if not latest:
            return False
        return self._normalize_text(candidate) == self._normalize_text(latest)

    @staticmethod
    def _detect_language(text: str) -> str:
        lowered = " ".join(text.lower().split())
        if any(token in lowered for token in FRENCH_TOKENS):
            return "fr"
        return "en"

    def _build_prompt(
        self,
        *,
        conversation_history: list[dict[str, str]],
        conversation_state: dict[str, Any] | None,
        memory_context: dict[str, Any] | None,
        detected_product: str | None,
        alia_level: int,
        last_topic: str | None,
        knowledge_context: str,
        user_message: str,
        forced_language: str | None = None,
    ) -> str:
        history_text = self._normalize_history(conversation_history)
        product_text = detected_product or "unknown"
        topic_text = last_topic or "unknown"
        knowledge_text = knowledge_context.strip() or "No knowledge context available."
        state_text = json.dumps(conversation_state or {}, ensure_ascii=False, indent=2)
        memory_text = json.dumps(memory_context or {}, ensure_ascii=False, indent=2)
        recent_memory_text = ""
        if isinstance(memory_context, dict):
            recent_memory = memory_context.get("recent_memory", [])
            if isinstance(recent_memory, list):
                preview: list[str] = []
                for item in recent_memory[-6:]:
                    if not isinstance(item, dict):
                        continue
                    role = str(item.get("role", "")).strip().upper() or "UNKNOWN"
                    content = str(item.get("content", "")).strip()
                    if content:
                        preview.append(f"{role}: {content}")
                recent_memory_text = "\n".join(preview)
        response_language = forced_language if forced_language in {"fr", "en"} else self._detect_language(user_message)
        language_instruction = "Reply in English."
        if response_language == "fr":
            language_instruction = "Reponds en francais."

        competency_level = coerce_competency_level(alia_level)
        competency_brief = get_competency_training_brief(competency_level)

        return f"""You are the dynamic training brain of a senior medical doctor coaching a pharmaceutical representative.

    You are dual-role at the same time:
    - ROLE A: Simulate a real medical doctor discussing clinical details.
    - ROLE B: Coach the representative to improve clarity, structure, and evidence-based detailing quality.

You must decide the next conversational move in real time.

HIGHEST PRIORITIES:
1. Conversation context from the history
2. JSON knowledge base content if provided
3. RAG fallback content if provided
4. Training behavior rules

BEHAVIOR RULES:
- Never restart onboarding if the product or context has already been established.
- If the conversation state says product_locked=true, never ask "which product" again.
- Do not repeat the same question if the conversation already covered it.
- Stay natural, adaptive, and clinically grounded.
- Use only the provided knowledge context. Do not invent medical facts.
- Choose the most appropriate next move based on the conversation state.
- If the user is clear and context exists, respond directly rather than interrogating.
- If the user is unclear, clarify once and move forward.
- If the user needs challenge or coaching, keep it realistic and concise.
- If the user says "better for elderly", "old people", or similar, interpret that clinically as tolerability, safety, ease of use, or suitability for older patients and probe that meaning.
- Always include one coaching signal in training mode: reinforce, challenge, or ask one follow-up that improves clinical communication quality.
- Avoid repetitive stock phrases and avoid repeating the same sentence patterns.
- Never assume a specific person name or product name from prior examples. Use only the current conversation state and detected product.
- Always reply in the same language as the current user message (French or English).
- Output a single JSON object only.

COMPETENCY BRIEF:
{competency_brief}

INPUTS:
- ALIA level: {alia_level}
- Detected product: {product_text}
- Last topic discussed: {topic_text}
- Current user message: {user_message}
- Language rule: {language_instruction}

CONVERSATION HISTORY:
{history_text or "(empty)"}

CONVERSATION STATE:
{state_text or "{}"}

MEMORY CONTEXT:
{memory_text}

PERSISTENT RECENT MEMORY (LATEST TURNS):
{recent_memory_text or "(empty)"}

KNOWLEDGE CONTEXT:
{knowledge_text}

RETURN THIS EXACT JSON SHAPE:
{{
  "action": "ask_question | respond | clarify | challenge",
  "message": "...",
  "topic": "indications | dosage | composition | warnings | side_effects | mechanism_of_action | administration | age | safety | patient_profile | other"
}}

If you need to choose a topic that is not listed, use "other".
If no product is known, prefer "clarify" or "ask_question".
If the user is greeting or making small talk, respond naturally and continue the conversation.
"""

    @staticmethod
    def _extract_json_object(raw_text: str) -> str:
        text = raw_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE | re.DOTALL)
            text = re.sub(r"\s*```$", "", text, flags=re.DOTALL)
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return text

    def _detect_contradiction(
        self,
        user_message: str,
        knowledge_context: str,
        detected_product: str | None,
    ) -> tuple[bool, str | None]:
        """Detect simple contradictions between a user's claim and JSON knowledge.

        Returns (is_contradiction, corrected_statement_or_None).
        This is a best-effort check that parses knowledge_context as JSON when
        possible and looks for obvious mismatches in composition/active
        ingredient declarations.
        """
        if not knowledge_context or not knowledge_context.strip():
            return False, None

        # Try to parse knowledge_context as JSON and look for common keys.
        try:
            kc = json.loads(knowledge_context)
        except Exception:
            kc = None

        # Simple claim extraction: look for phrases like "contains X", "active ingredient is X"
        m = re.search(r"(?:contains|contain|contains:|active ingredient is|active ingredient:|composed of|contains the active|it contains)\s+([A-Za-z0-9\-\s%]+)", user_message, flags=re.IGNORECASE)
        if not m:
            # try a looser pattern: "is X" after product mention
            if detected_product and detected_product.lower() in user_message.lower():
                m2 = re.search(rf"{re.escape(detected_product)}\s+(?:is|contains|has)\s+([A-Za-z0-9\-\s%]+)", user_message, flags=re.IGNORECASE)
                if m2:
                    m = m2

        if not m:
            return False, None

        claimed = m.group(1).strip().lower()

        # If we have JSON parsed, check common fields
        if isinstance(kc, dict):
            # normalize keys to search
            candidates = []
            for key in ("composition", "active_ingredient", "active", "ingredients", "components"):
                val = kc.get(key)
                if isinstance(val, str):
                    candidates.append(val.lower())
                elif isinstance(val, list):
                    candidates.extend([str(x).lower() for x in val])

            # also search nested product payloads
            if not candidates:
                # flatten any string values
                for v in kc.values():
                    if isinstance(v, str):
                        candidates.append(v.lower())

            for cand in candidates:
                # if claimed appears in candidate -> no contradiction
                if claimed in cand or cand in claimed:
                    return False, None

            # If none matched, produce a corrected short statement from JSON if possible
            if candidates:
                corrected = candidates[0]
                return True, f"According to our JSON knowledge, the composition/active ingredient is: {corrected}."

        # If knowledge_context is plain text, do a case-insensitive search
        txt = knowledge_context.lower()
        if claimed in txt:
            return False, None

        # No match found -> contradiction detected but no corrective text available
        return True, None

    def _sanitize_action(self, action: Any) -> str:
        value = str(action).strip().lower()
        if value in DEFAULT_BRAIN_ACTIONS:
            return value
        return "respond"

    def _sanitize_topic(self, topic: Any) -> str | None:
        if topic is None:
            return None
        value = str(topic).strip().lower()
        if not value:
            return None
        return value if value in DEFAULT_BRAIN_TOPICS else "other"

    def _fallback_decision(
        self,
        detected_product: str | None,
        knowledge_context: str,
        user_message: str,
        forced_language: str | None = None,
    ) -> TrainingBrainDecision:
        language = forced_language if forced_language in {"fr", "en"} else self._detect_language(user_message)
        if not detected_product:
            message = "Which product would you like to discuss so I can keep the coaching grounded?"
            if language == "fr":
                message = "Quel produit souhaitez-vous discuter pour que le coaching reste bien cible ?"
            return TrainingBrainDecision(
                action="clarify",
                message=message,
                topic="other",
            )
        if knowledge_context.strip():
            message = f"For {detected_product}, state your core clinical claim in one sentence, then justify it with one evidence-backed point."
            if language == "fr":
                message = f"Pour {detected_product}, formulez votre message clinique central en une phrase, puis justifiez-le avec un point appuye par les donnees."
            return TrainingBrainDecision(
                action="respond",
                message=message,
                topic="other",
            )
        message = f"Let us stay practical with {detected_product}. How would you position it in one short clinical sentence?"
        if language == "fr":
            message = f"Restons pratiques avec {detected_product}. Comment le positionneriez-vous en une courte phrase clinique ?"
        return TrainingBrainDecision(
            action="challenge",
            message=message,
            topic="other",
        )

    def _non_repetitive_fallback(
        self,
        *,
        detected_product: str | None,
        topic: str | None,
        user_message: str,
        forced_language: str | None = None,
    ) -> TrainingBrainDecision:
        language = forced_language if forced_language in {"fr", "en"} else self._detect_language(user_message)
        if language == "fr":
            product = detected_product or "ce produit"
            if topic in {"indications", "composition", "dosage", "warnings", "side_effects"}:
                message = f"Bien. Maintenant, reformulez le point-clé sur {product} en 20 secondes, puis donnez un exemple patient concret."
            else:
                message = f"Allons un cran plus loin sur {product}: quel benefice clinique principal mettriez-vous en avant, et pourquoi ?"
            return TrainingBrainDecision(action="challenge", message=message, topic=topic or "other")

        product = detected_product or "this product"
        if topic in {"indications", "composition", "dosage", "warnings", "side_effects"}:
            message = f"Good. Now reframe the key point for {product} in a 20-second pitch, then add one concrete patient example."
        else:
            message = f"Let us go one level deeper on {product}: what is the single most important clinical benefit, and why?"
        return TrainingBrainDecision(action="challenge", message=message, topic=topic or "other")

    def decide_next_action(
        self,
        *,
        conversation_history: list[dict[str, str]],
        conversation_state: dict[str, Any] | None,
        memory_context: dict[str, Any] | None,
        detected_product: str | None,
        alia_level: int,
        last_topic: str | None,
        knowledge_context: str,
        user_message: str,
        forced_language: str | None = None,
    ) -> TrainingBrainDecision:
        # Early contradiction detection against JSON knowledge (deterministic guard)
        is_contra, correction_text = self._detect_contradiction(
            user_message=user_message, knowledge_context=knowledge_context, detected_product=detected_product
        )
        if is_contra:
            # Choose tone based on ALIA level
            level = coerce_competency_level(alia_level)
            if correction_text is None:
                correction_text = "I will confirm this information."

            if level.name == "BEGINNER":
                message = f"This is incorrect. {correction_text}"
            elif level.name == "JUNIOR":
                message = f"This is incorrect. {correction_text}"
            elif level.name == "CONFIRMED":
                message = f"This is incorrect. {correction_text}"
            else:  # EXPERT
                message = f"This is incorrect. {correction_text}"

            return TrainingBrainDecision(action="respond", message=message, topic="other")

        prompt = self._build_prompt(
            conversation_history=conversation_history,
            conversation_state=conversation_state,
            memory_context=memory_context,
            detected_product=detected_product,
            alia_level=alia_level,
            last_topic=last_topic,
            knowledge_context=knowledge_context,
            user_message=user_message,
            forced_language=forced_language,
        )

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_ctx": 8192,
            },
        }

        req = request.Request(
            url=f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
            body = json.loads(raw)
            raw_answer = str(body.get("response", "")).strip()
            parsed = json.loads(self._extract_json_object(raw_answer))
            action = self._sanitize_action(parsed.get("action"))
            message = str(parsed.get("message", "")).strip()
            topic = self._sanitize_topic(parsed.get("topic"))

            if not message:
                return self._fallback_decision(
                    detected_product=detected_product,
                    knowledge_context=knowledge_context,
                    user_message=user_message,
                    forced_language=forced_language,
                )

            if self._is_repetitive(message, conversation_history):
                return self._non_repetitive_fallback(
                    detected_product=detected_product,
                    topic=topic,
                    user_message=user_message,
                    forced_language=forced_language,
                )

            return TrainingBrainDecision(action=action, message=message, topic=topic)
        except Exception:
            return self._fallback_decision(
                detected_product=detected_product,
                knowledge_context=knowledge_context,
                user_message=user_message,
                forced_language=forced_language,
            )
