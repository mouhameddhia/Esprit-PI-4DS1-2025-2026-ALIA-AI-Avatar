"""Main orchestration agent: JSON first, RAG fallback, doctor persona output."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .intent_detector import DetectedIntent, detect_intent
from .json_retriever import JSONRetriever
from .llm_interface import COMMERCIAL_DOCTOR_SYSTEM_PROMPT, TRAINING_DOCTOR_SYSTEM_PROMPT, DoctorLLM
from .persistent_memory import PersistentMemoryStore
from .rag_adapter import RAGAdapter
from .training_brain import TrainingBrain
from .user_memory_namespace import UserMemoryNamespace

NO_KB_MESSAGE = "No information available in knowledge base"
NO_KB_MESSAGE_FR = "Aucune information disponible dans la base de connaissances"
CLARIFY_PRODUCT_MESSAGE = "Please specify the product/drug name so I can provide the exact medical information."
CLARIFY_PRODUCT_MESSAGE_FR = "Veuillez preciser le nom du produit/medicament afin que je puisse fournir l'information medicale exacte."

CRITICAL_TOPICS = {
    "indications",
    "composition",
    "dosage",
    "administration",
    "warnings",
    "side_effects",
    "mechanism_of_action",
    "age",
}
PHARMA_DOMAIN_TERMS = {
    "pharma",
    "pharmaceutical",
    "medical",
    "medicine",
    "drug",
    "product",
    "indication",
    "indications",
    "composition",
    "dosage",
    "dose",
    "safety",
    "warning",
    "warnings",
    "side effect",
    "side effects",
    "administration",
    "treatment",
    "clinical",
    "age",
    "recommend",
    "recommendation",
    "choose",
    "best",
    "suitable",
}
OUT_OF_DOMAIN_MESSAGE = "I am specialized only in pharmaceutical-related topics."
NO_NAME_MEMORY_EN = "I do not have your name stored yet."
NO_NAME_MEMORY_FR = "Je n'ai pas encore votre nom en memoire."
GREETING_TOKENS = {"hello", "hi", "hey", "bonjour", "salut", "good morning", "good evening"}
SMALL_TALK_TOKENS = {"how are you", "thanks", "thank you", "nice to meet", "good", "great", "okay", "ok"}
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
ENGLISH_TOKENS = {
    "hello",
    "please",
    "what",
    "why",
    "how",
    "side effects",
    "warnings",
    "dosage",
    "drug",
    "medicine",
    "product",
    "indications",
    "treatment",
    "clinical",
}
MEMORY_ALLOWED_ROLE_TOKENS = {
    "doctor",
    "dr",
    "physician",
    "pharmacist",
    "cardiologist",
    "dermatologist",
    "pediatrician",
    "nurse",
    "medical representative",
    "medical rep",
    "pharma rep",
}
SENSITIVE_HEALTH_TOKENS = {
    "diagnosis",
    "diagnosed",
    "cancer",
    "hiv",
    "pregnant",
    "depression",
    "diabetes",
    "hypertension",
    "patient id",
}
NON_PHARMA_TOPICS = {
    "weather",
    "temperature",
    "rain",
    "forecast",
    "sports",
    "football",
    "soccer",
    "stock",
    "bitcoin",
    "movie",
    "music",
    "politics",
    "travel",
}
PRODUCT_SWITCH_TOKENS = {
    "switch to",
    "change to",
    "changing to",
    "talk about",
    "another product",
    "new product",
    "passons a",
    "changer vers",
    "parler de",
    "nouveau produit",
}
INVALID_NAME_TOKENS = {
    "going",
    "go",
    "talk",
    "present",
    "presentation",
    "working",
    "work",
    "medical",
    "rep",
    "doctor",
    "the",
    "a",
    "an",
}
INVALID_COMPANY_TOKENS = {
    "from",
    "company",
    "lab",
    "unknown",
    "none",
}
RAG_LEARNING_CONFIDENCE_THRESHOLD = 0.85
RAG_LEARNING_ENABLED = os.getenv("ALIA_ENABLE_RAG_LEARNING", "0") == "1"
RAG_LEARNING_MIN_DIAGNOSTIC_CONFIDENCE = 0.75
LLM_ROUTER_LABELS = {"medical", "greeting", "product_discussion", "off_topic"}

@dataclass
class AgentResponse:
    """Final response payload returned by the hybrid controller."""

    answer: str
    source: str
    context: str
    topic: str | None
    drug_name: str | None
    confidence: float
    follow_up: str | None = None
    citations: list[str] | None = None
    uncertainty: str | None = None
    conflict_notes: list[str] | None = None


@dataclass
class TrainingTurn:
    """Single doctor training interaction turn."""

    doctor_question: str
    expected_answer: str
    rep_answer: str
    feedback: str
    source: str


@dataclass
class TrainingConversationState:
    """Locked continuity state for training mode."""

    current_product: str | None = None
    product_locked: bool = False
    conversation_phase: str = "onboarding"
    last_topic: str | None = None
    last_user_intent: str | None = None
    current_intent: str | None = None
    user_role: str = "unknown"
    memory_context: dict | None = None
    turn_count: int = 0


class HybridMedicalController:
    """Decision layer orchestrating JSON, RAG fallback, and LLM response."""

    def __init__(
        self,
        workspace_root: Path,
        llm_model: str = "llama3:8b",
        llm_base_url: str = "http://localhost:11434",
    ) -> None:
        self.workspace_root = workspace_root
        self.json_retriever = JSONRetriever(workspace_root=workspace_root)
        self.rag_adapter = RAGAdapter(workspace_root=workspace_root)
        self.llm = DoctorLLM(model=llm_model, base_url=llm_base_url)
        self.training_brain = TrainingBrain(model=llm_model, base_url=llm_base_url)
        self.user_memory_namespace = UserMemoryNamespace(workspace_root=workspace_root)
        self.active_user_id = "anonymous"
        self.persistent_memory = self._memory_store_for_user(self.active_user_id)
        self._runtime_sessions: dict[str, dict[str, str | None]] = {
            "training": {
                "active_product": None,
                "current_intent": None,
            },
            "commercial": {
                "active_product": None,
                "current_intent": None,
            },
        }
        self._training_turn = 0
        self._last_training_follow_up: str | None = None

    def _memory_store_for_user(self, user_id: str | None) -> PersistentMemoryStore:
        """Create a user-scoped memory store for the current request."""

        resolved = self.user_memory_namespace.resolve(user_id)
        self.active_user_id = resolved.user_id
        return PersistentMemoryStore(
            path=resolved.persistent_path,
            conversation_path=resolved.conversation_path,
            max_qa_entries=10,
            summary_trigger_messages=10,
            recent_window_messages=6,
        )

    def reset_session(
        self,
        *,
        mode: str,
        user_id: str | None = None,
        conversation_state: dict | None = None,
        clear_short_term_memory: bool = True,
        keep_user_facts: bool = True,
    ) -> None:
        """Reset runtime session context to prevent behavior leakage across sessions."""

        self.persistent_memory = self._memory_store_for_user(user_id)

        normalized_mode = "training" if mode == "training" else "commercial"
        self._runtime_sessions[normalized_mode] = {
            "active_product": None,
            "current_intent": None,
        }

        if isinstance(conversation_state, dict):
            conversation_state["current_intent"] = "greeting"
            conversation_state["active_product"] = None
            conversation_state["current_product"] = None
            conversation_state["product_locked"] = False
            conversation_state["conversation_phase"] = "onboarding"
            conversation_state["last_topic"] = None
            conversation_state["last_user_intent"] = None
            conversation_state["turn_count"] = 0

        if clear_short_term_memory:
            self.persistent_memory.reset_session_memory(keep_user_facts=keep_user_facts)

    @staticmethod
    def _extract_name_fact(question: str) -> str | None:
        patterns = [
            r"\b(?:my name is|call me)\s+([A-Za-z][A-Za-z\-']{1,63})\b",
            r"\b(?:je m'appelle)\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\-']{1,63})\b",
            r"\b(?:i am|i'm|im)\s+([A-Za-z][A-Za-z\-']{1,63})(?=\s*(?:[.,!?;:]|$|\band\b))",
            r"\b(?:i am|i'm|im)\s+([A-Za-z][A-Za-z\-']{1,63})(?=\s+(?:a|an)?\s*(?:medical representative|medical rep|pharma rep|doctor|physician|pharmacist|cardiologist|dermatologist|pediatrician|nurse)\b)",
        ]
        for pattern in patterns:
            match = re.search(pattern, question, flags=re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                if candidate.lower() in INVALID_NAME_TOKENS:
                    continue
                return candidate
        return None

    @staticmethod
    def _extract_company_fact(question: str) -> str | None:
        patterns = [
            r"\b(?:i am from|i'm from|im from|i work at|i work for|i am with|i'm with|im with)\s+([A-Za-z0-9][A-Za-z0-9&\-'.\s]{1,80})\b",
            r"\b(?:je viens de|je suis de|je travaille chez|je travaille pour|je suis avec)\s+([A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9&\-'.\s]{1,80})\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, question, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = re.sub(r"\s+", " ", match.group(1)).strip(" .,:;!?")
            candidate = re.split(r"[.!?]", candidate, maxsplit=1)[0].strip(" .,:;!?")
            candidate = re.split(
                r"\b(?:i am|i'm|im|je suis|i prefer|my preference|je prefere)\b",
                candidate,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip(" .,:;!?")
            if len(candidate) < 3:
                continue
            if candidate.lower() in INVALID_COMPANY_TOKENS:
                continue
            return candidate
        return None

    @staticmethod
    def _is_personal_memory_query(question: str) -> bool:
        lowered = " ".join(question.lower().split())
        return any(
            phrase in lowered
            for phrase in (
                "what is my name",
                "what's my name",
                "who am i",
                "do you remember my name",
                "quel est mon nom",
                "comment je m'appelle",
                "where do i work",
                "where i work",
                "which company",
                "what company",
                "where am i from",
                "ou je travaille",
                "ou est ce que je travaille",
                "dans quelle entreprise",
                "quel est mon role",
                "what is my role",
            )
        )

    @staticmethod
    def _personal_memory_query_type(question: str) -> str:
        lowered = " ".join(question.lower().split())
        if any(
            phrase in lowered
            for phrase in (
                "what is my name",
                "what's my name",
                "who am i",
                "do you remember my name",
                "quel est mon nom",
                "comment je m'appelle",
            )
        ):
            return "name"
        if any(
            phrase in lowered
            for phrase in (
                "where do i work",
                "where i work",
                "which company",
                "what company",
                "where am i from",
                "ou je travaille",
                "ou est ce que je travaille",
                "dans quelle entreprise",
            )
        ):
            return "company"
        if any(phrase in lowered for phrase in ("what is my role", "quel est mon role")):
            return "role"
        return "unknown"

    def _capture_user_memory(self, question: str) -> None:
        stripped = question.strip()
        if not stripped:
            return

        # Persist identity-related facts immediately so they are available in the same request lifecycle.
        extracted_name = self._extract_name_fact(stripped)
        if extracted_name:
            existing_name = str(self.persistent_memory.get_user_facts().get("name", "")).strip()
            if existing_name.lower() != extracted_name.lower():
                self.persistent_memory.add_user_fact(
                    content=f"User name is {extracted_name}",
                    key="name",
                    value=extracted_name,
                )

        extracted_company = self._extract_company_fact(stripped)
        if extracted_company:
            self.persistent_memory.add_user_fact(
                content=f"User company is {extracted_company}",
                key="company",
                value=extracted_company,
            )

        role_value = self._infer_user_role(stripped)
        if role_value in {"rep", "doctor"}:
            self.persistent_memory.add_user_fact(
                content=f"User role is {role_value}",
                key="role",
                value=role_value,
            )

        lowered = " ".join(stripped.lower().split())
        if any(token in lowered for token in ("i prefer", "my preference", "prefer ", "je prefere")):
            self.persistent_memory.add_preference(stripped)

        if any(lowered.startswith(prefix) for prefix in ("please ", "you must", "you should", "il faut", "merci de")):
            self.persistent_memory.add_instruction(stripped)

        # Only keep non-sensitive professional context as generic statement memory.
        if "?" not in stripped and self._is_allowed_memory_statement(stripped):
            self.persistent_memory.add_statement(stripped)

    @staticmethod
    def _is_allowed_memory_statement(statement: str) -> bool:
        lowered = " ".join(statement.lower().split())
        if any(token in lowered for token in SENSITIVE_HEALTH_TOKENS):
            return False
        if any(token in lowered for token in MEMORY_ALLOWED_ROLE_TOKENS):
            return True
        if any(token in lowered for token in ("i work", "je travaille", "my role", "mon role", "specialty", "specialite")):
            return True
        return False

    def _answer_personal_memory(self, question: str, language: str) -> str | None:
        if not self._is_personal_memory_query(question):
            return None

        query_type = self._personal_memory_query_type(question)
        if query_type == "name":
            name = self.persistent_memory.get_latest_user_fact("name")
            if name:
                if language == "fr":
                    return f"Votre nom est {name}."
                return f"Your name is {name}."
            return NO_NAME_MEMORY_FR if language == "fr" else NO_NAME_MEMORY_EN

        if query_type == "company":
            company = self.persistent_memory.get_latest_user_fact("company")
            if company:
                if language == "fr":
                    return f"Vous travaillez chez {company}."
                return f"You work at {company}."
            if language == "fr":
                return "Je n'ai pas encore votre entreprise en memoire."
            return "I do not have your company stored yet."

        if query_type == "role":
            role_value = self.persistent_memory.get_latest_user_fact("role")
            if role_value:
                if language == "fr":
                    return f"Votre role est {role_value}."
                return f"Your role is {role_value}."
            if language == "fr":
                return "Je n'ai pas encore votre role en memoire."
            return "I do not have your role stored yet."

        return None

    def _finalize_response(self, question: str, mode: str, response: AgentResponse) -> AgentResponse:
        self.persistent_memory.add_qa(
            input_text=question,
            output_text=response.answer,
            metadata={
                "mode": mode,
                "source": response.source,
                "topic": response.topic,
                "drug_name": response.drug_name,
                "confidence": response.confidence,
                "citations": response.citations or [],
                "uncertainty": response.uncertainty,
                "conflict_notes": response.conflict_notes or [],
            },
        )
        return response

    @staticmethod
    def _stable_query_key(question: str) -> str:
        normalized = " ".join(question.strip().lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _store_high_confidence_rag(self, question: str, rag_result, threshold: float = RAG_LEARNING_CONFIDENCE_THRESHOLD) -> None:
        """Persist high-confidence RAG responses into the shared JSON cache file."""

        if not RAG_LEARNING_ENABLED:
            return

        confidence = float(getattr(rag_result, "answer_confidence", 0.0) or 0.0)
        if confidence < threshold:
            return

        citations = list(getattr(rag_result, "citations", []))
        if not citations:
            return

        uncertainty = str(getattr(rag_result, "uncertainty", "") or "").strip()
        if uncertainty:
            return

        conflict_notes = list(getattr(rag_result, "conflict_notes", []))
        if conflict_notes:
            return

        diagnostics = getattr(rag_result, "diagnostics", None)
        if diagnostics is not None:
            if bool(getattr(diagnostics, "fallback_triggered", False)):
                return

            quality = str(getattr(diagnostics, "quality", "") or "").lower()
            if quality and quality != "correct":
                return

            diagnostic_confidence = float(getattr(diagnostics, "confidence_score", 0.0) or 0.0)
            if diagnostic_confidence < RAG_LEARNING_MIN_DIAGNOSTIC_CONFIDENCE:
                return

        # Cache moved from `rag_knowledge_base` into `rag_knowledge_builder/scripts`
        cache_path = self.workspace_root / "rag_knowledge_builder" / "scripts" / "rag_response_cache.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_key = self._stable_query_key(question)

        try:
            if cache_path.exists():
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    payload = {}
            else:
                payload = {}

            if cache_key in payload:
                return

            payload[cache_key] = {
                "question": question,
                "answer": str(getattr(rag_result, "answer", "")),
                "citations": citations,
                "answer_confidence": confidence,
                "uncertainty": uncertainty,
                "conflict_notes": conflict_notes,
                "latency_ms": float(getattr(rag_result, "latency_ms", 0.0) or 0.0),
            }
            cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            # Learning cache is best-effort and must never affect answer serving.
            return

    @staticmethod
    def _build_rag_context_preview(rag_result) -> str:
        snippets: list[str] = []
        for item in getattr(rag_result, "supporting_evidence", [])[:3]:
            source = getattr(item, "source", "unknown_source")
            page = getattr(item, "page", "n/a")
            text = str(getattr(item, "text", "")).strip()
            if text:
                snippets.append(f"[{source}#page={page}] {text}")
        return "\n\n".join(snippets)

    @staticmethod
    def _narrow_cached_json_answer(answer: str, question: str, drug_name: str | None) -> str:
        """Narrow cached JSON text for explicit subproduct requests."""

        if not answer or not drug_name:
            return answer

        text = str(answer)
        # Handle bilingual cache format: "EN: ... | FR: ..." or multiline variants.
        en_match = re.search(r"EN:\s*(.*?)(?:\s*\|\s*FR:|\n\s*FR:|$)", text, flags=re.IGNORECASE | re.DOTALL)
        fr_match = re.search(r"FR:\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)

        if en_match or fr_match:
            parts: list[str] = []
            if en_match:
                en_text = en_match.group(1).strip()
                en_text = JSONRetriever._extract_segment_for_subproduct(en_text, drug_name, question=question)
                parts.append(f"EN: {en_text}")
            if fr_match:
                fr_text = fr_match.group(1).strip()
                fr_text = JSONRetriever._extract_segment_for_subproduct(fr_text, drug_name, question=question)
                parts.append(f"FR: {fr_text}")
            return "\n".join(parts)

        return JSONRetriever._extract_segment_for_subproduct(text, drug_name, question=question)

    @staticmethod
    def _cached_answer_matches_qualifier(answer: str, question: str, drug_name: str | None) -> bool:
        """Validate that cached text still includes explicit qualifier requested by the user."""

        if not drug_name:
            return True
        qualifiers = JSONRetriever._extract_subproduct_qualifiers(question, drug_name)
        if not qualifiers:
            return True

        normalized_answer = " ".join(str(answer).lower().split())
        for qualifier in qualifiers:
            variants = JSONRetriever._qualifier_variants(qualifier)
            if not any(variant in normalized_answer for variant in variants):
                return False
        return True

    @staticmethod
    def _build_rag_grounding_context(rag_result) -> str:
        """Build LLM grounding context from RAG outputs without relying on free-form memory."""

        sections: list[str] = []
        answer = str(getattr(rag_result, "answer", "")).strip()
        if answer:
            sections.append(f"RAG_ANSWER_DRAFT:\n{answer}")

        citations = list(getattr(rag_result, "citations", []))
        if citations:
            sections.append("RAG_CITATIONS:\n" + "\n".join(citations))

        evidence = HybridMedicalController._build_rag_context_preview(rag_result)
        if evidence:
            sections.append(f"RAG_EVIDENCE_SNIPPETS:\n{evidence}")

        uncertainty = str(getattr(rag_result, "uncertainty", "") or "").strip()
        if uncertainty:
            sections.append(f"RAG_UNCERTAINTY:\n{uncertainty}")

        conflict_notes = list(getattr(rag_result, "conflict_notes", []))
        if conflict_notes:
            sections.append("RAG_CONFLICT_NOTES:\n" + "\n".join(str(note) for note in conflict_notes))

        if not sections:
            return "Information not available in the provided data"
        return "\n\n".join(sections)

    def _get_last_mentioned_drug(self, messages: list[dict[str, str]]) -> str | None:
        """Extract most recent drug from user turns only to avoid assistant-topic contamination."""
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "").lower()
            for drug in self.json_retriever.drug_names:
                if drug.lower() in content:
                    return drug
        return None

    @staticmethod
    def _recent_turn_window(messages: list[dict[str, str]], max_turns: int = 5) -> list[dict[str, str]]:
        """Keep only the most recent conversation turns, preserving the newest user/assistant exchanges."""

        if max_turns <= 0 or not messages:
            return []

        user_count = 0
        start_index = 0
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") != "user":
                continue
            user_count += 1
            start_index = index
            if user_count >= max_turns:
                break

        return messages[start_index:]

    @staticmethod
    def _normalize_for_domain(text: str) -> str:
        return " ".join(text.lower().split())

    def detect(self, question: str) -> DetectedIntent:
        """Expose intent signals for observability/debugging."""

        return detect_intent(question=question, drug_candidates=self.json_retriever.drug_names)

    def _build_grounded_answer(
        self,
        question: str,
        context: str,
        language: str = "en",
        messages: list[dict[str, str]] | None = None,
    ) -> str:
        response_instruction = "Respond in English."
        if language == "fr":
            response_instruction = "Reponds en francais."

        user_facts = self.persistent_memory.get_user_facts()
        summary_structured = self.persistent_memory.get_summary_memory_structured()
        fact_memory = json.dumps(user_facts, ensure_ascii=False)
        conversation_summary = json.dumps(summary_structured, ensure_ascii=False)

        generated = self.llm.generate(
            context=context,
            user_input=f"{question}\n\n{response_instruction}",
            system_prompt=COMMERCIAL_DOCTOR_SYSTEM_PROMPT,
            conversation_history=messages,
            fact_memory=fact_memory,
            conversation_summary=conversation_summary,
        )
        return generated.answer

    def _build_training_answer(self, question: str, context: str, messages: list[dict[str, str]] | None = None) -> str:
        user_facts = self.persistent_memory.get_user_facts()
        summary_structured = self.persistent_memory.get_summary_memory_structured()
        fact_memory = json.dumps(user_facts, ensure_ascii=False)
        conversation_summary = json.dumps(summary_structured, ensure_ascii=False)

        generated = self.llm.generate(
            context=context,
            user_input=question,
            system_prompt=TRAINING_DOCTOR_SYSTEM_PROMPT,
            conversation_history=messages,
            fact_memory=fact_memory,
            conversation_summary=conversation_summary,
        )
        return generated.answer

    @staticmethod
    def _infer_user_role(question: str, previous_role: str | None = None) -> str:
        lowered = " ".join(question.lower().split())
        if any(token in lowered for token in ("i am a medical rep", "medical representative", "pharma rep", "delegue medical", "visiteur medical")):
            return "rep"
        if any(token in lowered for token in ("i am a doctor", "i'm a doctor", "physician", "medecin", "clinician")):
            return "doctor"
        if previous_role in {"rep", "doctor"}:
            return previous_role
        return "unknown"

    @staticmethod
    def _training_off_topic_redirect(language: str) -> str:
        if language == "fr":
            return "Je peux aider pour le detailing medical et les discussions cliniques produits. Recentrons-nous sur une indication, la posologie, la securite ou l'usage d'un produit."
        return "I can help with medical detailing and clinical product discussions. Let us refocus on indications, dosage, safety, or product use."

    def _classify_intent_with_llm(self, question: str, messages: list[dict[str, str]] | None = None) -> str:
        """LLM-first intent routing with deterministic fallback heuristics."""

        # Social openings must be handled deterministically to avoid stale cache/refusal loops.
        if self._is_greeting(question) or self._is_small_talk(question):
            return "greeting"

        try:
            classification = self.llm.classify_intent(user_input=question, conversation_history=messages)
            if classification.label in LLM_ROUTER_LABELS:
                return classification.label
        except Exception:
            # Fallback to deterministic heuristics if the classifier call fails.
            pass

        lowered = self._normalize(question)
        if self._is_greeting(question) or self._is_small_talk(question):
            return "greeting"
        if any(token in lowered for token in ("compare", "recommend", "best", "which", "choisir", "recommander")):
            return "product_discussion"
        if any(token in lowered for token in PHARMA_DOMAIN_TERMS):
            return "medical"
        return "off_topic"

    def _build_greeting_answer(self, question: str, language: str, messages: list[dict[str, str]] | None = None) -> str:
        """Generate a normal greeting reply while keeping pharma assistant positioning."""

        if language == "fr":
            context = (
                "Tu reponds a une salutation. Reponds naturellement, de facon concise et professionnelle. "
                "Rappelle que tu peux aider pour des questions medicales/pharmaceutiques sur des produits."
            )
        else:
            context = (
                "You are answering a greeting. Reply naturally, concise and professional. "
                "Mention that you can help with product-specific medical/pharmaceutical questions."
            )

        return self._build_grounded_answer(
            question=question,
            context=context,
            language=language,
            messages=messages,
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().split())

    def _is_greeting(self, question: str) -> bool:
        lowered = self._normalize(question)
        return any(token in lowered for token in GREETING_TOKENS)

    def _is_small_talk(self, question: str) -> bool:
        lowered = self._normalize(question)
        return any(token in lowered for token in SMALL_TALK_TOKENS) or self._is_greeting(question)

    def _detect_language(self, question: str, messages: list[dict[str, str]] | None = None) -> str:
        """Detect French vs English from the current question (fallback to history)."""
        lowered = self._normalize(question)

        fr_score = sum(1 for token in FRENCH_TOKENS if token in lowered)
        en_score = sum(1 for token in ENGLISH_TOKENS if token in lowered)

        # Accented characters are a strong French indicator.
        if re.search(r"[àâçéèêëîïôûùüÿœæ]", lowered):
            fr_score += 2

        if fr_score > en_score:
            return "fr"
        if en_score > fr_score:
            return "en"

        if messages:
            for message in reversed(messages):
                if message.get("role") != "user":
                    continue
                content = self._normalize(message.get("content", ""))
                fr_hist = sum(1 for token in FRENCH_TOKENS if token in content)
                en_hist = sum(1 for token in ENGLISH_TOKENS if token in content)
                if fr_hist > en_hist:
                    return "fr"
                if en_hist > fr_hist:
                    return "en"

        return "en"

    def _focus_pharma_question(self, question: str, resolved_drug: str | None) -> str:
        """Keep only pharma-relevant subparts for mixed-domain queries."""

        lowered = self._normalize_for_domain(question)
        has_non_pharma = any(token in lowered for token in NON_PHARMA_TOPICS)
        if not has_non_pharma:
            return question

        # Split on simple conjunctions and keep the medically relevant clauses.
        segments = re.split(r"\b(?:and|et|also|plus)\b|[,;]", question)
        kept: list[str] = []
        for segment in segments:
            normalized = self._normalize_for_domain(segment)
            if not normalized:
                continue
            if any(token in normalized for token in NON_PHARMA_TOPICS):
                continue
            if resolved_drug and resolved_drug.lower() in normalized:
                kept.append(segment.strip())
                continue
            if any(token in normalized for token in PHARMA_DOMAIN_TERMS):
                kept.append(segment.strip())
                continue
            if any(topic in normalized for topic in CRITICAL_TOPICS):
                kept.append(segment.strip())

        if kept:
            return ". ".join(part for part in kept if part)
        return question

    def _detect_product_switch(self, question: str, detected_drug: str | None) -> str | None:
        """Detect explicit user intent to switch product context."""

        lowered = self._normalize_for_domain(question)
        if not any(token in lowered for token in PRODUCT_SWITCH_TOKENS):
            return None
        return detected_drug

    def _has_product_switch_signal(self, question: str) -> bool:
        lowered = self._normalize_for_domain(question)
        return any(token in lowered for token in PRODUCT_SWITCH_TOKENS)

    @staticmethod
    def _reset_product_state(conversation_state: dict | None) -> None:
        if not isinstance(conversation_state, dict):
            return
        conversation_state["active_product"] = None
        conversation_state["current_product"] = None
        conversation_state["product_locked"] = False
        conversation_state["conversation_phase"] = "onboarding"
        conversation_state["last_topic"] = None

    @staticmethod
    def _format_training_rag_answer(answer: str, language: str) -> str:
        if language == "fr":
            return f"Voici la reponse basee sur les preuves disponibles:\n{answer}"
        return f"Here is the evidence-based answer:\n{answer}"

    @staticmethod
    def _training_rag_follow_up(language: str) -> str:
        if language == "fr":
            return "Comment presenteriez-vous ce point de facon concise a un medecin ?"
        return "How would you present this point concisely to a physician?"

    def _extract_conversation_context(self, messages: list[dict[str, str]]) -> dict:
        """Extract conversation state from message history.
        
        Returns: {
            'drugs_mentioned': set,
            'topics_covered': set,
            'context_established': bool,
            'turn_count': int
        }
        """
        drugs_mentioned: set[str] = set()
        topics_covered: set[str] = set()
        context_established = False
        user_turn_count = 0
        
        for msg in messages:
            content = msg.get("content", "").lower()
            role = msg.get("role", "")
            
            # Track mentioned drugs
            for drug in self.json_retriever.drug_names:
                if drug.lower() in content:
                    drugs_mentioned.add(drug.lower())
            
            # Track covered topics
            for topic in CRITICAL_TOPICS:
                if topic.lower() in content:
                    topics_covered.add(topic)
            
            # Context markers
            if any(word in content for word in ["pharmacy", "clinic", "hospital", "gp", "setting", "physician", "patient"]):
                context_established = True
            
            # Count user turns
            if role == "user":
                user_turn_count += 1
        
        return {
            'drugs_mentioned': drugs_mentioned,
            'topics_covered': topics_covered,
            'context_established': context_established,
            'turn_count': user_turn_count
        }

    def _infer_user_intent(self, question: str) -> str:
        lowered = self._normalize(question)
        if any(token in lowered for token in ["new product", "present", "presentation", "introduce", "going to present"]):
            return "introduce_product"
        if any(token in lowered for token in ["better for elderly", "old people", "elderly", "older patients"]):
            return "elderly_positioning"
        if any(token in lowered for token in ["safety", "tolerability", "side effect", "adverse", "risk"]):
            return "safety_probe"
        if any(token in lowered for token in ["dosage", "dose", "how much", "posology"]):
            return "dosage_probe"
        if any(token in lowered for token in ["composition", "ingredient", "made of"]):
            return "composition_probe"
        return "general_follow_up"

    def _build_training_state(
        self,
        *,
        messages: list[dict[str, str]],
        question: str,
        resolved_drug: str | None,
        current_intent: str,
        previous_state: dict[str, str] | None = None,
        last_topic: str | None = None,
    ) -> TrainingConversationState:
        context = self._extract_conversation_context(messages)
        current_product = resolved_drug or (previous_state or {}).get("current_product")
        if current_product is None:
            current_product = self._get_last_mentioned_drug(messages)

        product_locked = bool(current_product)
        conversation_phase = "onboarding"
        if product_locked:
            conversation_phase = "clinical_exploration"
            if context["turn_count"] >= 3:
                conversation_phase = "deepening"
            if last_topic in {"mechanism_of_action", "warnings", "dosage"}:
                conversation_phase = "advanced_probe"

        user_facts = self.persistent_memory.get_user_facts()
        summary_structured = self.persistent_memory.get_summary_memory_structured()
        recent_persistent_messages = self.persistent_memory.get_recent_messages()
        user_role = self._infer_user_role(question, (previous_state or {}).get("user_role"))

        return TrainingConversationState(
            current_product=current_product,
            product_locked=product_locked,
            conversation_phase=conversation_phase,
            last_topic=last_topic or (previous_state or {}).get("last_topic"),
            last_user_intent=self._infer_user_intent(question),
            current_intent=current_intent,
            user_role=user_role,
            memory_context={
                "user_facts": user_facts,
                "summary_memory": summary_structured,
                "recent_memory": recent_persistent_messages[-8:],
            },
            turn_count=context["turn_count"],
        )
    
    def _get_last_mentioned_topic(self, messages: list[dict[str, str]]) -> str | None:
        """Extract the most recently discussed topic from conversation history."""
        for msg in reversed(messages):
            content = msg.get("content", "").lower()
            for topic in CRITICAL_TOPICS:
                if topic in content:
                    return topic
        return None

    def _estimate_training_level(
        self,
        context: dict,
        question: str,
        resolved_drug: str | None,
        last_topic: str | None,
    ) -> int:
        """Estimate ALIA level from conversation depth and complexity."""
        level = 2
        turn_count = int(context.get("turn_count", 0))
        topic_count = len(context.get("topics_covered", set()))
        lowered = self._normalize(question)

        if turn_count <= 1 and resolved_drug is None:
            level = 1
        elif turn_count >= 4 or topic_count >= 2:
            level = 3

        if any(token in lowered for token in ["mechanism", "contraindication", "interaction", "safety", "dosage", "dose"]):
            level = max(level, 4)

        if last_topic in {"mechanism_of_action", "warnings", "dosage"} and turn_count >= 3:
            level = max(level, 4)

        return max(1, min(4, level))

    def handle_query(
        self,
        question: str,
        mode: str = "commercial",
        user_id: str | None = None,
        preferred_drug_name: str | None = None,
        preferred_language: str | None = None,
        messages: list[dict[str, str]] | None = None,
        conversation_state: dict[str, str] | None = None,
        session_reset: bool = False,
    ) -> AgentResponse:
        """Resolve user question with JSON-priority + RAG fallback policy.
        
        Priority: conversation context > JSON > RAG > training brain rules.
        """
        # Always resolve user memory namespace per request to avoid stale shared state.
        self.persistent_memory = self._memory_store_for_user(user_id)

        if messages is None:
            messages = []

        if session_reset:
            self.reset_session(
                mode=mode,
                user_id=self.active_user_id,
                conversation_state=conversation_state,
                clear_short_term_memory=True,
                keep_user_facts=False,
            )

        recent_messages = self._recent_turn_window(messages, max_turns=5)
        self._capture_user_memory(question)

        if preferred_language in {"fr", "en"}:
            language = preferred_language
        else:
            language = self._detect_language(question, recent_messages)

        personal_query = self._is_personal_memory_query(question)

        conv_context = self._extract_conversation_context(recent_messages)
        intent = self.detect(question)
        resolved_drug = intent.drug_name or preferred_drug_name

        switch_signal = self._has_product_switch_signal(question)
        switch_drug = self._detect_product_switch(question, intent.drug_name)
        explicit_switch = switch_drug is not None
        if switch_signal:
            self._reset_product_state(conversation_state)
        if explicit_switch:
            resolved_drug = switch_drug

        if resolved_drug is None and not switch_signal:
            resolved_drug = self._get_last_mentioned_drug(recent_messages)

        intent_label = self._classify_intent_with_llm(question, recent_messages)
        runtime_mode = "training" if mode == "training" else "commercial"
        self._runtime_sessions[runtime_mode]["current_intent"] = intent_label
        if intent_label in {"medical", "product_discussion"}:
            self._runtime_sessions[runtime_mode]["active_product"] = resolved_drug
        else:
            self._runtime_sessions[runtime_mode]["active_product"] = None

        personal_answer = self._answer_personal_memory(question, language)
        if personal_answer is not None:
            return self._finalize_response(
                question=question,
                mode=mode,
                response=AgentResponse(
                    answer=personal_answer,
                    source="memory",
                    context="",
                    topic=None,
                    drug_name=None,
                    confidence=1.0,
                ),
            )

        if intent_label == "greeting":
            greeting_answer = self._build_greeting_answer(
                question=question,
                language=language,
                messages=recent_messages,
            )
            return self._finalize_response(
                question=question,
                mode=mode,
                response=AgentResponse(
                    answer=greeting_answer,
                    source="assistant",
                    context="",
                    topic=intent.topic,
                    drug_name=resolved_drug,
                    confidence=1.0,
                ),
            )

        is_social_opening = self._is_greeting(question) or self._is_small_talk(question)
        skip_cache = session_reset or switch_signal or explicit_switch or personal_query or is_social_opening or intent_label in {"off_topic", "greeting"}
        can_use_cache = mode != "training" and intent_label in {"medical", "product_discussion"}
        cached = None
        if not skip_cache and can_use_cache:
            cached = self.persistent_memory.find_cached_qa(
                question,
                mode=mode,
                allowed_sources={"json", "rag", "json_cache"},
            )
        if cached is not None:
            cached_answer = str(cached.get("output", ""))
            if intent.topic in CRITICAL_TOPICS and resolved_drug:
                cached_answer = self._narrow_cached_json_answer(
                    cached_answer,
                    question=question,
                    drug_name=resolved_drug,
                )
                if not self._cached_answer_matches_qualifier(cached_answer, question, resolved_drug):
                    cached = None

        if cached is not None:
            return self._finalize_response(
                question=question,
                mode=mode,
                response=AgentResponse(
                    answer=cached_answer,
                    source="json_cache",
                    context="",
                    topic=intent.topic,
                    drug_name=resolved_drug,
                    confidence=1.0,
                ),
            )

        if intent_label == "off_topic" and mode != "training":
            return self._finalize_response(
                question=question,
                mode=mode,
                response=AgentResponse(
                    answer=OUT_OF_DOMAIN_MESSAGE,
                    source="assistant",
                    context="",
                    topic=None,
                    drug_name=resolved_drug,
                    confidence=1.0,
                ),
            )

        if intent_label == "off_topic" and mode == "training":
            return self._finalize_response(
                question=question,
                mode=mode,
                response=AgentResponse(
                    answer=self._training_off_topic_redirect(language),
                    source="assistant",
                    context="",
                    topic=None,
                    drug_name=resolved_drug,
                    confidence=1.0,
                ),
            )

        focused_question = self._focus_pharma_question(question, resolved_drug)


        if mode == "training":
            last_topic = intent.topic or self._get_last_mentioned_topic(recent_messages)
            training_state = self._build_training_state(
                messages=recent_messages,
                question=question,
                resolved_drug=resolved_drug,
                current_intent="identity" if personal_query else intent_label,
                previous_state=conversation_state,
                last_topic=last_topic,
            )
            alia_level = self._estimate_training_level(
                conv_context,
                question=question,
                resolved_drug=resolved_drug,
                last_topic=last_topic,
            )

            knowledge_source = "assistant"
            knowledge_context = ""
            confidence = 0.85

            json_result = self.json_retriever.find(drug_name=resolved_drug, topic=intent.topic, question=question)
            if json_result is None and resolved_drug is not None and intent.topic is None:
                payload = self.json_retriever.get_payload(resolved_drug)
                if payload is not None:
                    json_result = type("JsonFallback", (), {})()
                    json_result.source = "json"
                    json_result.content = self.json_retriever.build_full_context(payload, drug_name=resolved_drug)
                    json_result.confidence = 1.0
                    json_result.drug_name = payload.get("drug_name") or resolved_drug

            if json_result is not None:
                knowledge_source = "json"
                knowledge_context = json_result.content
                confidence = float(getattr(json_result, "confidence", 1.0))
            else:
                rag_result = self.rag_adapter.query(focused_question, response_language=language)
                if rag_result is not None and rag_result.answer.strip():
                    self._store_high_confidence_rag(focused_question, rag_result)
                    rag_context = self._build_rag_grounding_context(rag_result)
                    rag_answer = self._build_training_answer(
                        question=focused_question,
                        context=rag_context,
                        messages=recent_messages,
                    )
                    return self._finalize_response(
                        question=question,
                        mode=mode,
                        response=AgentResponse(
                            answer=self._format_training_rag_answer(rag_answer, language),
                            source="rag",
                            context=self._build_rag_context_preview(rag_result),
                            topic=intent.topic,
                            drug_name=resolved_drug,
                            confidence=float(rag_result.answer_confidence),
                            follow_up=self._training_rag_follow_up(language),
                            citations=list(rag_result.citations),
                            uncertainty=rag_result.uncertainty or None,
                            conflict_notes=list(rag_result.conflict_notes),
                        ),
                    )

            decision = self.training_brain.decide_next_action(
                conversation_history=recent_messages,
                conversation_state={
                    "current_product": training_state.current_product,
                    "product_locked": training_state.product_locked,
                    "conversation_phase": training_state.conversation_phase,
                    "last_topic": training_state.last_topic,
                    "last_user_intent": training_state.last_user_intent,
                    "current_intent": training_state.current_intent,
                    "active_product": training_state.current_product,
                    "user_role": training_state.user_role,
                    "turn_count": training_state.turn_count,
                },
                memory_context=training_state.memory_context,
                detected_product=resolved_drug,
                alia_level=alia_level,
                last_topic=last_topic,
                knowledge_context=knowledge_context,
                user_message=focused_question,
                forced_language=language,
            )

            if training_state.product_locked and resolved_drug is None:
                resolved_drug = training_state.current_product

            return self._finalize_response(
                question=question,
                mode=mode,
                response=AgentResponse(
                    answer=decision.message,
                    source=knowledge_source,
                    context=knowledge_context,
                    topic=decision.topic or last_topic or intent.topic,
                    drug_name=resolved_drug,
                    confidence=confidence if knowledge_context else 0.85,
                    follow_up=None,
                ),
            )

        # COMMERCIAL MODE: Critical topic without product - ask for clarification ONLY on first turn
        if (mode == "commercial" and intent.topic in CRITICAL_TOPICS and resolved_drug is None 
            and self.json_retriever.has_multiple_drugs and conv_context['turn_count'] <= 1):
            examples = ", ".join(self.json_retriever.sample_drugs[:5])
            clarify_message = CLARIFY_PRODUCT_MESSAGE
            example_label = "Example products"
            if language == "fr":
                clarify_message = CLARIFY_PRODUCT_MESSAGE_FR
                example_label = "Exemples de produits"
            return self._finalize_response(
                question=question,
                mode=mode,
                response=AgentResponse(
                    answer=f"{clarify_message} {example_label}: {examples}",
                    source="assistant",
                    context="",
                    topic=intent.topic,
                    drug_name=None,
                    confidence=1.0,
                ),
            )

        # Try to get knowledge from JSON first
        json_result = self.json_retriever.find(drug_name=resolved_drug, topic=intent.topic, question=question)
        if json_result is None and resolved_drug is not None and intent.topic is None:
            payload = self.json_retriever.get_payload(resolved_drug)
            if payload is not None:
                json_result = type("JsonFallback", (), {})()
                json_result.source = "json"
                json_result.content = self.json_retriever.build_full_context(payload, drug_name=resolved_drug)
                json_result.confidence = 1.0
                json_result.drug_name = payload.get("drug_name") or resolved_drug

        if json_result is not None:
            return self._finalize_response(
                question=question,
                mode=mode,
                response=AgentResponse(
                    answer=json_result.content,
                    source="json",
                    context=json_result.content,
                    topic=intent.topic,
                    drug_name=json_result.drug_name or resolved_drug,
                    confidence=json_result.confidence,
                ),
            )

        # Fall back to RAG
        rag_result = self.rag_adapter.query(focused_question, response_language=language)
        if rag_result is not None and rag_result.answer.strip():
            self._store_high_confidence_rag(focused_question, rag_result)
            return self._finalize_response(
                question=question,
                mode=mode,
                response=AgentResponse(
                    answer=rag_result.answer,
                    source="rag",
                    context=self._build_rag_context_preview(rag_result),
                    topic=intent.topic,
                    drug_name=resolved_drug,
                    confidence=float(rag_result.answer_confidence),
                    citations=list(rag_result.citations),
                    uncertainty=rag_result.uncertainty or None,
                    conflict_notes=list(rag_result.conflict_notes),
                ),
            )

        # No knowledge found - give appropriate fallback
        no_kb_message = NO_KB_MESSAGE_FR if language == "fr" else NO_KB_MESSAGE
        return self._finalize_response(
            question=question,
            mode=mode,
            response=AgentResponse(
                answer=no_kb_message,
                source="none",
                context="",
                topic=intent.topic,
                drug_name=resolved_drug,
                confidence=0.0,
            ),
        )

    def start_training_turn(
        self,
        drug_name: str | None = None,
        topic: str | None = None,
        user_id: str | None = None,
    ) -> tuple[str, str, str]:
        """Generate a dynamic training opener with the training brain."""

        self.persistent_memory = self._memory_store_for_user(user_id)

        self.json_retriever.refresh()
        selected_drug = drug_name or (self.json_retriever.drug_names[0] if self.json_retriever.drug_names else None)
        knowledge_context = ""

        if selected_drug:
            payload = self.json_retriever.get_payload(selected_drug)
            if payload is not None:
                knowledge_context = self.json_retriever.build_full_context(payload, drug_name=selected_drug)

        decision = self.training_brain.decide_next_action(
            conversation_history=[],
            conversation_state=None,
            memory_context=None,
            detected_product=selected_drug,
            alia_level=2,
            last_topic=topic,
            knowledge_context=knowledge_context,
            user_message="Start a new training conversation.",
        )
        return decision.message, "", decision.action

    def evaluate_training_answer(
        self,
        doctor_question: str,
        rep_answer: str,
        user_id: str | None = None,
    ) -> TrainingTurn:
        """Evaluate rep answer against context-grounded expected answer."""

        expected = self.handle_query(doctor_question, mode="training", user_id=user_id)
        if expected.source == "none":
            feedback = "No knowledge-base evidence found; training item cannot be scored."
        elif rep_answer.strip().lower() in expected.answer.strip().lower():
            feedback = "Good response. Your answer is consistent with the available medical context."
        else:
            feedback = (
                "Partial/incorrect coverage. Focus on the exact context-backed facts only and avoid assumptions."
            )

        return TrainingTurn(
            doctor_question=doctor_question,
            expected_answer=expected.answer,
            rep_answer=rep_answer,
            feedback=feedback,
            source=expected.source,
        )
