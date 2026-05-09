"""Dual-mode Streamlit UI for the Hybrid Medical Agent."""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import streamlit as st

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from hybrid_medical_agent.agent.controller import HybridMedicalController

TRAINING_MODE = "Training Mode"
COMMERCIAL_MODE = "Commercial Mode"
MODE_OPTIONS = [TRAINING_MODE, COMMERCIAL_MODE]
LANGUAGE_OPTIONS = {
    "Auto-detect": "auto",
    "English": "en",
    "Francais": "fr",
}


@st.cache_resource
def get_controller(llm_model: str, llm_base_url: str) -> HybridMedicalController:
    """Create one controller instance per UI configuration."""

    return HybridMedicalController(
        workspace_root=WORKSPACE_ROOT,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
    )


def ensure_history(mode: str) -> list[dict[str, str]]:
    """Create a mode-specific conversation history if needed."""

    history_key = f"messages_{mode}"
    if history_key not in st.session_state:
        if mode == TRAINING_MODE:
            opening = "Doctor online. Happy to discuss your products and help you refine your clinical detailing approach."
        else:
            opening = "Doctor online. Ask a product-specific medical question and I will answer precisely."

        st.session_state[history_key] = [
            {
                "role": "assistant",
                "content": opening,
                "source": "system",
            }
        ]
    return st.session_state[history_key]


def ensure_conversation_state(mode: str) -> dict:
    """Create mode-specific conversation tracking state."""
    state_key = f"conv_state_{mode}"
    if state_key not in st.session_state:
        st.session_state[state_key] = {
            "current_intent": "greeting",
            "active_product": None,
            "user_role": "unknown",
            "current_product": None,
            "product_locked": False,
            "conversation_phase": "onboarding",
            "drugs_mentioned": set(),
            "topics_covered": set(),
            "context_established": False,
            "last_topic": None,
            "last_user_intent": None,
            "memory_context": {
                "user_facts": {},
                "summary_memory": {},
            },
            "turn_count": 0,
        }
    return st.session_state[state_key]


def ensure_session_memory(mode: str) -> dict:
    """Create mode-specific short-term session memory used by the UI."""

    key = f"session_memory_{mode}"
    if key not in st.session_state:
        st.session_state[key] = {
            "user_facts": {
                "name": "",
                "company": "",
                "role": "",
                "preferences": [],
            }
        }
    return st.session_state[key]


def _extract_name_from_message(message: str) -> str | None:
    patterns = [
        r"\b(?:my name is|call me)\s+([A-Za-z][A-Za-z\-']{1,63})\b",
        r"\b(?:je m'appelle)\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\-']{1,63})\b",
        r"\b(?:i am|i'm|im)\s+([A-Za-z][A-Za-z\-']{1,63})(?=\s*(?:[.,!?;:]|$|\band\b))",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _extract_company_from_message(message: str) -> str | None:
    patterns = [
        r"\b(?:i am from|i'm from|im from|i work at|i work for|i am with|i'm with|im with)\s+([A-Za-z0-9][A-Za-z0-9&\-'.\s]{1,80})\b",
        r"\b(?:je viens de|je suis de|je travaille chez|je travaille pour|je suis avec)\s+([A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9&\-'.\s]{1,80})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            candidate = re.sub(r"\s+", " ", match.group(1)).strip(" .,:;!?")
            candidate = re.split(r"[.!?]", candidate, maxsplit=1)[0].strip(" .,:;!?")
            candidate = re.split(
                r"\b(?:i am|i'm|im|je suis|i prefer|my preference|je prefere)\b",
                candidate,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip(" .,:;!?")
            return candidate
    return None


def _extract_role_from_message(message: str) -> str | None:
    lowered = " ".join(message.lower().split())
    if any(token in lowered for token in ("medical representative", "medical rep", "pharma rep", "delegue medical", "visiteur medical")):
        return "rep"
    if any(token in lowered for token in ("i am a doctor", "i'm a doctor", "physician", "medecin", "clinician")):
        return "doctor"
    return None


def _extract_preference_from_message(message: str) -> str | None:
    normalized = " ".join(message.strip().split())
    if any(token in normalized.lower() for token in ("i prefer", "my preference", "prefer ", "je prefere")):
        return normalized
    return None


def sync_session_profile_to_persistent_memory(controller: HybridMedicalController, user_id: str, user_message: str, session_memory: dict) -> None:
    """Persist profile facts from Streamlit session memory immediately."""

    extracted_name = _extract_name_from_message(user_message)
    extracted_company = _extract_company_from_message(user_message)
    extracted_role = _extract_role_from_message(user_message)
    extracted_preference = _extract_preference_from_message(user_message)

    if not any([extracted_name, extracted_company, extracted_role, extracted_preference]):
        return

    session_facts = session_memory.setdefault("user_facts", {})

    # Make sure we are writing to the active authenticated user namespace.
    controller.persistent_memory = controller._memory_store_for_user(user_id)
    current_facts = controller.persistent_memory.get_user_facts()

    if extracted_name:
        session_facts["name"] = extracted_name
        current_name = str(current_facts.get("name", "")).strip()
        if current_name.lower() != extracted_name.lower():
            controller.persistent_memory.add_user_fact(
                content=f"User name is {extracted_name}",
                key="name",
                value=extracted_name,
            )

    if extracted_company:
        session_facts["company"] = extracted_company
        current_company = str(current_facts.get("company", "")).strip()
        if current_company.lower() != extracted_company.lower():
            controller.persistent_memory.add_user_fact(
                content=f"User company is {extracted_company}",
                key="company",
                value=extracted_company,
            )

    if extracted_role:
        session_facts["role"] = extracted_role
        current_role = str(current_facts.get("role", "")).strip()
        if current_role.lower() != extracted_role.lower():
            controller.persistent_memory.add_user_fact(
                content=f"User role is {extracted_role}",
                key="role",
                value=extracted_role,
            )

    if extracted_preference:
        preferences = session_facts.setdefault("preferences", [])
        if extracted_preference not in preferences:
            preferences.append(extracted_preference)

        current_preferences = current_facts.get("preferences", [])
        normalized_current = {
            str(pref).strip().lower()
            for pref in (current_preferences if isinstance(current_preferences, list) else [])
            if str(pref).strip()
        }
        if extracted_preference.strip().lower() not in normalized_current:
            controller.persistent_memory.add_preference(extracted_preference)


def render_messages(messages: list[dict[str, str]]) -> None:
    """Render conversation history in chronological order."""

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            source = msg.get("source")
            if source:
                st.caption(f"source: {source}")


def main() -> None:
    st.set_page_config(page_title="Hybrid Medical Agent", page_icon="🩺", layout="centered")
    st.title("Hybrid Medical Agent")
    st.caption("JSON first, RAG fallback, doctor-controlled responses for training and commercial use.")

    with st.sidebar:
        st.subheader("Settings")
        mode = st.selectbox("Mode", MODE_OPTIONS, index=0)
        selected_language_label = st.selectbox("Response language", options=list(LANGUAGE_OPTIONS.keys()), index=0)
        selected_language = LANGUAGE_OPTIONS[selected_language_label]
        llm_model = st.text_input("LLM Model", value="llama3:8b")
        llm_base_url = st.text_input("LLM Base URL", value="http://localhost:11434")
        user_id = st.text_input("Authenticated User ID", value="demo_user", help="Memory namespace key for the current authenticated user.")
        controller = get_controller(llm_model=llm_model, llm_base_url=llm_base_url)
        active_memory_paths = controller.user_memory_namespace.resolve(user_id)
        st.caption(f"Active conversation memory file: {active_memory_paths.conversation_path.relative_to(WORKSPACE_ROOT)}")

        available_drugs = controller.json_retriever.drug_names
        if available_drugs:
            selected_drug = st.selectbox(
                "Drug / product",
                options=["(auto-detect)"] + available_drugs,
                index=0,
                help="Optional product scope for direct medical questions.",
            )
        else:
            selected_drug = "(auto-detect)"
            st.info("No products found in the JSON knowledge base yet.")

        if st.button("Reset current mode conversation"):
            if hasattr(controller, "reset_session"):
                reset_kwargs = {
                    "mode": "training" if mode == TRAINING_MODE else "commercial",
                    "keep_user_facts": False,
                }
                if "user_id" in inspect.signature(controller.reset_session).parameters:
                    reset_kwargs["user_id"] = user_id
                controller.reset_session(**reset_kwargs)
            st.session_state[f"messages_{mode}"] = []
            st.session_state[f"conv_state_{mode}"] = {
                "current_intent": "greeting",
                "active_product": None,
                "user_role": "unknown",
                "current_product": None,
                "product_locked": False,
                "conversation_phase": "onboarding",
                "drugs_mentioned": set(),
                "topics_covered": set(),
                "context_established": False,
                "last_topic": None,
                "last_user_intent": None,
                "memory_context": {
                    "user_facts": {},
                    "summary_memory": {},
                },
                "turn_count": 0,
            }
            st.session_state[f"session_reset_{mode}"] = True
            st.rerun()

    history = ensure_history(mode)
    conversation_state = ensure_conversation_state(mode)
    session_memory = ensure_session_memory(mode)
    reset_flag_key = f"session_reset_{mode}"
    session_reset_flag = bool(st.session_state.get(reset_flag_key, False))
    render_messages(history)

    if mode == TRAINING_MODE:
        prompt_text = "Type your message as a medical representative..."
    else:
        prompt_text = "Ask about a product, composition, dosage, indications, or safety..."

    user_message = st.chat_input(prompt_text)
    if not user_message:
        return

    history.append({"role": "user", "content": user_message})

    # Streamlit-session bridge: persist profile facts as soon as they are provided.
    sync_session_profile_to_persistent_memory(controller, user_id, user_message, session_memory)

    preferred_drug = None if selected_drug == "(auto-detect)" else selected_drug

    with st.spinner("Doctor is reviewing the available knowledge..."):
        query_kwargs = {
            "mode": "training" if mode == TRAINING_MODE else "commercial",
            "preferred_drug_name": preferred_drug,
            "messages": history,  # PASS FULL CONVERSATION HISTORY to avoid repetitive loops
            "conversation_state": conversation_state if mode == TRAINING_MODE else None,
        }
        if "user_id" in inspect.signature(controller.handle_query).parameters:
            query_kwargs["user_id"] = user_id
        if "preferred_language" in inspect.signature(controller.handle_query).parameters:
            query_kwargs["preferred_language"] = selected_language
        if "session_reset" in inspect.signature(controller.handle_query).parameters:
            query_kwargs["session_reset"] = session_reset_flag

        result = controller.handle_query(user_message, **query_kwargs)
        st.session_state[reset_flag_key] = False

    # Keep UI session memory synchronized with persisted user facts.
    latest_user_facts = controller.persistent_memory.get_user_facts()
    latest_summary_memory = controller.persistent_memory.get_summary_memory_structured()
    session_memory["user_facts"] = latest_user_facts
    conversation_state.setdefault("memory_context", {})["user_facts"] = latest_user_facts
    conversation_state.setdefault("memory_context", {})["summary_memory"] = latest_summary_memory

    history.append(
        {
            "role": "assistant",
            "content": result.answer,
            "source": result.source,
        }
    )

    if mode == TRAINING_MODE and result.follow_up:
        history.append(
            {
                "role": "assistant",
                "content": result.follow_up,
                "source": "training",
            }
        )

    if mode == TRAINING_MODE:
        conversation_state["turn_count"] += 1
        conversation_state["last_topic"] = result.topic or conversation_state.get("last_topic")
        conversation_state["last_user_intent"] = user_message
        lowered = " ".join(user_message.lower().split())
        if result.source == "memory":
            conversation_state["current_intent"] = "identity"
        elif result.source == "assistant" and "specialized only in pharmaceutical" in result.answer.lower():
            conversation_state["current_intent"] = "off_topic"
        elif any(token in lowered for token in ("hello", "hi", "bonjour", "salut")):
            conversation_state["current_intent"] = "greeting"
        elif any(token in lowered for token in ("introduce", "presentation", "pitch", "compare", "recommend")):
            conversation_state["current_intent"] = "product"
        else:
            conversation_state["current_intent"] = "medical"

        if any(token in lowered for token in ("medical representative", "medical rep", "pharma rep", "delegue medical")):
            conversation_state["user_role"] = "rep"
        elif any(token in lowered for token in ("i am a doctor", "physician", "medecin", "clinician")):
            conversation_state["user_role"] = "doctor"

        if result.drug_name:
            conversation_state["current_product"] = result.drug_name
            conversation_state["active_product"] = result.drug_name
            conversation_state["product_locked"] = True
            if conversation_state["conversation_phase"] == "onboarding":
                conversation_state["conversation_phase"] = "clinical_exploration"
        if conversation_state["product_locked"] and conversation_state["conversation_phase"] == "clinical_exploration" and result.topic in {"mechanism_of_action", "warnings", "dosage"}:
            conversation_state["conversation_phase"] = "advanced_probe"

    st.rerun()


if __name__ == "__main__":
    main()
