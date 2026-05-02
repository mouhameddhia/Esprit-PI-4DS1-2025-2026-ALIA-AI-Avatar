"""Persistent JSON memory store for user facts, preferences, statements, instructions, and QA."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import re
import tempfile
from uuid import uuid4


SUMMARY_TOPIC_ALLOWLIST = {
    "indications",
    "composition",
    "dosage",
    "administration",
    "warnings",
    "side_effects",
    "mechanism_of_action",
    "age",
    "safety",
    "patient_profile",
}

INVALID_USER_NAME_VALUES = {
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
}


class PersistentMemoryStore:
    """Persistent memory with separated log and structured rolling conversation state.

    Storage model:
    - log file: append-only event log for QA/facts/instructions/preferences.
    - conversation file: structured rolling memory with summary + recent messages + user facts.
    """

    def __init__(
        self,
        path: Path,
        conversation_path: Path | None = None,
        max_entries: int = 5000,
        max_qa_entries: int = 10,
        summary_trigger_messages: int = 10,
        recent_window_messages: int = 6,
        max_summary_items: int = 30,
    ) -> None:
        self.path = path
        self.conversation_path = conversation_path or path.with_name("conversation_memory.json")
        self.max_entries = max_entries
        self.max_qa_entries = max(1, max_qa_entries)
        self.summary_trigger_messages = max(2, summary_trigger_messages)
        self.recent_window_messages = max(2, min(recent_window_messages, self.summary_trigger_messages))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conversation_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_summary_items = max(5, max_summary_items)

        if not self.path.exists():
            self._write_log({"memory": []})
        if not self.conversation_path.exists():
            self._write_conversation(self._default_conversation_payload())

    @staticmethod
    def _default_summary_payload() -> dict[str, Any]:
        return {
            "user_profile": {
                "name": "",
                "company": "",
                "role": "",
                "preferences": [],
            },
            "medical_topics": [],
            "questions": [],
            "decisions": [],
            "source_message_ids": [],
            "last_updated": "",
        }

    def _default_conversation_payload(self) -> dict[str, Any]:
        return {
            "summary_memory": self._default_summary_payload(),
            "recent_messages": [],
            "user_facts": {
                "name": "",
                "company": "",
                "role": "",
                "preferences": [],
                "competency_level": "",
            },
            "turn_count": 0,
            "last_sync_event_id": "",
        }

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(str(text).strip().lower().split())

    def _read_log(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("memory"), list):
                return data
        except Exception:
            pass
        return {"memory": []}

    def _read_conversation(self) -> dict[str, Any]:
        payload = self._default_conversation_payload()
        try:
            data = json.loads(self.conversation_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return payload

            summary = data.get("summary_memory")
            recent_messages = data.get("recent_messages")
            user_facts = data.get("user_facts")
            turn_count = data.get("turn_count")
            last_sync_event_id = data.get("last_sync_event_id")

            if isinstance(summary, dict):
                merged_summary = payload["summary_memory"]
                for key in merged_summary.keys():
                    if key in summary:
                        merged_summary[key] = summary[key]
                payload["summary_memory"] = merged_summary

            if isinstance(recent_messages, list):
                payload["recent_messages"] = recent_messages

            if isinstance(user_facts, dict):
                merged_facts = payload["user_facts"]
                for key in merged_facts.keys():
                    if key in user_facts:
                        merged_facts[key] = user_facts[key]
                payload["user_facts"] = merged_facts

            if isinstance(turn_count, int) and turn_count >= 0:
                payload["turn_count"] = turn_count

            if isinstance(last_sync_event_id, str):
                payload["last_sync_event_id"] = last_sync_event_id

            return payload
        except Exception:
            return payload

    def _write_log(self, data: dict[str, Any]) -> None:
        self._atomic_write_json(self.path, data)

    def _write_conversation(self, data: dict[str, Any]) -> None:
        self._atomic_write_json(self.conversation_path, data)

    @staticmethod
    def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
        """Atomically write JSON content to disk to reduce race/corruption risk."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=str(path.parent),
            suffix=".tmp",
        ) as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            temp_path = Path(handle.name)
        temp_path.replace(path)

    @staticmethod
    def _entry_signature(entry: dict[str, Any]) -> tuple[str, str, str]:
        kind = str(entry.get("type", ""))
        content = str(entry.get("content", ""))
        qa_input = str(entry.get("input", ""))
        return kind, content, qa_input

    def _append_entry(self, entry: dict[str, Any]) -> str | None:
        data = self._read_log()
        memory = data.setdefault("memory", [])
        if memory:
            if self._entry_signature(memory[-1]) == self._entry_signature(entry):
                return None

        event_id = f"evt_{uuid4().hex}"
        entry["timestamp"] = self._now_iso()
        entry["event_id"] = event_id
        memory.append(entry)

        # Keep only the most recent QA prompts to avoid stale conversational residue.
        qa_count = sum(1 for item in memory if str(item.get("type", "")).strip().lower() == "qa")
        if qa_count > self.max_qa_entries:
            to_remove = qa_count - self.max_qa_entries
            pruned: list[dict[str, Any]] = []
            for item in memory:
                if to_remove > 0 and str(item.get("type", "")).strip().lower() == "qa":
                    to_remove -= 1
                    continue
                pruned.append(item)
            memory = pruned

        if len(memory) > self.max_entries:
            memory = memory[-self.max_entries :]
        data["memory"] = memory
        self._write_log(data)
        return event_id

    def _set_last_sync_event(self, event_id: str | None) -> None:
        if not event_id:
            return
        data = self._read_conversation()
        data["last_sync_event_id"] = event_id
        self._write_conversation(data)

    @staticmethod
    def _extract_role_from_text(text: str) -> str | None:
        lowered = " ".join(str(text).lower().split())
        role_patterns = [
            r"\b(?:i am|i'm|im|my role is)\s+(doctor|physician|pharmacist|cardiologist|dermatologist|pediatrician|nurse|medical representative|medical rep|pharma rep)\b",
            r"\b(?:je suis|mon role est)\s+(medecin|pharmacien|cardiologue|dermatologue|pediatre|infirmier|delegue medical)\b",
        ]
        for pattern in role_patterns:
            match = re.search(pattern, lowered, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def _extract_preference_from_text(text: str) -> str | None:
        lowered = " ".join(str(text).strip().split())
        if any(token in lowered.lower() for token in ("i prefer", "my preference", "je prefere")):
            return lowered
        return None

    def _build_summary_delta(self, messages: list[dict[str, Any]], user_facts: dict[str, Any]) -> dict[str, Any]:
        if not messages:
            return self._default_summary_payload()

        topics: set[str] = set()
        questions: list[str] = []
        decisions: list[str] = []
        source_ids: list[str] = []

        for message in messages:
            role = str(message.get("role", ""))
            content = str(message.get("content", "")).strip()
            topic = message.get("topic")
            source = str(message.get("source", "")).strip().lower()
            citations = message.get("citations", [])
            message_id = str(message.get("id", "")).strip()
            if message_id:
                source_ids.append(message_id)
            if isinstance(topic, str) and topic.strip():
                normalized_topic = topic.strip().lower()
                if normalized_topic in SUMMARY_TOPIC_ALLOWLIST:
                    topics.add(normalized_topic)

            if role == "user" and content:
                if content.endswith("?") or "?" in content:
                    questions.append(content[:320])
            if role == "assistant" and content:
                # Keep only trusted assistant decisions to reduce summary contamination.
                if source == "json" or (source == "rag" and isinstance(citations, list) and len(citations) > 0):
                    decisions.append(content[:320])

        name = user_facts.get("name")
        company = user_facts.get("company")
        role = user_facts.get("role")
        preferences = user_facts.get("preferences", [])
        cleaned_preferences = []
        if isinstance(preferences, list):
            cleaned_preferences = [str(item).strip() for item in preferences if str(item).strip()][:5]

        return {
            "user_profile": {
                "name": str(name or "").strip(),
                "company": str(company or "").strip(),
                "role": str(role or "").strip(),
                "preferences": cleaned_preferences,
            },
            "medical_topics": sorted(topics)[: self.max_summary_items],
            "questions": questions[-self.max_summary_items :],
            "decisions": decisions[-self.max_summary_items :],
            "source_message_ids": source_ids,
            "last_updated": self._now_iso(),
        }

    def _validate_summary_delta(self, delta: dict[str, Any]) -> bool:
        if not isinstance(delta, dict):
            return False
        profile = delta.get("user_profile", {})
        topics = delta.get("medical_topics", [])
        questions = delta.get("questions", [])
        decisions = delta.get("decisions", [])
        source_ids = delta.get("source_message_ids", [])

        if not isinstance(profile, dict):
            return False
        if not isinstance(topics, list) or not isinstance(questions, list) or not isinstance(decisions, list):
            return False
        if not isinstance(source_ids, list) or not source_ids:
            return False

        meaningful_content = bool(topics or questions or decisions)
        if not meaningful_content:
            return False

        # Ensure question/decision fragments are not trivially empty.
        if any(len(str(item).strip()) < 3 for item in questions + decisions if str(item).strip()):
            return False
        return True

    def _merge_summary(self, existing: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
        merged = self._default_summary_payload()
        if isinstance(existing, dict):
            merged.update(existing)

        existing_profile = merged.get("user_profile", {})
        delta_profile = delta.get("user_profile", {})
        if isinstance(existing_profile, dict) and isinstance(delta_profile, dict):
            profile = {
                "name": str(delta_profile.get("name") or existing_profile.get("name") or "").strip(),
                "company": str(delta_profile.get("company") or existing_profile.get("company") or "").strip(),
                "role": str(delta_profile.get("role") or existing_profile.get("role") or "").strip(),
                "preferences": [],
            }
            existing_pref = existing_profile.get("preferences", []) if isinstance(existing_profile.get("preferences", []), list) else []
            delta_pref = delta_profile.get("preferences", []) if isinstance(delta_profile.get("preferences", []), list) else []
            seen_pref: set[str] = set()
            for pref in list(existing_pref) + list(delta_pref):
                candidate = str(pref).strip()
                if not candidate or candidate in seen_pref:
                    continue
                seen_pref.add(candidate)
                profile["preferences"].append(candidate)
            profile["preferences"] = profile["preferences"][-self.max_summary_items :]
            merged["user_profile"] = profile

        def _merge_unique_strings(existing_items: Any, delta_items: Any) -> list[str]:
            merged_items: list[str] = []
            seen: set[str] = set()
            for item in (existing_items if isinstance(existing_items, list) else []) + (delta_items if isinstance(delta_items, list) else []):
                candidate = str(item).strip()
                if not candidate or candidate in seen:
                    continue
                seen.add(candidate)
                merged_items.append(candidate)
            return merged_items[-self.max_summary_items :]

        merged["medical_topics"] = _merge_unique_strings(merged.get("medical_topics", []), delta.get("medical_topics", []))
        merged["questions"] = _merge_unique_strings(merged.get("questions", []), delta.get("questions", []))
        merged["decisions"] = _merge_unique_strings(merged.get("decisions", []), delta.get("decisions", []))
        merged["source_message_ids"] = _merge_unique_strings(
            merged.get("source_message_ids", []),
            delta.get("source_message_ids", []),
        )
        merged["last_updated"] = self._now_iso()
        return merged

    def _append_recent_messages(self, input_text: str, output_text: str, metadata: dict[str, Any] | None = None) -> None:
        data = self._read_conversation()
        recent = data.setdefault("recent_messages", [])

        topic = None
        mode = None
        source = None
        citations: list[str] = []
        if metadata:
            topic = metadata.get("topic")
            mode = metadata.get("mode")
            source = metadata.get("source")
            raw_citations = metadata.get("citations", [])
            if isinstance(raw_citations, list):
                citations = [str(item).strip() for item in raw_citations if str(item).strip()]

        recent.append(
            {
                "id": f"msg_{uuid4().hex}",
                "role": "user",
                "content": input_text,
                "topic": topic,
                "mode": mode,
                "timestamp": self._now_iso(),
            }
        )
        recent.append(
            {
                "id": f"msg_{uuid4().hex}",
                "role": "assistant",
                "content": output_text,
                "topic": topic,
                "mode": mode,
                "source": source,
                "citations": citations,
                "timestamp": self._now_iso(),
            }
        )

        data["turn_count"] = int(data.get("turn_count", 0)) + 1

        if len(recent) >= self.summary_trigger_messages:
            overflow = len(recent) - self.recent_window_messages
            if overflow > 0:
                older_messages = recent[:overflow]
                summary_delta = self._build_summary_delta(older_messages, data.get("user_facts", {}))
                if self._validate_summary_delta(summary_delta):
                    data["summary_memory"] = self._merge_summary(data.get("summary_memory", {}), summary_delta)
                data["recent_messages"] = recent[overflow:]
            else:
                data["recent_messages"] = recent
        else:
            data["recent_messages"] = recent

        self._write_conversation(data)

    def add_user_fact(self, content: str, key: str | None = None, value: str | None = None) -> None:
        entry: dict[str, Any] = {"type": "user_fact", "content": content}
        if key:
            entry["key"] = key
        if value:
            entry["value"] = value
        event_id = self._append_entry(entry)

        if key and value:
            data = self._read_conversation()
            user_facts = data.setdefault("user_facts", self._default_conversation_payload()["user_facts"])
            user_facts[str(key).strip().lower()] = str(value).strip()
            self._write_conversation(data)
        self._set_last_sync_event(event_id)

    def add_statement(self, content: str) -> None:
        event_id = self._append_entry({"type": "statement", "content": content})

        inferred_role = self._extract_role_from_text(content)
        if inferred_role:
            data = self._read_conversation()
            data.setdefault("user_facts", {})["role"] = inferred_role
            self._write_conversation(data)
        self._set_last_sync_event(event_id)

    def add_instruction(self, content: str) -> None:
        event_id = self._append_entry({"type": "instruction", "content": content})
        self._set_last_sync_event(event_id)

    def add_preference(self, content: str) -> None:
        event_id = self._append_entry({"type": "preference", "content": content})

        normalized_preference = self._extract_preference_from_text(content)
        if normalized_preference:
            data = self._read_conversation()
            user_facts = data.setdefault("user_facts", {})
            preferences = user_facts.setdefault("preferences", [])
            if normalized_preference not in preferences:
                preferences.append(normalized_preference)
            if len(preferences) > 20:
                user_facts["preferences"] = preferences[-20:]
            self._write_conversation(data)
        self._set_last_sync_event(event_id)

    def add_qa(self, input_text: str, output_text: str, metadata: dict[str, Any] | None = None) -> None:
        entry: dict[str, Any] = {
            "type": "qa",
            "input": input_text,
            "output": output_text,
        }
        if metadata:
            entry["metadata"] = metadata
        event_id = self._append_entry(entry)
        self._append_recent_messages(input_text=input_text, output_text=output_text, metadata=metadata)
        self._set_last_sync_event(event_id)

    def get_summary_memory(self) -> str:
        data = self._read_conversation()
        summary = data.get("summary_memory", self._default_summary_payload())
        if isinstance(summary, dict):
            return json.dumps(summary, ensure_ascii=False)
        return ""

    def get_summary_memory_structured(self) -> dict[str, Any]:
        data = self._read_conversation()
        summary = data.get("summary_memory", self._default_summary_payload())
        if isinstance(summary, dict):
            return summary
        return self._default_summary_payload()

    def get_recent_messages(self) -> list[dict[str, Any]]:
        data = self._read_conversation()
        recent = data.get("recent_messages", [])
        if isinstance(recent, list):
            return recent
        return []

    def get_user_facts(self) -> dict[str, Any]:
        data = self._read_conversation()
        user_facts = data.get("user_facts", {})
        if isinstance(user_facts, dict):
            return user_facts
        return {}

    def find_cached_qa(
        self,
        question: str,
        *,
        mode: str | None = None,
        allowed_sources: set[str] | None = None,
    ) -> dict[str, Any] | None:
        q = self._normalize(question)
        if not q:
            return None

        memory = self._read_log().get("memory", [])
        for entry in reversed(memory):
            if entry.get("type") != "qa":
                continue
            metadata = entry.get("metadata")
            if not isinstance(metadata, dict):
                continue
            if mode:
                candidate_mode = str(metadata.get("mode", "")).strip().lower()
                if candidate_mode != mode.strip().lower():
                    continue
            source = str(metadata.get("source", "")).strip().lower()
            if allowed_sources is not None and source not in allowed_sources:
                continue
            candidate = self._normalize(str(entry.get("input", "")))
            if candidate == q:
                return entry
        return None

    def get_latest_user_fact(self, key: str) -> str | None:
        normalized_key = key.strip().lower()
        memory = self._read_log().get("memory", [])
        for entry in reversed(memory):
            if entry.get("type") != "user_fact":
                continue
            if str(entry.get("key", "")).strip().lower() != normalized_key:
                continue
            value = str(entry.get("value", "")).strip()
            if value:
                return value

        conversation_facts = self.get_user_facts()
        direct_value = str(conversation_facts.get(normalized_key, "")).strip()
        if direct_value:
            return direct_value
        return None

    @staticmethod
    def _sanitize_user_facts_payload(user_facts: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(user_facts, dict):
            return {}

        sanitized = dict(user_facts)
        raw_name = str(sanitized.get("name", "")).strip()
        if raw_name.lower() in INVALID_USER_NAME_VALUES:
            sanitized["name"] = ""

        sanitized["company"] = str(sanitized.get("company", "")).strip()
        sanitized["competency_level"] = str(sanitized.get("competency_level", "")).strip()

        preferences = sanitized.get("preferences", [])
        if not isinstance(preferences, list):
            sanitized["preferences"] = []
        else:
            sanitized["preferences"] = [str(item).strip() for item in preferences if str(item).strip()]
        return sanitized

    def get_competency_level(self) -> str | None:
        """Return the stored competency level for the current user, if any."""

        value = self.get_latest_user_fact("competency_level")
        if value:
            return value

        data = self.get_user_facts()
        direct_value = str(data.get("competency_level", "")).strip()
        return direct_value or None

    def set_competency_level(self, level: str) -> None:
        """Persist the user's selected competency level when it changes."""

        normalized_level = str(level).strip()
        if not normalized_level:
            return

        current_level = self.get_competency_level()
        if current_level and current_level.strip().lower() == normalized_level.lower():
            return

        self.add_user_fact(
            content=f"User selected competency level {normalized_level}",
            key="competency_level",
            value=normalized_level,
        )

    def reset_session_memory(self, keep_user_facts: bool = True) -> None:
        """Reset short-term conversation memory while optionally preserving user facts."""

        current = self._read_conversation()
        payload = self._default_conversation_payload()
        if keep_user_facts:
            existing_facts = current.get("user_facts", {})
            if isinstance(existing_facts, dict):
                payload["user_facts"] = self._sanitize_user_facts_payload(existing_facts)
        self._write_conversation(payload)

        if not keep_user_facts:
            data = self._read_log()
            memory = data.get("memory", [])
            if isinstance(memory, list):
                data["memory"] = [entry for entry in memory if str(entry.get("type", "")).strip().lower() != "user_fact"]
                self._write_log(data)
