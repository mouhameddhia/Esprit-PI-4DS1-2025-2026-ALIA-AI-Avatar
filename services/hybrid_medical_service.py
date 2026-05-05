"""
Hybrid Medical Agent service wrapper.

Safe lazy import of HybridMedicalController to avoid startup-time import crashes.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from typing import Any

from utils.config import config

logger = logging.getLogger(__name__)


class HybridMedicalService:
    """Wrapper around `hybrid_medical_agent.agent.controller.HybridMedicalController`."""

    def __init__(self) -> None:
        self.controller: Any | None = None
        self.is_ready = False
        self.competency_level = str(config.MEDICAL_COMPETENCY_LEVEL or "JUNIOR").strip().upper()
        self.agent_path = self._discover_agent_path()

    def _discover_agent_path(self) -> Path:
        configured = str(config.HYBRID_MEDICAL_AGENT_PATH or "").strip()
        candidates = [
            Path(configured) if configured else None,
            Path(r"D:/projects/aliaFinal/External/knowledge-retriever/Desktop/ALIA/Multi-agents-Alia/hybrid_medical_agent"),
            Path(str(config.RAG_REPO_PATH or ".")) / "hybrid_medical_agent",
        ]
        for candidate in candidates:
            if not candidate:
                continue
            if candidate.exists() and (candidate / "agent" / "controller.py").exists():
                return candidate.resolve()
        raise FileNotFoundError(
            "Could not find hybrid_medical_agent. Set HYBRID_MEDICAL_AGENT_PATH in backend .env"
        )

    def _ensure_controller(self) -> Any:
        if self.controller is not None:
            return self.controller

        workspace_root = self.agent_path.parent
        if str(workspace_root) not in sys.path:
            sys.path.insert(0, str(workspace_root))

        controller_mod = import_module("hybrid_medical_agent.agent.controller")
        competency_mod = import_module("hybrid_medical_agent.agent.competency_framework")
        HybridMedicalController = getattr(controller_mod, "HybridMedicalController")
        coerce_competency_level = getattr(competency_mod, "coerce_competency_level")

        provider = str(getattr(config, "MEDICAL_LLM_PROVIDER", "groq") or "groq").strip().lower()
        model = str(getattr(config, "MEDICAL_LLM_MODEL", "") or "").strip() or "llama3:8b"
        base_url = str(getattr(config, "MEDICAL_LLM_BASE_URL", "") or "").strip()
        if not base_url:
            base_url = "https://api.groq.com/openai/v1" if provider == "groq" else str(config.OLLAMA_URL)

        self.controller = HybridMedicalController(
            workspace_root=workspace_root,
            llm_model=model,
            llm_base_url=base_url,
            competency_level=coerce_competency_level(self.competency_level),
            llm_provider=provider,
            llm_api_key=(
                getattr(config, "GROQ_API_KEY", None)
                or os.getenv("GROQ_API_KEY")
                or os.getenv("LLM_API_KEY")
            ),
        )
        self.is_ready = True
        logger.info("HybridMedicalController initialized (provider=%s, level=%s)", provider, self.competency_level)
        return self.controller

    def warmup(self) -> None:
        self._ensure_controller()

    def set_competency_level(self, level: str) -> None:
        self.competency_level = str(level or "JUNIOR").strip().upper()
        if self.controller is not None:
            competency_mod = import_module("hybrid_medical_agent.agent.competency_framework")
            coerce = getattr(competency_mod, "coerce_competency_level")
            self.controller.competency_level = coerce(self.competency_level)

    def handle_query(
        self,
        question: str,
        mode: str,
        user_id: str | None,
        preferred_drug_name: str | None,
        preferred_language: str | None,
        messages: list[dict[str, str]] | None,
        conversation_state: dict[str, str] | None,
        session_reset: bool,
    ) -> dict[str, Any]:
        """Delegate to `HybridMedicalController.handle_query`; dict for `LLMService` (uses `.get()`)."""
        ctrl = self._ensure_controller()
        response = ctrl.handle_query(
            question,
            mode,
            user_id,
            preferred_drug_name,
            preferred_language,
            messages,
            conversation_state,
            session_reset,
        )
        return asdict(response)

