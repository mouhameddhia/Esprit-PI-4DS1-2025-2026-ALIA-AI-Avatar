"""LLM answer generation with citation-safe structured output."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.config import Settings
from app.prompts.prompt_registry import get_prompt
from app.schemas.models import GeneratedAnswer, ScoredDocument
from app.utils.logger import get_logger


class AnswerGenerator:
    """Generate grounded answers from retrieved documents."""

    def __init__(self, settings: Settings) -> None:
        """Initialize local Ollama chat model and prompt registry selection."""

        self.settings = settings
        self.logger = get_logger("knowledge_retrieval_agent.generator")
        self.prompt_text = get_prompt(settings.prompt_name)
        self.llm = ChatOllama(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            temperature=0,
            num_ctx=settings.llm_num_ctx,
            request_timeout=settings.request_timeout_seconds,
        )
        self.structured_llm = self.llm.with_structured_output(GeneratedAnswer)
        self.prompt = ChatPromptTemplate.from_template(self.prompt_text)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count using a simple character heuristic."""

        return max(1, len(text) // 4)

    @staticmethod
    def _language_instruction(response_language: str) -> str:
        """Return a strict language instruction for final answer generation."""

        if response_language == "fr":
            return "Reponds strictement en francais."
        return "Respond strictly in English."

    @staticmethod
    def _format_context(docs: list[ScoredDocument]) -> str:
        """Render documents as numbered context blocks for prompting."""

        blocks: list[str] = []
        for idx, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown_source")
            page = doc.metadata.get("page", "n/a")
            blocks.append(f"[{idx}] source={source} page={page}\n{doc.text}")
        return "\n\n".join(blocks)

    def _build_context_with_budget(self, query: str, docs: list[ScoredDocument]) -> tuple[str, list[ScoredDocument], bool]:
        """Trim context to fit within the configured model budget."""

        budget = max(0, self.settings.llm_num_ctx - self.settings.llm_max_output_tokens)
        prompt_overhead = self._estimate_tokens(self.prompt_text.format(query=query, context=""))
        remaining_budget = max(0, budget - prompt_overhead)

        selected_docs: list[ScoredDocument] = []
        current_tokens = 0
        truncated = False

        for doc in docs:
            doc_text = f"[x] source={doc.metadata.get('source', 'unknown_source')} page={doc.metadata.get('page', 'n/a')}\n{doc.text}"
            doc_tokens = self._estimate_tokens(doc_text)
            if selected_docs and current_tokens + doc_tokens > remaining_budget:
                truncated = True
                break
            selected_docs.append(doc)
            current_tokens += doc_tokens

        if not selected_docs and docs:
            selected_docs = docs[:1]
            truncated = True

        return self._format_context(selected_docs), selected_docs, truncated

    @staticmethod
    def _extract_used_citations(answer: str, docs: list[ScoredDocument]) -> list[str]:
        """Return citations only for the chunks explicitly referenced in the answer."""

        used_indices = {int(match) - 1 for match in re.findall(r"\[(\d+)\]", answer)}
        citations: list[str] = []
        seen: set[str] = set()

        for index, doc in enumerate(docs):
            if index not in used_indices:
                continue
            source = str(doc.metadata.get("source", "unknown_source"))
            page = str(doc.metadata.get("page", "n/a"))
            citation = f"{source}#page={page}"
            if citation not in seen:
                seen.add(citation)
                citations.append(citation)
        return citations

    @staticmethod
    def _fallback_result(query: str, reason: str) -> GeneratedAnswer:
        """Return a deterministic fallback response when the LLM fails."""

        return GeneratedAnswer(
            answer=f"Unable to generate a reliable answer for '{query}' because {reason}.",
            citation_indices=[],
            confidence=0.0,
            uncertainty=reason,
            conflict_notes=[],
        )

    def _invoke_llm(self, prompt_query: str, context: str, fallback_query: str) -> GeneratedAnswer:
        """Invoke the structured LLM with one retry and structured fallback."""

        chain = self.prompt | self.structured_llm
        for attempt in range(2):
            try:
                response = chain.invoke({"query": prompt_query, "context": context})
                if isinstance(response, GeneratedAnswer):
                    return response
                if isinstance(response, dict):
                    return GeneratedAnswer.model_validate(response)
                return GeneratedAnswer.model_validate(response)
            except Exception as exc:
                self.logger.exception(
                    "generation_attempt_failed",
                    extra={"extra": {"attempt": attempt + 1, "error": type(exc).__name__}},
                )
                last_error = f"LLM invocation failed after attempt {attempt + 1}"
        return self._fallback_result(fallback_query, last_error)

    def generate(
        self,
        query: str,
        docs: list[ScoredDocument],
        response_language: str = "en",
    ) -> tuple[str, list[str], float, str, list[str]]:
        """Generate final answer text, used citations, and uncertainty signals."""

        if not docs:
            fallback = self._fallback_result(query, "no relevant context was retrieved")
            return fallback.answer, [], fallback.confidence, fallback.uncertainty, fallback.conflict_notes

        language = response_language if response_language in {"en", "fr"} else "en"
        prompt_query = f"{query}\n\n{self._language_instruction(language)}"

        context, trimmed_docs, was_truncated = self._build_context_with_budget(prompt_query, docs)
        self.logger.info(
            "generation_context_prepared",
            extra={
                "extra": {
                    "query": query,
                    "response_language": language,
                    "retrieved_docs": [doc.doc_id for doc in trimmed_docs],
                    "context_truncated": was_truncated,
                    "estimated_context_tokens": self._estimate_tokens(context),
                }
            },
        )

        response = self._invoke_llm(prompt_query=prompt_query, context=context, fallback_query=query)
        answer = response.answer
        citations = self._extract_used_citations(answer, trimmed_docs)

        if not citations and response.citation_indices:
            seen: set[str] = set()
            for index in response.citation_indices:
                mapped_index = index - 1
                if mapped_index < 0 or mapped_index >= len(trimmed_docs):
                    continue
                doc = trimmed_docs[mapped_index]
                source = str(doc.metadata.get("source", "unknown_source"))
                page = str(doc.metadata.get("page", "n/a"))
                citation = f"{source}#page={page}"
                if citation not in seen:
                    seen.add(citation)
                    citations.append(citation)

        self.logger.info(
            "generation_complete",
            extra={
                "extra": {
                    "query": query,
                    "confidence": response.confidence,
                    "uncertainty": response.uncertainty,
                    "conflict_notes": response.conflict_notes,
                    "citation_count": len(citations),
                    "answer_preview": answer[:240],
                }
            },
        )
        return answer, citations, response.confidence, response.uncertainty, response.conflict_notes
