"""End-to-end pipeline for hybrid retrieval + CRAG + reranking + generation."""

from __future__ import annotations

import re

from app.config import Settings
from app.correction.crag import CRAGValidator
from app.generation.generator import AnswerGenerator
from app.query.rewrite import QueryRewriter
from app.reranking.reranker import CrossEncoderReranker
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.sparse_retriever import SparseRetriever
from app.schemas.models import EvidenceSnippet, PipelineResponse, RetrievalDiagnostics, RetrievalQuality, ScoredDocument
from app.utils.helpers import measure_time
from app.utils.logger import get_logger


class RAGPipeline:
    """Production-oriented retrieval pipeline with corrective fallback flow."""

    def __init__(self, settings: Settings) -> None:
        """Wire all pipeline components using runtime settings."""

        self.settings = settings
        self.logger = get_logger("knowledge_retrieval_agent.pipeline")

        self.rewriter = QueryRewriter()
        self.dense = DenseRetriever(settings.vector_store_dir, settings.embedding_model)
        self.sparse = SparseRetriever(settings.vector_store_dir)
        self.hybrid = HybridRetriever(settings.alpha, settings.beta, settings.gamma)
        self.crag = CRAGValidator(
            min_top_score=settings.min_top_score,
            min_mean_score=settings.min_mean_score,
            min_source_count=settings.min_source_count,
            fallback_enabled=settings.fallback_enabled,
        )
        self.reranker = CrossEncoderReranker(settings.reranker_model)
        self.generator = AnswerGenerator(settings)

    def _retrieve_hybrid(self, query: str) -> list[ScoredDocument]:
        """Retrieve dense + sparse results and combine with weighted scoring."""

        dense_docs = self.dense.retrieve(query, top_k=self.settings.dense_top_k)
        sparse_docs = self.sparse.retrieve(query, top_k=self.settings.sparse_top_k)
        return self.hybrid.combine(dense_docs, sparse_docs, top_k=self.settings.hybrid_top_k)

    @staticmethod
    def _normalized_query_tokens(query: str) -> list[str]:
        """Extract stable alphanumeric query tokens for identifier-aware matching."""

        tokens = re.findall(r"[a-z0-9]+", query.lower())
        return [token for token in tokens if len(token) >= 4]

    @staticmethod
    def _product_query_tokens(query: str) -> list[str]:
        """Extract likely product tokens by excluding common intent/function words."""

        stopwords = {
            "what",
            "which",
            "where",
            "when",
            "how",
            "pourquoi",
            "quelle",
            "quelles",
            "quel",
            "quels",
            "sont",
            "with",
            "from",
            "dans",
            "avec",
            "pour",
            "indication",
            "indications",
            "composition",
            "posologie",
            "dose",
            "dosage",
            "contre",
            "effets",
            "effet",
            "de",
            "des",
            "les",
            "the",
            "this",
            "that",
        }
        tokens = RAGPipeline._normalized_query_tokens(query)
        return [token for token in tokens if token not in stopwords]

    @staticmethod
    def _extract_variant_qualifiers(search_text: str, base_token: str) -> set[str]:
        """Extract words directly following a base token (for variant disambiguation)."""

        qualifiers: set[str] = set()
        pattern = re.compile(rf"\b{re.escape(base_token)}\s+([a-z][a-z0-9-]{{2,}})\b", re.IGNORECASE)
        for match in pattern.finditer(search_text):
            qualifier = re.sub(r"[^a-z0-9]+", "", match.group(1).lower())
            if qualifier and qualifier not in {"page", "mg", "ml", "capsule", "capsules", "gelules"}:
                qualifiers.add(qualifier)
        return qualifiers

    @staticmethod
    def _extract_number_marker(text: str) -> str | None:
        """Extract product number marker such as N°1, No 1, or n 1."""

        normalized = text.lower().replace("º", "°")
        match = re.search(r"\bn\s*[°o]?\s*([0-9il])\b", normalized)
        if not match:
            return None
        marker = match.group(1)
        if marker in {"i", "l"}:
            return "1"
        return marker

    @staticmethod
    def _is_indication_query(query: str) -> bool:
        """Return True when the user asks for indications."""

        lowered = query.lower()
        return any(token in lowered for token in ("indication", "indications", "indiqué", "indication de"))

    @staticmethod
    def _extract_indication_points(text: str, max_points: int = 8) -> list[str]:
        """Extract all indication bullet lines from a document chunk when possible."""

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return []

        start_index = -1
        for idx, line in enumerate(lines):
            if "indication" in line.lower():
                start_index = idx + 1
                break
        if start_index < 0:
            return []

        stop_headers = {
            "composition",
            "posologie",
            "utilisation",
            "age",
            "conseils dutilisation",
        }
        points: list[str] = []
        for line in lines[start_index:]:
            lowered = line.lower()
            normalized = re.sub(r"[^a-z0-9]+", " ", lowered).strip()
            normalized_compact = normalized.replace(" ", "")
            if any(
                normalized == header
                or normalized.startswith(f"{header} ")
                or normalized_compact == header
                for header in stop_headers
            ):
                break
            points.append(line.lstrip("-• "))
            if len(points) >= max_points:
                break

        return [point for point in points if point]

    @staticmethod
    def _parse_citation(citation: str) -> tuple[str, str]:
        """Parse citation format source#page=n into comparable parts."""

        source, _, page = citation.partition("#page=")
        return source, page or "n/a"

    def _enforce_indication_answer(self, query: str, answer: str, citations: list[str], docs: list[ScoredDocument]) -> str:
        """Force indication answers to include all indication points from cited evidence."""

        if not self._is_indication_query(query) or not citations:
            return answer

        cited_pairs = {self._parse_citation(citation) for citation in citations}
        for doc in docs:
            doc_pair = (str(doc.metadata.get("source", "unknown_source")), str(doc.metadata.get("page", "n/a")))
            if doc_pair not in cited_pairs:
                continue
            points = self._extract_indication_points(doc.text)
            if len(points) < 2:
                continue
            return "\n".join(points)

        return answer

    def _identifier_bonus(self, query: str, doc: ScoredDocument) -> float:
        """Boost documents that exactly match product identifiers in the query."""

        query_marker = self._extract_number_marker(query)
        doc_marker = self._extract_number_marker(doc.text)

        bonus = 0.0
        if query_marker and doc_marker == query_marker:
            bonus += 4.0
        if query_marker and doc_marker and doc_marker != query_marker:
            bonus -= 4.0

        query_tokens = self._normalized_query_tokens(query)
        doc_text = doc.text.lower()
        token_hits = sum(1 for token in query_tokens if token in doc_text)
        bonus += 0.25 * float(token_hits)

        # Variant-sensitive disambiguation.
        # If user asks for a base product name without a qualifier, penalize variant docs
        # (e.g., "ferbiotic" should not prioritize "ferbiotic lipo").
        product_tokens = self._product_query_tokens(query)
        doc_search_text = " ".join(
            [
                str(doc.metadata.get("source", "")),
                doc.doc_id,
                doc.text[:600],
            ]
        ).lower()
        for base_token in product_tokens:
            qualifiers = self._extract_variant_qualifiers(doc_search_text, base_token)
            if not qualifiers:
                continue

            if any(qualifier in product_tokens for qualifier in qualifiers):
                bonus += 1.5
            else:
                bonus -= 1.5
        return bonus

    def _postprocess_reranked(self, query: str, docs: list[ScoredDocument]) -> list[ScoredDocument]:
        """Apply deterministic identifier-aware ordering on top of reranker scores."""

        if not docs:
            return docs

        ranked = sorted(
            docs,
            key=lambda item: item.rerank_score + self._identifier_bonus(query, item),
            reverse=True,
        )
        return self._apply_variant_disambiguation(query, ranked)

    def _apply_variant_disambiguation(self, query: str, docs: list[ScoredDocument]) -> list[ScoredDocument]:
        """Filter variant/base product collisions to honor explicit user product naming."""

        product_tokens = self._product_query_tokens(query)
        if not product_tokens:
            return docs

        filtered_docs = docs
        for base_token in product_tokens:
            variant_docs: list[tuple[ScoredDocument, set[str]]] = []
            base_docs: list[ScoredDocument] = []

            for doc in filtered_docs:
                doc_search_text = " ".join(
                    [
                        str(doc.metadata.get("source", "")),
                        doc.doc_id,
                        doc.text[:600],
                    ]
                ).lower()
                qualifiers = self._extract_variant_qualifiers(doc_search_text, base_token)
                if qualifiers:
                    variant_docs.append((doc, qualifiers))
                elif re.search(rf"\b{re.escape(base_token)}\b", doc_search_text):
                    base_docs.append(doc)

            if not variant_docs:
                continue

            query_has_explicit_qualifier = any(
                qualifier in product_tokens for _, qualifiers in variant_docs for qualifier in qualifiers
            )

            if query_has_explicit_qualifier:
                wanted = [
                    doc
                    for doc, qualifiers in variant_docs
                    if any(qualifier in product_tokens for qualifier in qualifiers)
                ]
                if wanted:
                    kept_ids = {doc.doc_id for doc in wanted}
                    filtered_docs = [doc for doc in filtered_docs if doc.doc_id in kept_ids]
                continue

            if base_docs:
                kept_ids = {doc.doc_id for doc in base_docs}
                filtered_docs = [doc for doc in filtered_docs if doc.doc_id in kept_ids]

        return filtered_docs if filtered_docs else docs

    @staticmethod
    def _extract_indication_snippet(text: str, max_lines: int = 4) -> str:
        """Extract a compact snippet centered on INDICATIONS lines when present."""

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""

        for index, line in enumerate(lines):
            if "indication" in line.lower():
                selected = lines[index : index + max_lines]
                return "\n".join(selected)

        return "\n".join(lines[:max_lines])

    def _build_supporting_evidence(
        self,
        query: str,
        docs: list[ScoredDocument],
        citations: list[str],
        max_items: int = 1,
    ) -> list[EvidenceSnippet]:
        """Return concise evidence snippets to show why the answer was selected."""

        candidates = docs
        if citations:
            cited_pairs: set[tuple[str, str]] = set()
            for citation in citations:
                source, _, page = citation.partition("#page=")
                cited_pairs.add((source, page or "n/a"))

            filtered = [
                doc
                for doc in docs
                if (str(doc.metadata.get("source", "unknown_source")), str(doc.metadata.get("page", "n/a"))) in cited_pairs
            ]
            if filtered:
                candidates = filtered

        ranked_candidates = sorted(
            candidates,
            key=lambda item: self._identifier_bonus(query, item),
            reverse=True,
        )

        snippets: list[EvidenceSnippet] = []
        for doc in ranked_candidates[:max_items]:
            snippets.append(
                EvidenceSnippet(
                    doc_id=doc.doc_id,
                    text=self._extract_indication_snippet(doc.text),
                    source=str(doc.metadata.get("source", "unknown_source")),
                    page=doc.metadata.get("page", "n/a"),
                )
            )
        return snippets

    def run(self, query: str, response_language: str = "en") -> PipelineResponse:
        """Execute full RAG pipeline with CRAG fallback and citations."""

        with measure_time() as timing:
            rewritten = self.rewriter.rewrite(query)
            self.logger.info(
                "query_received",
                extra={"extra": {"query": query, "rewritten_query": rewritten}},
            )

            docs = self._retrieve_hybrid(rewritten)
            diagnostics: RetrievalDiagnostics = self.crag.evaluate(docs)
            diagnostics.query_variants = [rewritten]

            if self.crag.should_expand(diagnostics):
                variants = self.rewriter.build_fallback_queries(query)
                diagnostics.query_variants = variants
                self.logger.info(
                    "crag_ambiguous_expand",
                    extra={"extra": {"query": query, "quality": diagnostics.quality.value, "variants": variants}},
                )

                expanded_docs: list[ScoredDocument] = []
                for variant in variants:
                    expanded_docs.extend(self._retrieve_hybrid(variant))

                best_by_id: dict[str, ScoredDocument] = {doc.doc_id: doc for doc in docs}
                for doc in expanded_docs:
                    existing = best_by_id.get(doc.doc_id)
                    if existing is None or doc.final_score > existing.final_score:
                        best_by_id[doc.doc_id] = doc

                docs = sorted(best_by_id.values(), key=lambda d: d.final_score, reverse=True)
                docs = docs[: self.settings.hybrid_top_k]
                diagnostics = self.crag.evaluate(docs)
                diagnostics.query_variants = variants

            if self.crag.should_fallback(diagnostics):
                variants = self.rewriter.build_fallback_queries(query)
                diagnostics.query_variants = variants
                diagnostics.fallback_triggered = True
                self.logger.warning(
                    "fallback_triggered",
                    extra={
                        "extra": {
                            "query": query,
                            "quality": diagnostics.quality.value,
                            "reasons": diagnostics.reasons,
                            "variants": variants,
                        }
                    },
                )

                fallback_docs: list[ScoredDocument] = []
                for variant in variants:
                    fallback_docs.extend(self._retrieve_hybrid(variant))

                # Re-combine through sparse/dense score fusion behavior by deduping on best final_score.
                best_by_id: dict[str, ScoredDocument] = {}
                for doc in fallback_docs:
                    existing = best_by_id.get(doc.doc_id)
                    if existing is None or doc.final_score > existing.final_score:
                        best_by_id[doc.doc_id] = doc
                docs = sorted(best_by_id.values(), key=lambda d: d.final_score, reverse=True)
                docs = docs[: self.settings.hybrid_top_k]

                diagnostics = self.crag.evaluate(docs)
                diagnostics.query_variants = variants
                diagnostics.fallback_triggered = True
                diagnostics.quality = RetrievalQuality.INCORRECT

            self.logger.info(
                "retrieval_complete",
                extra={
                    "extra": {
                        "retrieved_count": len(docs),
                        "top_doc_ids": [doc.doc_id for doc in docs[:5]],
                        "confidence_score": diagnostics.confidence_score,
                    }
                },
            )

            reranked = self.reranker.rerank(rewritten, docs, top_k=self.settings.rerank_top_k)
            reranked = self._postprocess_reranked(query, reranked)
            answer, citations, answer_confidence, uncertainty, conflict_notes = self.generator.generate(
                query,
                reranked,
                response_language=response_language,
            )
            answer = self._enforce_indication_answer(query, answer, citations, reranked)
            supporting_evidence = self._build_supporting_evidence(query, reranked, citations, max_items=1)

            self.logger.info(
                "generation_complete",
                extra={
                    "extra": {
                        "citation_count": len(citations),
                        "answer_confidence": answer_confidence,
                        "uncertainty": uncertainty,
                        "conflict_notes": conflict_notes,
                        "answer_preview": answer[:160],
                    }
                },
            )

        return PipelineResponse(
            answer=answer,
            citations=citations,
            supporting_evidence=supporting_evidence,
            retrieved_docs=reranked,
            diagnostics=diagnostics,
            answer_confidence=answer_confidence,
            uncertainty=uncertainty,
            conflict_notes=conflict_notes,
            latency_ms=timing["elapsed_ms"],
        )
