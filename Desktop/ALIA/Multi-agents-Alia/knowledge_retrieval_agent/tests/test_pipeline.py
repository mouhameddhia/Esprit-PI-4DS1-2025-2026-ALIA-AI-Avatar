"""Pipeline test using lightweight fakes for deterministic behavior."""

from app.config import Settings
from app.pipelines.rag_pipeline import RAGPipeline
from app.schemas.models import RetrievalDiagnostics, ScoredDocument


class FakeDense:
    """Deterministic dense retriever for tests."""

    def retrieve(self, query: str, top_k: int) -> list[ScoredDocument]:
        return [
            ScoredDocument(
                doc_id="doc_a",
                text="Ibuprofen may cause gastrointestinal adverse effects in some patients.",
                metadata={"source": "leaflet_a.pdf", "page": 2, "domain": "pharma"},
                dense_score=0.9,
            )
        ]


class FakeSparse:
    """Deterministic sparse retriever for tests."""

    def retrieve(self, query: str, top_k: int) -> list[ScoredDocument]:
        return [
            ScoredDocument(
                doc_id="doc_a",
                text="Ibuprofen may cause gastrointestinal adverse effects in some patients.",
                metadata={"source": "leaflet_a.pdf", "page": 2, "domain": "pharma"},
                bm25_score=0.8,
            )
        ]


class FakeReranker:
    """Pass-through reranker for unit testing."""

    def rerank(self, query: str, docs: list[ScoredDocument], top_k: int) -> list[ScoredDocument]:
        return docs[:top_k]


class FakeGenerator:
    """Deterministic answer generator for unit testing."""

    def generate(
        self,
        query: str,
        docs: list[ScoredDocument],
        response_language: str = "en",
    ) -> tuple[str, list[str], float, str, list[str]]:
        return (
            "Ibuprofen can cause gastrointestinal adverse effects. [1]",
            ["leaflet_a.pdf#page=2"],
            0.91,
            "",
            [],
        )


class FakeRewriter:
    """Simple query rewriter for deterministic tests."""

    def rewrite(self, query: str) -> str:
        return query + " side effects"

    def build_fallback_queries(self, query: str) -> list[str]:
        return [query, query + " toxicity"]


class FakeCRAG:
    """CRAG mock that does not trigger fallback."""

    def evaluate(self, docs: list[ScoredDocument]) -> RetrievalDiagnostics:
        return RetrievalDiagnostics(low_confidence=False, confidence_score=0.9)

    def should_expand(self, diagnostics: RetrievalDiagnostics) -> bool:
        return False

    def should_fallback(self, diagnostics: RetrievalDiagnostics) -> bool:
        return False


def test_pipeline_returns_answer_and_citations(tmp_path) -> None:
    """Validate that the pipeline returns grounded answer and citations."""

    settings = Settings(
        vector_store_dir=str(tmp_path),
        llm_model="llama3:8b",
        llm_base_url="http://localhost:11434",
    )
    pipeline = RAGPipeline(settings)

    pipeline.rewriter = FakeRewriter()
    pipeline.dense = FakeDense()
    pipeline.sparse = FakeSparse()
    pipeline.crag = FakeCRAG()
    pipeline.reranker = FakeReranker()
    pipeline.generator = FakeGenerator()

    response = pipeline.run("What are adverse effects of ibuprofen?")

    assert "Ibuprofen" in response.answer
    assert response.citations == ["leaflet_a.pdf#page=2"]
    assert len(response.retrieved_docs) == 1
    assert response.diagnostics.low_confidence is False
    assert response.answer_confidence == 0.91
    assert response.uncertainty == ""
    assert response.conflict_notes == []
    assert response.latency_ms >= 0


def test_pipeline_returns_all_indication_points(tmp_path) -> None:
    """Ensure indication questions return all indication lines from cited context."""

    settings = Settings(
        vector_store_dir=str(tmp_path),
        llm_model="llama3:8b",
        llm_base_url="http://localhost:11434",
    )
    pipeline = RAGPipeline(settings)

    pipeline.rewriter = FakeRewriter()
    pipeline.crag = FakeCRAG()
    pipeline.reranker = FakeReranker()

    class IndicationDense:
        def retrieve(self, query: str, top_k: int) -> list[ScoredDocument]:
            return [
                ScoredDocument(
                    doc_id="Gamme PEDIAKIDS_p12_c1",
                    text=(
                        "INDICATIONS\n"
                        "Soulage la toux grasse productive\n"
                        "Dégage les voies respiratoires\n"
                        "Effet Expectorant\n"
                        "COMPOSITION"
                    ),
                    metadata={"source": "Gamme PEDIAKIDS.pptx", "page": 12, "domain": "pharma"},
                    dense_score=0.9,
                    bm25_score=0.8,
                    final_score=0.9,
                )
            ]

    class IndicationSparse:
        def retrieve(self, query: str, top_k: int) -> list[ScoredDocument]:
            return []

    class OnePointGenerator:
        def generate(
            self,
            query: str,
            docs: list[ScoredDocument],
            response_language: str = "en",
        ) -> tuple[str, list[str], float, str, list[str]]:
            return (
                "Soulage la toux grasse productive",
                ["Gamme PEDIAKIDS.pptx#page=12"],
                0.95,
                "",
                [],
            )

    pipeline.dense = IndicationDense()
    pipeline.sparse = IndicationSparse()
    pipeline.generator = OnePointGenerator()

    response = pipeline.run("Quelle est l'indication de pediakids APITOU N°1 ?")

    assert response.answer == (
        "Soulage la toux grasse productive\n"
        "Dégage les voies respiratoires\n"
        "Effet Expectorant"
    )


def test_pipeline_disambiguates_base_product_vs_variant(tmp_path) -> None:
    """Ensure base product query excludes qualified sub-variant docs."""

    settings = Settings(
        vector_store_dir=str(tmp_path),
        llm_model="llama3:8b",
        llm_base_url="http://localhost:11434",
    )
    pipeline = RAGPipeline(settings)

    pipeline.rewriter = FakeRewriter()
    pipeline.crag = FakeCRAG()
    pipeline.reranker = FakeReranker()

    class FerbioticDense:
        def retrieve(self, query: str, top_k: int) -> list[ScoredDocument]:
            return [
                ScoredDocument(
                    doc_id="ferbiotic_base",
                    text=(
                        "INDICATIONS\n"
                        "Correction de la carence martiale\n"
                        "Prévention de l'anémie ferriprive\n"
                        "FERBIOTIC\n"
                        "60 Gélules"
                    ),
                    metadata={"source": "Gamme FERBIOTIC.pptx", "page": 2, "domain": "pharma"},
                    dense_score=0.8,
                    bm25_score=0.8,
                    final_score=0.8,
                    rerank_score=0.8,
                ),
                ScoredDocument(
                    doc_id="ferbiotic_lipo",
                    text=(
                        "INDICATIONS\n"
                        "Traitement des carences martiales avec fer liposomal\n"
                        "FERBIOTIC LIPO\n"
                        "30 Gélules"
                    ),
                    metadata={"source": "Gamme FERBIOTIC.pptx", "page": 5, "domain": "pharma"},
                    dense_score=0.9,
                    bm25_score=0.9,
                    final_score=0.9,
                    rerank_score=0.9,
                ),
            ]

    class EmptySparse:
        def retrieve(self, query: str, top_k: int) -> list[ScoredDocument]:
            return []

    class EchoGenerator:
        def generate(
            self,
            query: str,
            docs: list[ScoredDocument],
            response_language: str = "en",
        ) -> tuple[str, list[str], float, str, list[str]]:
            if not docs:
                return ("", [], 0.0, "", [])
            source = str(docs[0].metadata.get("source", "unknown"))
            page = str(docs[0].metadata.get("page", "n/a"))
            return (docs[0].text, [f"{source}#page={page}"], 1.0, "", [])

    pipeline.dense = FerbioticDense()
    pipeline.sparse = EmptySparse()
    pipeline.generator = EchoGenerator()

    base_response = pipeline.run("Quelle sont l indications de FERBIOTIC ?")
    assert "FERBIOTIC LIPO" not in base_response.answer

    variant_response = pipeline.run("Quelle sont l indications de FERBIOTIC LIPO ?")
    assert "FERBIOTIC LIPO" in variant_response.answer
