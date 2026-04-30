"""Tests for metadata-driven retrieval behavior."""

from datetime import datetime, timezone

from app.ingestion.loader import PDFLoader
from app.retrieval.hybrid import HybridRetriever


def test_loader_infers_metadata_fields(tmp_path) -> None:
    """Loader should populate doc_date and source_tier metadata."""

    pdf_path = tmp_path / "ema_smpc_fda.pdf"
    pdf_path.write_bytes(b"dummy")

    assert PDFLoader._source_tier(pdf_path) == 1

    doc_date = PDFLoader._doc_date(pdf_path)
    datetime.strptime(doc_date, "%Y-%m-%d")


def test_loader_discovers_nested_files(tmp_path) -> None:
    """Loader should find all files in nested folders under data/."""

    (tmp_path / "root.pdf").write_bytes(b"x")
    nested_dir = tmp_path / "nested" / "deep"
    nested_dir.mkdir(parents=True)
    (nested_dir / "inside.PDF").write_bytes(b"x")
    (nested_dir / "ignore.txt").write_text("nope", encoding="utf-8")
    (nested_dir / "table.csv").write_text("a,b\n1,2", encoding="utf-8")

    loader = PDFLoader(str(tmp_path))
    discovered = loader._iter_file_paths()

    discovered_names = {path.name for path in discovered}
    assert discovered_names == {"root.pdf", "inside.PDF", "ignore.txt", "table.csv"}


def test_hybrid_metadata_quality_prefers_authority_and_recency() -> None:
    """Recent tier-1 pharma docs should score above stale lower-tier docs."""

    today = datetime.now(timezone.utc).date().isoformat()
    stale = "2000-01-01"

    recent_score = HybridRetriever._metadata_quality(
        {"source": "ema.pdf", "page": 1, "domain": "pharma", "source_tier": 1, "doc_date": today}
    )
    stale_score = HybridRetriever._metadata_quality(
        {"source": "blog.pdf", "page": 1, "domain": "pharma", "source_tier": 4, "doc_date": stale}
    )

    assert recent_score > stale_score

