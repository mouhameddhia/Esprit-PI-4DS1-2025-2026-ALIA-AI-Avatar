"""Document ingestion pipeline — indexes PDFs and Excel files into the vector DB."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def build_documents(source_dir: str | Path) -> List[Dict[str, Any]]:
    """
    Scan source_dir for PDF/Excel files and return a list of document dicts
    ready for upsert into the vector store.
    Each dict has: id, text, metadata.
    """
    source_dir = Path(source_dir)
    documents: List[Dict[str, Any]] = []

    for path in sorted(source_dir.rglob("*")):
        if path.suffix.lower() not in {".pdf", ".xlsx", ".xls", ".csv"}:
            continue
        try:
            text = _extract_text(path)
            if text:
                documents.append({
                    "id": path.stem,
                    "text": text,
                    "metadata": {"source": str(path), "filename": path.name},
                })
        except Exception as exc:
            logger.warning("Failed to extract %s: %s", path, exc)

    logger.info("Built %d documents from %s", len(documents), source_dir)
    return documents


def upsert_documents(documents: List[Dict[str, Any]], index_name: Optional[str] = None) -> int:
    """Upsert documents into Pinecone. Returns count of upserted vectors."""
    if not documents:
        return 0
    try:
        from backend.vector_db.indexing import upsert_to_pinecone
        return upsert_to_pinecone(documents, index_name=index_name)
    except Exception as exc:
        logger.error("Upsert failed: %s", exc)
        return 0


async def main() -> int:
    """CLI entry point: ingest all documents in useful-files/."""
    import os
    source = Path(os.getenv("INGESTION_SOURCE_DIR", "useful-files"))
    if not source.exists():
        logger.error("Source dir not found: %s", source)
        return 1
    docs = build_documents(source)
    n = upsert_documents(docs)
    logger.info("Ingested %d vectors", n)
    return 0


def _extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix in {".xlsx", ".xls"}:
        return _extract_excel(path)
    if suffix == ".csv":
        return path.read_text(encoding="utf-8", errors="replace")
    return ""


def _extract_pdf(path: Path) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except ImportError:
        logger.warning("pdfplumber not installed; skipping %s", path)
        return ""


def _extract_excel(path: Path) -> str:
    try:
        import openpyxl
        wb = openpyxl.load_workbook(path, data_only=True)
        rows = []
        for sheet in wb.worksheets:
            for row in sheet.iter_rows(values_only=True):
                rows.append("\t".join(str(c) for c in row if c is not None))
        return "\n".join(rows)
    except ImportError:
        logger.warning("openpyxl not installed; skipping %s", path)
        return ""
