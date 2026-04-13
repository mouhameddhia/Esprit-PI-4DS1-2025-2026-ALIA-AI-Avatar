"""Import useful training files into MongoDB and the vector database."""

from __future__ import annotations

import argparse
import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from dotenv import load_dotenv
from openpyxl import load_workbook
from pypdf import PdfReader


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FOLDER = REPO_ROOT / "useful-files"


def _load_env() -> None:
    load_dotenv(dotenv_path=REPO_ROOT / "backend" / ".env")


def _get_backend_services():
    from backend.main import db, knowledge_document_indexer

    return db, knowledge_document_indexer


def _utc_now():
    return datetime.now(timezone.utc)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def chunk_text(text: str, max_chars: int = 1400, overlap: int = 150) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + max_chars)
        if end < len(normalized):
            boundary = normalized.rfind(". ", start, end)
            if boundary > start + 200:
                end = boundary + 1
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _page_heading(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0][:180] if lines else ""


def extract_pdf_documents(pdf_path: Path) -> List[Dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    documents: List[Dict[str, Any]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        text = normalize_text(raw_text)
        if not text:
            continue
        heading = _page_heading(raw_text)
        for chunk_index, chunk in enumerate(chunk_text(text), start=1):
            documents.append(
                {
                    "source_name": pdf_path.stem,
                    "source_type": "pdf",
                    "source_path": str(pdf_path),
                    "title": pdf_path.stem,
                    "section_title": heading or pdf_path.stem,
                    "language": "fr",
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "chunk_text": chunk,
                    "keywords_text": heading,
                    "topic_tags": [],
                }
            )
    return documents


def _row_values(row: Iterable[Any]) -> List[str]:
    values: List[str] = []
    for cell in row:
        if cell is None:
            continue
        value = normalize_text(str(cell))
        if value:
            values.append(value)
    return values


def extract_spreadsheet_documents(xlsx_path: Path) -> List[Dict[str, Any]]:
    wb = load_workbook(xlsx_path, data_only=True, read_only=True)
    documents: List[Dict[str, Any]] = []

    for sheet in wb.worksheets:
        for row_index, row in enumerate(sheet.iter_rows(values_only=True), start=1):
            values = _row_values(row)
            if not values:
                continue
            row_text = " | ".join(values)
            if row_index == 1:
                continue

            metadata: Dict[str, Any] = {
                "source_name": xlsx_path.stem,
                "source_type": "spreadsheet",
                "source_path": str(xlsx_path),
                "title": xlsx_path.stem,
                "section_title": sheet.title,
                "sheet_name": sheet.title,
                "language": "fr",
                "row_number": row_index,
                "chunk_index": 1,
                "chunk_text": row_text,
                "keywords_text": "",
                "topic_tags": [sheet.title.lower()],
            }

            if sheet.title.lower() in {"matrice", "evaluation", "checklist", "seuils de passage"}:
                metadata["source_type"] = "competency_matrix"

            documents.append(metadata)

    return documents


def build_documents(folder: Path) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            documents.extend(extract_pdf_documents(path))
        elif suffix in {".xlsx", ".xlsm", ".xls"}:
            documents.extend(extract_spreadsheet_documents(path))
    return documents


async def upsert_documents(folder: Path, db, knowledge_document_indexer) -> Dict[str, Any]:
    documents = build_documents(folder)
    if not documents:
        return {"success": False, "error": f"No supported files found in {folder}"}

    now = _utc_now()
    inserted_or_updated = 0

    for document in documents:
        query = {
            "source_name": document["source_name"],
            "source_type": document["source_type"],
            "title": document["title"],
            "chunk_index": document.get("chunk_index", 1),
        }
        if document["source_type"] == "pdf":
            query["page_number"] = document["page_number"]
        else:
            query["sheet_name"] = document.get("sheet_name")
            query["row_number"] = document.get("row_number")

        existing = await db.knowledge_documents.find_one(query, {"_id": 1, "created_at": 1})
        record = {**document, "created_at": now, "updated_at": now}

        if existing:
            await db.knowledge_documents.update_one(
                {"_id": existing["_id"]},
                {"$set": {**record, "created_at": existing.get("created_at", now)}},
            )
        else:
            await db.knowledge_documents.insert_one(record)

        inserted_or_updated += 1

    source_names = sorted({document["source_name"] for document in documents})
    total_indexed = 0
    for source_name in source_names:
        index_result = await knowledge_document_indexer.index_documents(db, source_name=source_name)
        if not index_result.get("success"):
            return index_result
        total_indexed += index_result.get("indexed_count", 0)

    return {
        "success": True,
        "chunks_saved": inserted_or_updated,
        "indexed_count": total_indexed,
        "sources": source_names,
        "total_documents": len(documents),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Import useful training files into MongoDB and Pinecone.")
    parser.add_argument("folder", nargs="?", default=str(DEFAULT_FOLDER), help="Folder containing useful files")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"Folder not found: {folder}")
        return 1

    _load_env()
    db, knowledge_document_indexer = _get_backend_services()

    try:
        await db.client.admin.command("ping")
    except Exception as exc:
        print(f"MongoDB connection failed: {exc}")
        return 1

    result = await upsert_documents(folder, db=db, knowledge_document_indexer=knowledge_document_indexer)
    if not result.get("success"):
        print(f"Import failed: {result.get('error', 'Unknown error')}")
        return 1

    print(f"Sources indexed: {', '.join(result.get('sources', []))}")
    print(f"Saved chunks: {result.get('chunks_saved', 0)}")
    print(f"Indexed vectors: {result.get('indexed_count', 0)}")
    print(f"Total documents in MongoDB: {result.get('total_documents', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
