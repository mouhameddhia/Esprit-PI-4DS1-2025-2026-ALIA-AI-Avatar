"""Import a PDF into MongoDB and Pinecone as searchable knowledge chunks."""

import argparse
import asyncio
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv(dotenv_path=REPO_ROOT / "backend" / ".env")

from backend.main import db, knowledge_document_indexer  # noqa: E402


def _utc_now():
    return datetime.now(timezone.utc)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    if len(normalized) <= max_chars:
        return [normalized]

    chunks: list[str] = []
    start = 0
    text_length = len(normalized)

    while start < text_length:
        end = min(text_length, start + max_chars)
        if end < text_length:
            boundary = normalized.rfind(". ", start, end)
            if boundary > start + 200:
                end = boundary + 1

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(end - overlap, start + 1)

    return chunks


def extract_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []

    for page_index, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        pages.append((page_index, text))

    return pages


async def upsert_documents(pdf_path: Path, source_name: str, language: str) -> dict:
    pdf_pages = extract_pdf_pages(pdf_path)
    if not any(page_text for _, page_text in pdf_pages):
        return {
            "success": False,
            "error": "No text could be extracted from the PDF. If it is scanned, OCR is required.",
        }

    now = _utc_now()
    inserted_or_updated = 0

    for page_number, page_text in pdf_pages:
        if not page_text:
            continue

        for chunk_index, chunk in enumerate(chunk_text(page_text), start=1):
            document = {
                "source_name": source_name,
                "source_path": str(pdf_path),
                "language": language,
                "title": source_name,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "chunk_text": chunk,
                "keywords_text": "",
                "created_at": now,
                "updated_at": now,
            }

            existing = await db.knowledge_documents.find_one(
                {
                    "source_name": source_name,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                },
                {"_id": 1, "created_at": 1},
            )

            if existing:
                await db.knowledge_documents.update_one(
                    {"_id": existing["_id"]},
                    {
                        "$set": {
                            **document,
                            "created_at": existing.get("created_at", now),
                        }
                    },
                )
            else:
                await db.knowledge_documents.insert_one(document)

            inserted_or_updated += 1

    index_result = await knowledge_document_indexer.index_documents(db, source_name=source_name)
    if not index_result.get("success"):
        return index_result

    index_result["chunks_saved"] = inserted_or_updated
    index_result["source_name"] = source_name
    return index_result


async def main() -> int:
    parser = argparse.ArgumentParser(description="Import a PDF into MongoDB and Pinecone.")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--source-name", help="Display name for the source", default=None)
    parser.add_argument("--language", help="Language code for the source", default="fr")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path).expanduser().resolve()
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return 1

    source_name = args.source_name or pdf_path.stem

    print(f"Importing PDF: {pdf_path}")
    print(f"Source name: {source_name}")

    try:
        await db.client.admin.command("ping")
    except Exception as exc:
        print(f"MongoDB connection failed: {exc}")
        return 1

    result = await upsert_documents(pdf_path, source_name, args.language)
    if not result.get("success"):
        print(f"Import failed: {result.get('error', 'Unknown error')}")
        return 1

    print(f"Saved chunks: {result.get('chunks_saved', 0)}")
    print(f"Indexed vectors: {result.get('indexed_count', 0)}")
    print(f"Total documents in MongoDB: {result.get('total_documents', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))