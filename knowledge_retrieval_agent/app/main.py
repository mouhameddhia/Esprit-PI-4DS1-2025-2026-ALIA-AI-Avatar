"""CLI entrypoint for ingestion and query execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import get_settings
from app.evaluation.llm_judge import evaluate_answer_with_llm_judge
from app.evaluation.xai import build_xai_report
from app.ingestion.chunker import DocumentChunker
from app.ingestion.indexer import Indexer
from app.ingestion.loader import PDFLoader
from app.inspection.raw_text import load_raw_documents, save_raw_documents, search_raw_documents
from app.pipelines.rag_pipeline import RAGPipeline


def run_ingestion() -> None:
    """Load PDFs, chunk text, and build dense/sparse indexes."""

    settings = get_settings()
    loader = PDFLoader(settings.data_dir, max_workers=settings.ingestion_max_workers)
    chunker = DocumentChunker(settings.chunk_size, settings.chunk_overlap)
    indexer = Indexer(
        settings.vector_store_dir,
        settings.embedding_model,
        embedding_batch_size=settings.embedding_batch_size,
        embedding_min_batch_size=settings.embedding_min_batch_size,
        embedding_max_tokens=settings.embedding_max_tokens,
        min_chunk_chars=settings.min_chunk_chars,
        max_non_printable_ratio=settings.max_non_printable_ratio,
    )

    vector_store_dir = Path(settings.vector_store_dir)
    vector_store_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = vector_store_dir / "ingest_manifest.json"
    raw_cache_path = vector_store_dir / "raw_extracted.jsonl"

    current_manifest = loader.build_source_manifest()
    existing_manifest = {}
    if manifest_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            existing_manifest = {}

    faiss_exists = (vector_store_dir / "faiss").exists()
    bm25_exists = (vector_store_dir / "bm25.pkl").exists()
    if faiss_exists and bm25_exists and existing_manifest == current_manifest:
        print(json.dumps({"status": "skipped", "reason": "no_data_changes", "files": len(current_manifest)}))
        return

    raw_docs = loader.load()
    save_raw_documents(raw_cache_path, raw_docs)
    chunks = chunker.chunk(raw_docs)
    indexer.build_all(chunks)
    manifest_path.write_text(json.dumps(current_manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({"status": "ok", "raw_docs": len(raw_docs), "chunks": len(chunks)}))


def run_inspect_raw(query: str, top_k: int) -> None:
    """Search the raw extracted corpus before embedding/retrieval."""

    settings = get_settings()
    vector_store_dir = Path(settings.vector_store_dir)
    raw_cache_path = vector_store_dir / "raw_extracted.jsonl"

    documents = load_raw_documents(raw_cache_path)
    if not documents:
        loader = PDFLoader(settings.data_dir, max_workers=settings.ingestion_max_workers)
        documents = loader.load()

    matches = search_raw_documents(documents, query=query, top_k=top_k)
    payload = []
    for document in matches:
        payload.append(
            {
                "doc_id": document.doc_id,
                "source": document.metadata.get("source"),
                "page": document.metadata.get("page"),
                "text_preview": document.text[:400],
            }
        )

    print(json.dumps({"query": query, "results": payload, "result_count": len(payload)}, indent=2))


def run_query(query: str) -> None:
    """Execute the full retrieval-generation pipeline for one query."""

    settings = get_settings()
    pipeline = RAGPipeline(settings)
    response = pipeline.run(query)
    print(response.model_dump_json(indent=2))


def run_judge(mode: str, question: str, answer: str, context: str, model: str | None = None) -> None:
    """Run strict LLM-as-judge evaluation for one QA/context sample."""

    settings = get_settings()
    result = evaluate_answer_with_llm_judge(
        mode=mode,
        question=question,
        answer=answer,
        context=context,
        base_url=settings.llm_base_url,
        model=model or settings.llm_judge_model,
        timeout_seconds=settings.request_timeout_seconds,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def run_xai_report(
    *,
    question: str,
    answer: str,
    context: str,
    crag_decision: str | None = None,
    judge_scores_json: str | None = None,
    judge_scores_file: str | None = None,
) -> None:
    """Generate strict XAI report for RAG + Judge decision traceability."""

    judge_scores: dict[str, float] | None = None
    if judge_scores_file:
        payload = Path(judge_scores_file).read_text(encoding="utf-8-sig")
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            judge_scores = parsed
    elif judge_scores_json:
        parsed = json.loads(judge_scores_json)
        if isinstance(parsed, dict):
            judge_scores = parsed

    report = build_xai_report(
        question=question,
        answer=answer,
        context=context,
        judge_scores=judge_scores,
        crag_decision=crag_decision,
    )
    print(report)


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""

    parser = argparse.ArgumentParser(description="Knowledge Retrieval Agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Ingest PDFs and build retrieval indexes")

    query_parser = subparsers.add_parser("query", help="Run retrieval pipeline for a question")
    query_parser.add_argument("--text", required=True, help="User query text")

    inspect_parser = subparsers.add_parser("inspect-raw", help="Search raw extracted text before embedding")
    inspect_parser.add_argument("--text", required=True, help="Search text")
    inspect_parser.add_argument("--top-k", required=False, type=int, default=5, help="Number of results")

    judge_parser = subparsers.add_parser("judge", help="Evaluate one answer with LLM-as-judge")
    judge_parser.add_argument("--mode", required=True, choices=["commercial", "training"], help="Evaluation mode")
    judge_parser.add_argument("--question", required=True, help="Original user question")
    judge_parser.add_argument("--answer", required=True, help="Answer to evaluate")
    judge_parser.add_argument("--context", required=True, help="Retrieved context used for the answer")
    judge_parser.add_argument("--model", required=False, default=None, help="Override judge model name")

    xai_parser = subparsers.add_parser("xai-report", help="Generate strict explainability report for one QA case")
    xai_parser.add_argument("--question", required=True, help="Original user question")
    xai_parser.add_argument("--answer", required=True, help="Generated answer")
    xai_parser.add_argument("--context", required=True, help="Retrieved context/chunks")
    xai_parser.add_argument("--crag-decision", required=False, default="UNKNOWN", help="CRAG decision: CORRECT, AMBIGUOUS, INCORRECT")
    xai_parser.add_argument(
        "--judge-scores-json",
        required=False,
        default=None,
        help="Optional judge scores JSON object string",
    )
    xai_parser.add_argument(
        "--judge-scores-file",
        required=False,
        default=None,
        help="Optional path to JSON file containing judge scores object",
    )

    return parser


def main() -> None:
    """Handle CLI command dispatch."""

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        run_ingestion()
        return

    if args.command == "query":
        run_query(args.text)

    if args.command == "judge":
        run_judge(
            mode=args.mode,
            question=args.question,
            answer=args.answer,
            context=args.context,
            model=args.model,
        )

    if args.command == "xai-report":
        run_xai_report(
            question=args.question,
            answer=args.answer,
            context=args.context,
            crag_decision=args.crag_decision,
            judge_scores_json=args.judge_scores_json,
            judge_scores_file=args.judge_scores_file,
        )

    if args.command == "inspect-raw":
        run_inspect_raw(query=args.text, top_k=args.top_k)


if __name__ == "__main__":
    main()
