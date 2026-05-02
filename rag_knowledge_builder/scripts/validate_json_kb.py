"""Validate and benchmark the JSON knowledge base.

This script performs two phases:
1) JSON integrity checks across all files.
2) Retrieval benchmarks by drug name, document name, and topic field.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_knowledge_builder.kb_loader import discover_json_files


TOPIC_FIELDS = [
    "indications",
    "composition",
    "mechanism_of_action",
    "age",
    "dosage",
    "administration",
    "warnings",
    "side_effects",
]


@dataclass
class KnowledgeDoc:
    source_file: str
    payload: dict[str, Any]
    format_type: str
    document_name: str | None
    drug_name: str | None


# Discovery of JSON KB files is delegated to rag_knowledge_builder.kb_loader.discover_json_files


def detect_format(payload: dict[str, Any]) -> str:
    if "document_name" in payload or "drug_name" in payload:
        return "flat_payload"
    if "document" in payload and ("topics" in payload or "products" in payload):
        return "builder_payload"
    return "unknown"


def extract_document_name(payload: dict[str, Any], format_type: str, default_name: str) -> str:
    if format_type == "flat_payload":
        return str(payload.get("document_name") or default_name)
    if format_type == "builder_payload":
        return str(payload.get("document") or default_name)
    return default_name


def extract_drug_name(payload: dict[str, Any], format_type: str) -> str | None:
    if format_type == "flat_payload":
        value = payload.get("drug_name")
        return str(value) if value else None
    return None


def load_documents(json_files: list[Path], workspace_root: Path) -> tuple[list[KnowledgeDoc], list[dict[str, Any]]]:
    docs: list[KnowledgeDoc] = []
    parse_errors: list[dict[str, Any]] = []

    for path in json_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            parse_errors.append(
                {
                    "file": str(path.relative_to(workspace_root)).replace("\\", "/"),
                    "error": str(exc),
                }
            )
            continue

        if not isinstance(payload, dict):
            parse_errors.append(
                {
                    "file": str(path.relative_to(workspace_root)).replace("\\", "/"),
                    "error": "Top-level JSON is not an object",
                }
            )
            continue

        format_type = detect_format(payload)
        file_label = str(path.relative_to(workspace_root)).replace("\\", "/")
        document_name = extract_document_name(payload, format_type, path.stem)
        drug_name = extract_drug_name(payload, format_type)

        docs.append(
            KnowledgeDoc(
                source_file=file_label,
                payload=payload,
                format_type=format_type,
                document_name=document_name,
                drug_name=drug_name,
            )
        )

    return docs, parse_errors


def topic_exists(doc: KnowledgeDoc, topic: str) -> bool:
    payload = doc.payload
    if doc.format_type == "flat_payload":
        value = payload.get(topic)
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, dict):
            # Consider nested multilingual fields missing when all values are null/empty.
            for nested in value.values():
                if nested is None:
                    continue
                if isinstance(nested, str) and not nested.strip():
                    continue
                return True
            return False
        if isinstance(value, list):
            return len(value) > 0
        return True

    if doc.format_type == "builder_payload":
        topics = payload.get("topics", {})
        if isinstance(topics, dict) and topic in topics:
            topic_payload = topics.get(topic)
            if isinstance(topic_payload, dict):
                content = str(topic_payload.get("content", "")).strip()
                if content:
                    return True

        products = payload.get("products", {})
        if isinstance(products, dict):
            for _, per_product in products.items():
                if not isinstance(per_product, dict):
                    continue
                topic_payload = per_product.get(topic)
                if isinstance(topic_payload, dict):
                    content = str(topic_payload.get("content", "")).strip()
                    if content:
                        return True
        return False

    return False


def query_by_document(docs: list[KnowledgeDoc], query: str) -> list[KnowledgeDoc]:
    q = query.casefold().strip()
    return [d for d in docs if d.document_name and q in d.document_name.casefold()]


def query_by_drug(docs: list[KnowledgeDoc], query: str) -> list[KnowledgeDoc]:
    q = query.casefold().strip()
    return [d for d in docs if d.drug_name and q in d.drug_name.casefold()]


def build_integrity_report(docs: list[KnowledgeDoc], parse_errors: list[dict[str, Any]]) -> dict[str, Any]:
    total_valid = len(docs)
    by_format: dict[str, int] = {}
    required_field_missing = {
        "flat_payload": {"document_name": 0, "drug_name": 0},
        "builder_payload": {"document": 0},
    }

    for doc in docs:
        by_format[doc.format_type] = by_format.get(doc.format_type, 0) + 1
        if doc.format_type == "flat_payload":
            if not doc.payload.get("document_name"):
                required_field_missing["flat_payload"]["document_name"] += 1
            if not doc.payload.get("drug_name"):
                required_field_missing["flat_payload"]["drug_name"] += 1
        elif doc.format_type == "builder_payload":
            if not doc.payload.get("document"):
                required_field_missing["builder_payload"]["document"] += 1

    topic_missing_counts: dict[str, int] = {}
    for topic in TOPIC_FIELDS:
        topic_missing_counts[topic] = sum(1 for doc in docs if not topic_exists(doc, topic))

    return {
        "total_valid_json_files": total_valid,
        "total_invalid_json_files": len(parse_errors),
        "formats": by_format,
        "required_field_missing": required_field_missing,
        "topic_missing_counts": topic_missing_counts,
        "parse_errors": parse_errors,
    }


def retrieval_tests(
    docs: list[KnowledgeDoc],
    document_query: str,
    drug_query: str,
    topic_query: str,
    samples: int,
) -> dict[str, Any]:
    tests: list[dict[str, Any]] = []

    start = time.perf_counter()
    by_doc = query_by_document(docs, document_query)
    latency_doc = (time.perf_counter() - start) * 1000.0
    tests.append(
        {
            "query_type": "document_name",
            "query": document_query,
            "latency_ms": round(latency_doc, 3),
            "result_count": len(by_doc),
            "correctness": len(by_doc) > 0,
            "sample_results": [d.document_name for d in by_doc[:samples]],
        }
    )

    start = time.perf_counter()
    by_drug = query_by_drug(docs, drug_query)
    latency_drug = (time.perf_counter() - start) * 1000.0
    tests.append(
        {
            "query_type": "drug_name",
            "query": drug_query,
            "latency_ms": round(latency_drug, 3),
            "result_count": len(by_drug),
            "correctness": len(by_drug) > 0,
            "sample_results": [d.drug_name for d in by_drug[:samples]],
        }
    )

    start = time.perf_counter()
    docs_with_topic = [d for d in docs if topic_exists(d, topic_query)]
    latency_topic = (time.perf_counter() - start) * 1000.0
    missing_rate = 0.0
    if docs:
        missing_rate = (len(docs) - len(docs_with_topic)) / len(docs)

    tests.append(
        {
            "query_type": "topic_field",
            "query": topic_query,
            "latency_ms": round(latency_topic, 3),
            "result_count": len(docs_with_topic),
            "correctness": len(docs_with_topic) > 0,
            "missing_data_rate": round(missing_rate, 4),
            "sample_results": [d.document_name for d in docs_with_topic[:samples]],
        }
    )

    avg_latency = sum(test["latency_ms"] for test in tests) / len(tests)
    overall_correctness = sum(1 for test in tests if test["correctness"]) / len(tests)

    return {
        "tests": tests,
        "avg_latency_ms": round(avg_latency, 3),
        "overall_correctness_rate": round(overall_correctness, 4),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and benchmark JSON knowledge base")
    parser.add_argument(
        "--paths",
        nargs="*",
        default=["rag_knowledge_builder/scripts"],
        help="Folders/files to scan recursively for JSON knowledge files (moved to rag_knowledge_builder/scripts)",
    )
    parser.add_argument("--document-query", default="PÉDIAKIDS", help="Document name query for retrieval benchmark")
    parser.add_argument("--drug-query", default="PÉDIAKIDS", help="Drug name query for retrieval benchmark")
    parser.add_argument("--topic-query", default="indications", help="Topic field query for retrieval benchmark")
    parser.add_argument("--samples", type=int, default=5, help="Number of example hits to show")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[2]

    # Use centralized discovery; args.paths are relative folders/files under workspace root
    json_files = discover_json_files(workspace_root, folders=list(args.paths or []))
    docs, parse_errors = load_documents(json_files, workspace_root)

    integrity = build_integrity_report(docs, parse_errors)
    retrieval = retrieval_tests(
        docs=docs,
        document_query=args.document_query,
        drug_query=args.drug_query,
        topic_query=args.topic_query,
        samples=args.samples,
    )

    report = {
        "status": "ok",
        "scanned_paths": [str(p.relative_to(workspace_root)).replace("\\", "/") for p in scan_paths if p.exists()],
        "total_json_files_found": len(json_files),
        "integrity": integrity,
        "retrieval": retrieval,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
