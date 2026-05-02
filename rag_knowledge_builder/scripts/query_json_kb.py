"""Query the JSON knowledge base with simple question-style lookups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from rag_knowledge_builder.kb_loader import discover_json_files

TOPIC_ALIASES = {
    "indication": "indications",
    "indications": "indications",
    "composition": "composition",
    "ingredient": "composition",
    "ingredients": "composition",
    "dosage": "dosage",
    "dose": "dosage",
    "dosing": "dosage",
    "age": "age",
    "warnings": "warnings",
    "warning": "warnings",
    "administration": "administration",
    "route": "administration",
    "usage": "administration",
    "mechanism": "mechanism_of_action",
    "mechanism_of_action": "mechanism_of_action",
    "side_effects": "side_effects",
    "side effect": "side_effects",
    "side effects": "side_effects",
}


# Discovery of JSON KB files is delegated to rag_knowledge_builder.kb_loader.discover_json_files


def load_payloads(json_files: list[Path], workspace_root: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in json_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        payload["__source_file"] = str(path.relative_to(workspace_root)).replace("\\", "/")
        payloads.append(payload)
    return payloads


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def infer_topic(question: str) -> str | None:
    q = normalize(question)
    for alias, topic in TOPIC_ALIASES.items():
        if alias in q:
            return topic
    return None


def extract_content(payload: dict[str, Any], topic: str, product: str | None = None) -> Any:
    if "document_name" in payload or "drug_name" in payload:
        value = payload.get(topic)
        if isinstance(value, dict):
            return value
        return value

    if "document" in payload and ("topics" in payload or "products" in payload):
        if product:
            topic_payload = payload.get("products", {}).get(product, {}).get(topic)
            if isinstance(topic_payload, dict):
                return topic_payload.get("content") or topic_payload
            return topic_payload
        topic_payload = payload.get("topics", {}).get(topic)
        if isinstance(topic_payload, dict):
            return topic_payload.get("content") or topic_payload
        return topic_payload

    return None


def match_payloads(payloads: list[dict[str, Any]], document: str | None, drug: str | None) -> list[dict[str, Any]]:
    matches = payloads
    if document:
        doc_query = normalize(document)
        matches = [p for p in matches if doc_query in normalize(str(p.get("document_name") or p.get("document") or ""))]
    if drug:
        drug_query = normalize(drug)
        matches = [p for p in matches if drug_query in normalize(str(p.get("drug_name") or ""))]
    return matches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query structured JSON knowledge")
    parser.add_argument("--question", required=True, help="Natural language question to answer")
    parser.add_argument("--document", default=None, help="Optional document name filter")
    parser.add_argument("--drug", default=None, help="Optional drug name filter")
    parser.add_argument("--topic", default=None, help="Optional explicit topic field")
    parser.add_argument(
        "--paths",
        nargs="*",
        default=["rag_knowledge_builder/scripts"],
        help="Folders/files to scan (knowledge base moved to rag_knowledge_builder/scripts)",
    )
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of hits to show")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[2]
    # Use centralized discovery for KB JSON files
    json_files = discover_json_files(workspace_root, folders=list(args.paths or []))
    payloads = load_payloads(json_files, workspace_root)

    topic = args.topic or infer_topic(args.question)
    filtered = match_payloads(payloads, args.document, args.drug)

    start = perf_counter()
    answers: list[dict[str, Any]] = []
    if topic:
        for payload in filtered:
            value = extract_content(payload, topic)
            if value in (None, "", {}, []):
                continue
            answers.append(
                {
                    "document_name": payload.get("document_name") or payload.get("document"),
                    "drug_name": payload.get("drug_name"),
                    "topic": topic,
                    "value": value,
                    "source_file": payload.get("__source_file"),
                }
            )
    else:
        q = normalize(args.question)
        for payload in filtered:
            haystack = " ".join(
                [
                    str(payload.get("document_name") or payload.get("document") or ""),
                    str(payload.get("drug_name") or ""),
                    json.dumps(payload, ensure_ascii=False),
                ]
            )
            if q not in normalize(haystack):
                continue
            answers.append(
                {
                    "document_name": payload.get("document_name") or payload.get("document"),
                    "drug_name": payload.get("drug_name"),
                    "source_file": payload.get("__source_file"),
                }
            )

    latency_ms = (perf_counter() - start) * 1000.0

    report = {
        "question": args.question,
        "topic": topic,
        "filter": {
            "document": args.document,
            "drug": args.drug,
        },
        "latency_ms": round(latency_ms, 3),
        "hits": answers[: args.limit],
        "hit_count": len(answers),
        "scanned_files": len(json_files),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
