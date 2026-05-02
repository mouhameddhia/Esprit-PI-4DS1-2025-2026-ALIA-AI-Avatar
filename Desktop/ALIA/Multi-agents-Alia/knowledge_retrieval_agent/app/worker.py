"""Long-lived worker for low-latency RAG queries."""

from __future__ import annotations

import json
import io
import sys

from app.config import get_settings
from app.pipelines.rag_pipeline import RAGPipeline
from app.utils.logger import get_logger


def main() -> None:
    """Load the pipeline once and serve query requests from stdin."""

    # Ensure stdout/stdin use UTF-8 encoding for cross-platform compatibility
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    if sys.stdin.encoding.lower() != 'utf-8':
        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

    logger = get_logger("knowledge_retrieval_agent.worker")
    settings = get_settings()
    pipeline = RAGPipeline(settings)

    print("READY\t{\"status\":\"ready\"}", flush=True)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        if line in {"quit", "exit"}:
            print("RESULT\t{\"status\":\"stopped\"}", flush=True)
            return

        try:
            request = json.loads(line)
            query = str(request.get("text", "")).strip()
            if not query:
                raise ValueError("query text is empty")

            response = pipeline.run(query)
            print(f"RESULT\t{response.model_dump_json()}", flush=True)
        except Exception as exc:
            logger.exception("worker_request_failed")
            error_payload = json.dumps({"error": str(exc)}, ensure_ascii=False)
            print(f"ERROR\t{error_payload}", flush=True)


if __name__ == "__main__":
    main()