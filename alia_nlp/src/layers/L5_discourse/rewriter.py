"""Query rewriter — generates a concise semantic-search query for RAG."""

from typing import Any, Dict


def rewrite(parsed_llm: Dict[str, Any], user_text: str) -> str:
    """
    Return the LLM-generated rewritten query if valid, else fall back to raw text.
    The rewritten query is used by the RAG pipeline for vector search.
    """
    rq = parsed_llm.get("rewritten_query")
    if isinstance(rq, str) and rq.strip():
        return rq.strip()
    return user_text.strip()
