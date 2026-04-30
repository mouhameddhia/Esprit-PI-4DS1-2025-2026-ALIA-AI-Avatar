"""Embedding model provider for dense retrieval."""

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # pragma: no cover - optional embedding dependency
    HuggingFaceEmbeddings = None


def get_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    """Build a sentence-transformers embedding model wrapper."""

    if HuggingFaceEmbeddings is None:
        raise RuntimeError("langchain_huggingface is not installed")
    return HuggingFaceEmbeddings(model_name=model_name)
