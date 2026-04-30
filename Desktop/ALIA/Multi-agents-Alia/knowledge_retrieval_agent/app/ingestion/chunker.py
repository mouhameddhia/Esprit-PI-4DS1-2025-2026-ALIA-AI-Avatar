"""Chunk source documents for retrieval."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.schemas.models import DocumentChunk


class DocumentChunker:
    """Split page-level docs into chunk-level docs with overlap."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        """Create a chunker with user-configured splitting settings."""

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, documents: list[DocumentChunk]) -> list[DocumentChunk]:
        """Return chunked documents preserving source metadata."""

        chunked: list[DocumentChunk] = []
        for doc in documents:
            parts = self.splitter.split_text(doc.text)
            for idx, part in enumerate(parts):
                chunked.append(
                    DocumentChunk(
                        doc_id=f"{doc.doc_id}_c{idx + 1}",
                        text=part,
                        metadata={**doc.metadata, "chunk": idx + 1},
                    )
                )
        return chunked
