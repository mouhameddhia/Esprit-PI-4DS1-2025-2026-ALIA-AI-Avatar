"""Load pharmaceutical/scientific PDFs into raw text records."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET
import zipfile

try:
    import fitz
except ImportError:  # pragma: no cover - optional PDF dependency
    fitz = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional imaging dependency
    Image = None

try:
    import pytesseract
except ImportError:  # pragma: no cover - optional OCR dependency
    pytesseract = None

from app.schemas.models import DocumentChunk


class PDFLoader:
    """Read all files in a directory tree and extract text records."""

    def __init__(self, data_dir: str, max_workers: int = 4) -> None:
        """Initialize loader with a source directory path."""

        self.data_dir = Path(data_dir)
        self.max_workers = max(1, max_workers)

    def load(self) -> list[DocumentChunk]:
        """Return extracted documents from every file under data_dir recursively."""

        documents: list[DocumentChunk] = []
        if not self.data_dir.exists():
            return documents

        file_paths = self._iter_file_paths()
        if self.max_workers == 1:
            extracted = [self._extract_with_metadata(file_path) for file_path in file_paths]
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                extracted = list(executor.map(self._extract_with_metadata, file_paths))

        for file_path, relative_source, source_key, file_chunks in extracted:
            for idx, text in enumerate(file_chunks, start=1):
                if not text.strip():
                    continue
                documents.append(
                    DocumentChunk(
                        doc_id=f"{source_key}_p{idx}",
                        text=text,
                        metadata={
                            "source": relative_source,
                            "page": idx,
                            "file_type": file_path.suffix.lower().lstrip("."),
                            "domain": "pharma",
                            "doc_date": self._doc_date(file_path),
                            "source_tier": self._source_tier(file_path),
                        },
                    )
                )

        return documents

    def _extract_with_metadata(self, file_path: Path) -> tuple[Path, str, str, list[str]]:
        """Extract chunks and precomputed metadata keys for one file path."""

        relative_source = str(file_path.relative_to(self.data_dir)).replace("\\", "/")
        source_key = relative_source.rsplit(".", 1)[0].replace("/", "_")
        file_chunks = self._extract_file_chunks(file_path)
        return file_path, relative_source, source_key, file_chunks

    def build_source_manifest(self) -> dict[str, dict[str, int]]:
        """Return a stable manifest describing all source files under data_dir."""

        if not self.data_dir.exists():
            return {}

        manifest: dict[str, dict[str, int]] = {}
        for file_path in self._iter_file_paths():
            relative_source = str(file_path.relative_to(self.data_dir)).replace("\\", "/")
            manifest[relative_source] = self._file_signature(file_path)
        return manifest

    @staticmethod
    def _file_signature(file_path: Path) -> dict[str, int]:
        """Return a lightweight signature for change detection."""

        stat = file_path.stat()
        return {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    def _iter_file_paths(self) -> list[Path]:
        """Return all files recursively under data_dir."""

        return sorted(path for path in self.data_dir.rglob("*") if path.is_file())

    def _extract_file_chunks(self, file_path: Path) -> list[str]:
        """Extract text chunks from a single file based on file type."""

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf_chunks(file_path)
        if suffix in {".pptx", ".pptm"}:
            return self._extract_pptx_chunks(file_path)
        if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
            image_text = self._ocr_image_file(file_path)
            return [image_text] if image_text else []
        if suffix == ".ppt":
            return []
        text = self._read_text_file(file_path)
        return [text] if text else []

    def _extract_pdf_chunks(self, pdf_path: Path) -> list[str]:
        """Extract text from PDF pages with OCR fallback for scanned pages."""

        if fitz is None:
            text = self._read_text_file(pdf_path)
            return [text] if text else []

        chunks: list[str] = []
        doc = fitz.open(pdf_path)
        try:
            for page in doc:
                text = page.get_text("text").strip()
                if not text:
                    text = self._ocr_page(page)
                if text:
                    chunks.append(text)
        finally:
            doc.close()

        return chunks

    def _extract_pptx_chunks(self, pptx_path: Path) -> list[str]:
        """Extract text from PPTX slides by reading the OOXML slide XML directly."""

        chunks: list[str] = []
        namespaces = {
            "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
            "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
        }

        try:
            with zipfile.ZipFile(pptx_path) as archive:
                slide_names = sorted(
                    name for name in archive.namelist() if name.startswith("ppt/slides/slide") and name.endswith(".xml")
                )
                for slide_name in slide_names:
                    slide_xml = archive.read(slide_name)
                    root = ET.fromstring(slide_xml)
                    paragraphs: list[str] = []
                    for paragraph in root.findall(".//a:p", namespaces):
                        runs = [node.text for node in paragraph.findall(".//a:t", namespaces) if node.text]
                        text = "".join(runs).strip()
                        if text:
                            paragraphs.append(text)
                    if paragraphs:
                        chunks.append("\n".join(paragraphs))
        except Exception:
            return []

        return chunks

    @staticmethod
    def _read_text_file(file_path: Path) -> str:
        """Read text from arbitrary file types using robust encoding fallbacks."""

        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                content = file_path.read_text(encoding=encoding, errors="ignore")
                if content.strip():
                    return content
            except Exception:
                continue
        return ""

    def _ocr_image_file(self, image_path: Path) -> str:
        """Run OCR directly on image files when OCR dependencies are available."""

        if Image is None or pytesseract is None:
            return ""
        try:
            with Image.open(image_path) as image:
                return pytesseract.image_to_string(image).strip()
        except Exception:
            return ""

    def _ocr_page(self, page) -> str:
        """Extract text from a scanned page using OCR fallback."""

        if Image is None or pytesseract is None or fitz is None:
            return ""

        try:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.open(BytesIO(pixmap.tobytes("png")))
            return pytesseract.image_to_string(image).strip()
        except Exception:
            return ""

    @staticmethod
    def _doc_date(pdf_path: Path) -> str:
        """Return the source document date as an ISO 8601 string."""

        modified = datetime.fromtimestamp(pdf_path.stat().st_mtime, tz=timezone.utc)
        return modified.date().isoformat()

    @staticmethod
    def _source_tier(pdf_path: Path) -> int:
        """Infer a rough authority tier from the source filename."""

        name = pdf_path.stem.lower()
        if any(keyword in name for keyword in ["smpc", "ema", "label", "prescribing"]):
            return 1
        if any(keyword in name for keyword in ["trial", "clinical", "study", "pivotal"]):
            return 2
        if any(keyword in name for keyword in ["review", "meta", "systematic"]):
            return 3
        if any(keyword in name for keyword in ["blog", "news", "article"]):
            return 4
        return 3

