"""Document parsing and chunking for the RAG pipeline."""

import io
import re
from dataclasses import dataclass
from html.parser import HTMLParser

from pypdf import PdfReader

SUPPORTED_TYPES = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".csv": "text/csv",
    ".json": "application/json",
    ".html": "text/html",
    ".htm": "text/html",
}


class IngestError(ValueError):
    """The uploaded file cannot be ingested. Message is safe to show to users."""


@dataclass
class Page:
    number: int
    text: str


@dataclass
class ChunkDraft:
    ordinal: int
    page: int
    text: str


def safe_filename(name: str) -> str:
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .")
    return (base or "document")[:200]


def extension(name: str) -> str:
    m = re.search(r"\.[A-Za-z0-9]+$", name)
    return m.group(0).lower() if m else ""


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):  # noqa: ARG002
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in ("p", "br", "div", "li", "h1", "h2", "h3", "h4", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def parse(filename: str, data: bytes) -> list[Page]:
    ext = extension(filename)
    if ext not in SUPPORTED_TYPES:
        allowed = ", ".join(sorted(SUPPORTED_TYPES))
        raise IngestError(f"Unsupported file type '{ext or 'none'}'. Allowed: {allowed}")
    if ext == ".pdf":
        if not data.startswith(b"%PDF"):
            raise IngestError("File has a .pdf extension but is not a PDF.")
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise IngestError("Encrypted PDFs are not supported.")
            pages = [Page(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
        except IngestError:
            raise
        except Exception as exc:  # pypdf raises many exception types for malformed files
            raise IngestError(f"Could not read PDF: {type(exc).__name__}") from exc
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1")
        if ext in (".html", ".htm"):
            ex = _TextExtractor()
            ex.feed(text)
            text = "".join(ex.parts)
        pages = [Page(1, text)]
    pages = [Page(p.number, _normalize(p.text)) for p in pages]
    if not any(p.text for p in pages):
        raise IngestError("No extractable text found (scanned PDFs need OCR first).")
    return pages


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_pages(pages: list[Page], size: int = 900, overlap: int = 150) -> list[ChunkDraft]:
    """Split text into ~size-character chunks on paragraph/sentence boundaries with overlap.

    Chunks never span pages, so every citation points at a single page.
    """
    chunks: list[ChunkDraft] = []
    for page in pages:
        units = _split_units(page.text, size)
        current = ""
        for unit in units:
            if current and len(current) + len(unit) + 1 > size:
                chunks.append(ChunkDraft(len(chunks), page.number, current.strip()))
                tail = current[-overlap:] if overlap else ""
                # start the overlap at a word boundary
                current = tail[tail.find(" ") + 1 :] if " " in tail else tail
            current = f"{current} {unit}".strip() if current else unit
        if current.strip():
            chunks.append(ChunkDraft(len(chunks), page.number, current.strip()))
    return [c for c in chunks if len(c.text) >= 20]


def _split_units(text: str, size: int) -> list[str]:
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            units.append(para)
            continue
        for sent in re.split(r"(?<=[.!?])\s+", para):
            while len(sent) > size:  # very long sentence or table row
                units.append(sent[:size])
                sent = sent[size:]
            if sent:
                units.append(sent)
    return units
