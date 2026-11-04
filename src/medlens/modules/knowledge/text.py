"""Document text: extraction (pypdf for PDFs, UTF-8 for text and Markdown), normalisation, and a
structure-aware chunker that keeps page numbers and the nearest heading.

Chunks follow paragraphs and headings up to about ``max_tokens`` tokens, never cross a page,
and repeat the last ``overlap`` tokens of the previous chunk so a fact split across a boundary
is still found. Token counts are estimates (words and punctuation), close enough for sizing.
"""

import io
import re
import unicodedata
from dataclasses import dataclass

from pypdf import PdfReader

PDF = "application/pdf"
TEXT_TYPES = frozenset({"text/plain", "text/markdown", "text/x-markdown"})

_TOKEN = re.compile(r"\w+|[^\w\s]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class UnreadableDocument(Exception):
    """Not text we can extract (corrupt PDF, scanned pages without a text layer, bad UTF-8)."""


@dataclass(frozen=True, slots=True)
class Page:
    number: int | None  # 1-based for PDFs; None for text files
    text: str


@dataclass(frozen=True, slots=True)
class TextChunk:
    ordinal: int
    page: int | None
    section: str | None
    text: str
    token_count: int


def extract_pages(data: bytes, media_type: str) -> list[Page]:
    if media_type == PDF:
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = [
                Page(number, normalise(page.extract_text() or ""))
                for number, page in enumerate(reader.pages, start=1)
            ]
        except Exception as exc:  # noqa: BLE001 - pypdf raises many types on bad input
            raise UnreadableDocument(type(exc).__name__) from None
    else:
        try:
            pages = [Page(None, normalise(data.decode("utf-8")))]
        except UnicodeDecodeError:
            raise UnreadableDocument("not UTF-8") from None
    if not any(page.text for page in pages):
        raise UnreadableDocument("no text (a scanned document needs OCR)")
    return pages


def normalise(text: str) -> str:
    """NFKC, no control characters, single spaces within lines, at most one blank line."""
    text = _CONTROL.sub("", unicodedata.normalize("NFKC", text))
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def count_tokens(text: str) -> int:
    return len(_TOKEN.findall(text))


def _is_heading(block: str) -> bool:
    if block.startswith("#"):
        return True
    line = block.strip()
    return (
        "\n" not in line
        and 0 < len(line) <= 80
        and not line.endswith((".", ",", ";", ":"))
        and (line.isupper() or line.istitle())
    )


def _pieces(block: str, max_tokens: int) -> list[str]:
    """A paragraph, split by sentences and then by words if it is too long for one chunk."""
    if count_tokens(block) <= max_tokens:
        return [block]
    pieces: list[str] = []
    current: list[str] = []
    for sentence in _SENTENCE.split(block):
        words = sentence.split()
        while words:
            room = max_tokens - count_tokens(" ".join(current))
            take: list[str] = []
            while words and count_tokens(" ".join([*take, words[0]])) <= max(room, 1):
                take.append(words.pop(0))
            if not take and not current:
                take.append(words.pop(0))  # a single over-long word
            if take:
                current.extend(take)
            if words:
                pieces.append(" ".join(current))
                current = []
    if current:
        pieces.append(" ".join(current))
    return pieces


def _tail(text: str, tokens: int) -> str:
    words = text.split()
    tail: list[str] = []
    while words and count_tokens(" ".join([words[-1], *tail])) <= tokens:
        tail.insert(0, words.pop())
    return " ".join(tail)


def chunk(pages: list[Page], *, max_tokens: int = 400, overlap: int = 60) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    section: str | None = None

    def emit(page: int | None, body: list[str]) -> None:
        text = "\n\n".join(body).strip()
        if text:
            chunks.append(TextChunk(len(chunks), page, section, text, count_tokens(text)))

    for page in pages:
        body: list[str] = []
        for block in (part.strip() for part in page.text.split("\n\n")):
            if not block:
                continue
            if _is_heading(block):
                if any(not _is_heading(part) for part in body):
                    emit(page.number, body)
                    body = []
                section = block.lstrip("#").strip()[:300]
                body.append(block)
                continue
            for piece in _pieces(block, max(max_tokens - overlap, 1)):
                if body and count_tokens("\n\n".join([*body, piece])) > max_tokens:
                    previous = "\n\n".join(body)
                    emit(page.number, body)
                    room = min(overlap, max_tokens - count_tokens(piece) - 1)
                    carried = _tail(previous, room) if room > 0 else ""
                    body = [carried] if carried else []
                body.append(piece)
        emit(page.number, body)
    return chunks
