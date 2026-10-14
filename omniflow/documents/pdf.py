"""PDF validation and page-by-page text extraction.

Uploads are processed entirely in memory - nothing is written to disk, and the uploaded
filename is only ever used as a display label, never as a path.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field

from pypdf import PdfReader
from pypdf.errors import DependencyError, PdfReadError

from omniflow.documents.ocr import OcrEngine

# pypdf logs warnings about malformed files at WARNING; keep them out of the app log.
logging.getLogger("pypdf").setLevel(logging.ERROR)

PDF_MAGIC = b"%PDF-"
MIN_PAGE_CHARS = 20  # below this a page is treated as having no usable text
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n\s*\n+")


class DocumentError(Exception):
    """A document problem with a safe, user-facing message."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.safe_message = message


@dataclass(frozen=True)
class PageText:
    page: int  # 1-based PDF page number
    text: str
    source: str  # "text" or "ocr"


@dataclass
class Extraction:
    page_count: int
    pages: list[PageText] = field(default_factory=list)
    ocr_pages: list[int] = field(default_factory=list)
    image_only_pages: list[int] = field(default_factory=list)  # images but no text, not OCR'd
    empty_pages: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)


def safe_display_name(name: str | None) -> str:
    """A display label from an untrusted filename: no directories, no control chars."""
    base = re.split(r"[\\/]", name or "")[-1]
    base = _SPACES.sub(" ", _CONTROL.sub("", base)).strip().strip(".")
    if len(base) > 120:
        stem, dot, ext = base.rpartition(".")
        base = (stem[:110] + "…" + dot + ext) if dot else base[:120]
    return base or "document.pdf"


def validate_upload(name: str | None, data: bytes, max_bytes: int) -> None:
    if not (name or "").lower().endswith(".pdf"):
        raise DocumentError("unsupported_type", "Only PDF files (.pdf) are supported.")
    if not data:
        raise DocumentError("empty_file", "The file is empty.")
    if len(data) > max_bytes:
        raise DocumentError(
            "too_large", f"The file is larger than the {max_bytes / 1_000_000:g} MB limit."
        )
    if PDF_MAGIC not in data[:1024]:
        raise DocumentError("unsupported_type", "The file is not a valid PDF.")


def clean_text(text: str) -> str:
    text = _CONTROL.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = _SPACES.sub(" ", text)
    return _BLANK_LINES.sub("\n\n", text).strip()


def _open(data: bytes) -> PdfReader:
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
    except (PdfReadError, ValueError, TypeError, KeyError, IndexError, OSError) as exc:
        raise DocumentError("malformed", "The PDF could not be read; it may be damaged.") from exc
    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")  # some PDFs are "encrypted" with an empty password
        except (DependencyError, NotImplementedError, PdfReadError, ValueError) as exc:
            raise DocumentError(
                "encrypted", "The PDF is password-protected or encrypted and can't be read."
            ) from exc
        if not unlocked:
            raise DocumentError(
                "encrypted", "The PDF is password-protected. Remove the password and re-upload."
            )
    return reader


def _has_images(page) -> bool:
    try:
        return len(page.images) > 0
    except Exception:  # noqa: BLE001 - damaged image streams: treat as no images
        return False


def extract_pdf(
    data: bytes,
    *,
    max_pages: int,
    max_chars: int,
    ocr: OcrEngine | None = None,
    ocr_max_pages: int = 0,
) -> Extraction:
    reader = _open(data)
    try:
        page_count = len(reader.pages)
    except Exception as exc:  # noqa: BLE001
        raise DocumentError("malformed", "The PDF's page structure could not be read.") from exc
    if page_count == 0:
        raise DocumentError("empty_document", "The PDF has no pages.")
    if page_count > max_pages:
        raise DocumentError(
            "too_many_pages", f"The PDF has {page_count} pages; the limit is {max_pages}."
        )

    result = Extraction(page_count=page_count)
    total = 0
    skipped_for_ocr_limit = 0
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = clean_text(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - one bad page must not sink the document
            text = ""
            result.warnings.append(f"Page {number}: text could not be extracted.")
        source = "text"
        if len(text) < MIN_PAGE_CHARS:
            if not _has_images(page):
                result.empty_pages.append(number)
                continue
            if ocr is None:
                result.image_only_pages.append(number)
                continue
            if len(result.ocr_pages) >= ocr_max_pages:
                skipped_for_ocr_limit += 1
                result.image_only_pages.append(number)
                continue
            text = clean_text(ocr.page_text(page))
            source = "ocr"
            if len(text) < MIN_PAGE_CHARS:
                result.image_only_pages.append(number)
                continue
            result.ocr_pages.append(number)
        total += len(text)
        if total > max_chars:
            raise DocumentError(
                "too_much_text",
                f"The PDF contains more than {max_chars:,} characters of text; "
                "split it into smaller files.",
            )
        result.pages.append(PageText(page=number, text=text, source=source))
    if skipped_for_ocr_limit:
        result.warnings.append(
            f"OCR is limited to {ocr_max_pages} pages per file; "
            f"{skipped_for_ocr_limit} image-only page(s) were not read."
        )
    return result
