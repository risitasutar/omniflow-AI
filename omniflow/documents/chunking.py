"""Split extracted pages into overlapping chunks. Chunks never cross page boundaries, so
every chunk maps to exactly one source page for citations."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from omniflow.documents.pdf import PageText


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    filename: str
    page: int
    index: int  # position of the chunk within its page
    text: str
    source: str  # "text" or "ocr"


def chunk_pages(
    pages: list[PageText], *, doc_id: str, filename: str, chunk_size: int, overlap: int
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    for page in pages:
        for i, piece in enumerate(splitter.split_text(page.text)):
            piece = piece.strip()
            if piece:
                chunks.append(Chunk(doc_id, filename, page.page, i, piece, page.source))
    return chunks
