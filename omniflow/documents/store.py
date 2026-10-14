"""In-memory, per-session vector index.

Each document session has its own index, keyed by a random server-generated id. Nothing
is written to disk: documents and vectors live in this process's memory until the
session is cleared, expires (idle TTL), is evicted (session cap), or the app restarts.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from omniflow.documents.chunking import Chunk

_SESSION_ID = re.compile(r"^[0-9a-f]{32}$")


def is_valid_session_id(session_id: str | None) -> bool:
    return bool(session_id) and bool(_SESSION_ID.match(session_id))


@dataclass
class DocumentRecord:
    doc_id: str
    filename: str  # display label only
    sha256: str
    page_count: int
    text_pages: int
    ocr_pages: list[int]
    image_only_pages: list[int]
    empty_pages: list[int]
    chunk_count: int
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


class SessionIndex:
    def __init__(self) -> None:
        self.documents: dict[str, DocumentRecord] = {}
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def find_by_hash(self, sha256: str) -> DocumentRecord | None:
        return next((d for d in self.documents.values() if d.sha256 == sha256), None)

    def add(self, record: DocumentRecord, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        self.documents[record.doc_id] = record
        self.chunks.extend(chunks)
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])

    def remove(self, doc_id: str) -> bool:
        if self.documents.pop(doc_id, None) is None:
            return False
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        self.chunks = [self.chunks[i] for i in keep]
        self.vectors = self.vectors[keep] if keep and self.vectors is not None else None
        return True

    def search(self, query: np.ndarray, k: int, min_score: float) -> list[SearchHit]:
        if self.vectors is None or not len(self.chunks):
            return []
        scores = self.vectors @ query  # cosine similarity: both sides are L2-normalised
        order = np.argsort(-scores)[:k]
        return [SearchHit(self.chunks[i], float(scores[i])) for i in order
                if scores[i] >= min_score]


class DocumentStore:
    """Thread-safe map of session id -> SessionIndex with idle expiry and a session cap."""

    def __init__(
        self,
        *,
        max_sessions: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sessions: dict[str, SessionIndex] = {}
        self._last_used: dict[str, float] = {}
        self._lock = threading.RLock()
        self._max = max_sessions
        self._ttl = ttl_seconds
        self._clock = clock

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def _prune(self, keep: str | None = None) -> None:
        now = self._clock()
        # Expiry applies to every session, including the one being requested.
        for sid in [s for s, t in self._last_used.items() if now - t > self._ttl]:
            self._drop(sid)
        while len(self._sessions) > self._max:
            oldest = min((s for s in self._sessions if s != keep),
                         key=self._last_used.__getitem__, default=None)
            if oldest is None:
                break
            self._drop(oldest)

    def _drop(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._last_used.pop(session_id, None)

    def get(self, session_id: str | None, *, create: bool = False) -> SessionIndex | None:
        if not is_valid_session_id(session_id):
            return None
        with self._lock:
            self._prune(keep=session_id)
            index = self._sessions.get(session_id)
            if index is None and create:
                index = self._sessions[session_id] = SessionIndex()
            if index is not None:
                self._last_used[session_id] = self._clock()
                self._prune(keep=session_id)
            return index

    def clear(self, session_id: str | None) -> None:
        with self._lock:
            if is_valid_session_id(session_id):
                self._drop(session_id)

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)
