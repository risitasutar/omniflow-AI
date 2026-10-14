"""Generic per-session, in-memory storage with idle expiry and a session cap.

Session ids are random server-generated hex strings; anything else is refused, so a
caller can never address another session by guessing or crafting an id.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable

_SESSION_ID = re.compile(r"^[0-9a-f]{32}$")


def is_valid_session_id(session_id: str | None) -> bool:
    return bool(session_id) and bool(_SESSION_ID.match(session_id))


class SessionStore[T]:
    def __init__(
        self,
        factory: Callable[[], T],
        *,
        max_sessions: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        on_drop: Callable[[str, T], None] | None = None,
    ) -> None:
        self._factory = factory
        self._values: dict[str, T] = {}
        self._last_used: dict[str, float] = {}
        self._lock = threading.RLock()
        self._max = max_sessions
        self._ttl = ttl_seconds
        self._clock = clock
        self._on_drop = on_drop  # cleanup hook (e.g. delete a session's temp file)

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def _drop(self, session_id: str) -> None:
        value = self._values.pop(session_id, None)
        self._last_used.pop(session_id, None)
        if value is not None and self._on_drop is not None:
            self._on_drop(session_id, value)

    def _prune(self, keep: str | None) -> None:
        now = self._clock()
        for sid in [s for s, t in self._last_used.items() if now - t > self._ttl]:
            self._drop(sid)
        while len(self._values) > self._max:
            oldest = min((s for s in self._values if s != keep),
                         key=self._last_used.__getitem__, default=None)
            if oldest is None:
                break
            self._drop(oldest)

    def get(self, session_id: str | None, *, create: bool = False) -> T | None:
        if not is_valid_session_id(session_id):
            return None
        with self._lock:
            self._prune(keep=session_id)
            value = self._values.get(session_id)
            if value is None and create:
                value = self._values[session_id] = self._factory()
            if value is not None:
                self._last_used[session_id] = self._clock()
                self._prune(keep=session_id)
            return value

    def clear(self, session_id: str | None) -> None:
        with self._lock:
            if is_valid_session_id(session_id):
                self._drop(session_id)

    def session_count(self) -> int:
        with self._lock:
            return len(self._values)
