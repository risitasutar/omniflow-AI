"""SQLite store for long-term memories, separate from chat history and agent checkpoints.

Every query is parameterised and filtered by `owner_id`, which the app derives from the
access mode (one local owner, or one owner per browser session on a shared deployment).
Deleting memories never touches conversations, checkpoints or uploaded files.

Failures are explicit: if the database cannot be opened or migrated, `MemoryUnavailable` is
raised and the app reports it. There is no silent fallback to in-memory storage.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from omniflow.memory import policy

SCHEMA_VERSION = 1
CATEGORIES = ("preference", "project", "explicit")
SOURCES = ("explicit", "confirmed", "manual")
GLOBAL_SCOPE = "global"

_MIGRATIONS = {
    1: """
        CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            category TEXT NOT NULL CHECK (category IN ('preference', 'project', 'explicit')),
            scope TEXT NOT NULL,
            content TEXT NOT NULL,
            norm TEXT NOT NULL,
            source TEXT NOT NULL CHECK (source IN ('explicit', 'confirmed', 'manual')),
            active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            expires_at REAL,
            UNIQUE (owner_id, scope, norm)
        );
        CREATE INDEX idx_memories_owner ON memories(owner_id, active, scope);
        CREATE TABLE memory_settings (
            owner_id TEXT PRIMARY KEY,
            enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
            active_project TEXT,
            updated_at REAL NOT NULL
        );
    """,
}


class MemoryUnavailable(Exception):
    """Memory storage cannot be used; the message is safe to show."""


class MemoryStoreError(Exception):
    """A memory operation failed; the message is safe to show."""


def project_scope(name: str | None) -> str:
    cleaned = " ".join((name or "").split())[:60]
    return f"project:{cleaned}" if cleaned else GLOBAL_SCOPE


def scope_label(scope: str) -> str:
    return "All conversations" if scope == GLOBAL_SCOPE else f"Project: {scope[8:]}"


@dataclass(frozen=True)
class Memory:
    id: str
    category: str
    scope: str
    content: str
    source: str
    active: bool
    created_at: float
    updated_at: float
    expires_at: float | None


@dataclass(frozen=True)
class SaveResult:
    memory: Memory
    created: bool  # False when an equal or near-equal memory already existed


_COLUMNS = "id, category, scope, content, source, active, created_at, updated_at, expires_at"


def _row(row) -> Memory:
    return Memory(row[0], row[1], row[2], row[3], row[4], bool(row[5]), row[6], row[7], row[8])


class MemoryStore:
    def __init__(self, path: Path, clock=time.time) -> None:
        self.path = Path(path)
        self._clock = clock
        self._lock = threading.RLock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._migrate()
        except MemoryUnavailable:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise MemoryUnavailable(
                "Memory storage could not be opened (the storage location is not writable "
                "or the database is damaged). Nothing will be remembered until this is fixed."
            ) from exc

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _migrate(self) -> None:
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise MemoryUnavailable("The memory database was created by a newer version of "
                                    "OmniFlow, so it was not opened (nothing was changed).")
        for target in range(version + 1, SCHEMA_VERSION + 1):
            with self._conn:  # each migration is all-or-nothing
                self._conn.executescript("BEGIN;" + _MIGRATIONS[target]
                                         + f"PRAGMA user_version = {int(target)};COMMIT;")

    def _write(self, sql: str, params: tuple) -> sqlite3.Cursor:
        try:
            with self._lock, self._conn:
                return self._conn.execute(sql, params)
        except sqlite3.Error as exc:
            raise MemoryStoreError("The memory database could not be updated, so the change was "
                               "not saved. Please try again.") from exc

    def _read(self, sql: str, params: tuple) -> list:
        try:
            with self._lock:
                return self._conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise MemoryStoreError("Saved memories could not be read right now.") from exc

    # ------------------------------------------------------------- settings
    def enabled(self, owner: str) -> bool:
        rows = self._read("SELECT enabled FROM memory_settings WHERE owner_id = ?", (owner,))
        return bool(rows[0][0]) if rows else True

    def set_enabled(self, owner: str, enabled: bool) -> None:
        self._write("INSERT INTO memory_settings (owner_id, enabled, updated_at) VALUES (?,?,?) "
                    "ON CONFLICT(owner_id) DO UPDATE SET enabled = excluded.enabled, "
                    "updated_at = excluded.updated_at", (owner, int(enabled), self._clock()))

    def active_project(self, owner: str) -> str | None:
        rows = self._read("SELECT active_project FROM memory_settings WHERE owner_id = ?",
                          (owner,))
        return rows[0][0] if rows and rows[0][0] else None

    def set_active_project(self, owner: str, name: str | None) -> None:
        cleaned = " ".join((name or "").split())[:60] or None
        self._write("INSERT INTO memory_settings (owner_id, active_project, updated_at) "
                    "VALUES (?,?,?) ON CONFLICT(owner_id) DO UPDATE SET "
                    "active_project = excluded.active_project, updated_at = excluded.updated_at",
                    (owner, cleaned, self._clock()))

    # --------------------------------------------------------------- writes
    def add(self, owner: str, content: str, *, category: str, scope: str = GLOBAL_SCOPE,
            source: str = "explicit", expires_at: float | None = None) -> SaveResult:
        """Save a memory once. Secrets raise MemoryRejected; duplicates return the original."""
        if category not in CATEGORIES or source not in SOURCES:
            raise MemoryStoreError("Unknown memory category or source.")
        text = policy.clean_content(content)
        existing = self.find_duplicate(owner, text, scope)
        if existing is not None:
            if not existing.active:
                self.set_active(owner, existing.id, True)
                existing = self.get(owner, existing.id)
            return SaveResult(existing, False)
        memory_id = uuid.uuid4().hex[:16]
        now = self._clock()
        cursor = self._write(
            "INSERT OR IGNORE INTO memories (id, owner_id, category, scope, content, norm, "
            "source, active, created_at, updated_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)",
            (memory_id, owner, category, scope, text, policy.normalise(text), source, now, now,
             expires_at))
        if cursor.rowcount == 0:  # a concurrent session saved the same memory first
            return SaveResult(self.find_duplicate(owner, text, scope), False)
        return SaveResult(self.get(owner, memory_id), True)

    def update(self, owner: str, memory_id: str, content: str, *,
               category: str | None = None) -> Memory:
        text = policy.clean_content(content)
        current = self.get(owner, memory_id)
        if current is None:
            raise MemoryStoreError("That memory no longer exists.")
        clash = self.find_duplicate(owner, text, current.scope)
        if clash is not None and clash.id != memory_id:
            raise MemoryStoreError("Another saved memory already says this.")
        category = category or current.category
        if category not in CATEGORIES:
            raise MemoryStoreError("Unknown memory category.")
        self._write("UPDATE memories SET content = ?, norm = ?, category = ?, updated_at = ? "
                    "WHERE owner_id = ? AND id = ?",
                    (text, policy.normalise(text), category, self._clock(), owner, memory_id))
        return self.get(owner, memory_id)

    def set_active(self, owner: str, memory_id: str, active: bool) -> bool:
        cursor = self._write("UPDATE memories SET active = ?, updated_at = ? "
                             "WHERE owner_id = ? AND id = ?",
                             (int(active), self._clock(), owner, memory_id))
        return cursor.rowcount == 1

    def delete(self, owner: str, memory_id: str) -> bool:
        cursor = self._write("DELETE FROM memories WHERE owner_id = ? AND id = ?",
                             (owner, memory_id))
        return cursor.rowcount == 1

    def clear(self, owner: str) -> int:
        """Delete every memory of this owner (nothing else)."""
        return self._write("DELETE FROM memories WHERE owner_id = ?", (owner,)).rowcount

    def purge(self, *, owner_prefix: str, older_than: float) -> int:
        """Drop stale records of temporary per-session owners, and expired memories."""
        now = self._clock()
        removed = self._write("DELETE FROM memories WHERE expires_at IS NOT NULL AND "
                              "expires_at <= ?", (now,)).rowcount
        removed += self._write(
            "DELETE FROM memories WHERE substr(owner_id, 1, ?) = ? AND updated_at < ?",
            (len(owner_prefix), owner_prefix, older_than)).rowcount
        self._write("DELETE FROM memory_settings WHERE substr(owner_id, 1, ?) = ? AND "
                    "updated_at < ?", (len(owner_prefix), owner_prefix, older_than))
        return removed

    # ---------------------------------------------------------------- reads
    def get(self, owner: str, memory_id: str) -> Memory | None:
        rows = self._read(f"SELECT {_COLUMNS} FROM memories WHERE owner_id = ? AND id = ?",  # noqa: S608
                          (owner, memory_id))
        return _row(rows[0]) if rows else None

    def list(self, owner: str, *, category: str | None = None, scope: str | None = None,
             active: bool | None = None, include_expired: bool = False) -> list[Memory]:
        sql = f"SELECT {_COLUMNS} FROM memories WHERE owner_id = ?"  # noqa: S608
        params: list = [owner]
        if category is not None:
            sql += " AND category = ?"
            params.append(category)
        if scope is not None:
            sql += " AND scope = ?"
            params.append(scope)
        if active is not None:
            sql += " AND active = ?"
            params.append(int(active))
        if not include_expired:
            sql += " AND (expires_at IS NULL OR expires_at > ?)"
            params.append(self._clock())
        return [_row(r) for r in self._read(sql + " ORDER BY updated_at DESC", tuple(params))]

    def scopes(self, owner: str) -> list[str]:
        rows = self._read("SELECT DISTINCT scope FROM memories WHERE owner_id = ? ORDER BY scope",
                          (owner,))
        return [r[0] for r in rows]

    def find_duplicate(self, owner: str, text: str, scope: str) -> Memory | None:
        for memory in self.list(owner, scope=scope, include_expired=True):
            if policy.is_duplicate(memory.content, text):
                return memory
        return None
