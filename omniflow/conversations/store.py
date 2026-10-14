"""SQLite store for conversations, the messages shown in the UI, and attached images.

This is the *display* history (what the user saw, with sources, tables and charts). The
agent's own memory lives in the LangGraph checkpoint database under the same id - a
conversation id IS its LangGraph thread id - so reopening a conversation restores both.

All queries are parameterised. The schema is versioned with `PRAGMA user_version` and
migrated forward in place. Message text is passed through secret redaction before saving.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omniflow.core.logging_setup import redact

SCHEMA_VERSION = 2
LOCAL_OWNER = "local"
MAX_TITLE = 80
MAX_PAYLOAD_BYTES = 2_000_000
_ID = re.compile(r"^[0-9a-f]{32}$")

_MIGRATIONS = {
    1: """
        CREATE TABLE conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            turn_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            payload TEXT,
            created_at REAL NOT NULL,
            UNIQUE (conversation_id, turn_id, role)
        );
        CREATE INDEX idx_messages_conversation ON messages(conversation_id, id);
        CREATE TABLE images (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            mime TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            data BLOB NOT NULL,
            created_at REAL NOT NULL,
            UNIQUE (conversation_id, sha256)
        );
    """,
    # Owner of each conversation: "local" for a single-user install, or a per-browser-session
    # id on a shared deployment, so visitors never see each other's conversations.
    2: """
        ALTER TABLE conversations ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'local';
        CREATE INDEX idx_conversations_owner ON conversations(owner_id, updated_at);
    """,
}


class StoreError(Exception):
    """A storage problem with a safe, user-facing message."""


def is_conversation_id(value: str | None) -> bool:
    return bool(value) and bool(_ID.match(value))


def new_conversation_id() -> str:
    return uuid.uuid4().hex


def make_title(text: str) -> str:
    title = re.sub(r"\s+", " ", redact(text or "")).strip()
    if len(title) > MAX_TITLE:
        title = title[: MAX_TITLE - 1].rstrip() + "…"
    return title or "New conversation"


@dataclass(frozen=True)
class Conversation:
    id: str
    title: str
    created_at: float
    updated_at: float
    message_count: int


@dataclass(frozen=True)
class StoredImage:
    id: str
    conversation_id: str
    name: str
    mime: str
    width: int
    height: int
    sha256: str
    size_bytes: int


class ConversationStore:
    def __init__(self, path: Path, clock=time.time) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _migrate(self) -> None:
        with self._lock:
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise StoreError("The conversation database was created by a newer version "
                                 "of OmniFlow.")
            for target in range(version + 1, SCHEMA_VERSION + 1):
                with self._conn:
                    self._conn.executescript(_MIGRATIONS[target])
                    self._conn.execute(f"PRAGMA user_version = {int(target)}")

    def _check(self, conversation_id: str) -> None:
        if not is_conversation_id(conversation_id):
            raise StoreError("Invalid conversation id.")

    def _claim(self, conversation_id: str, owner: str, title: str, now: float) -> None:
        """Create the row for `owner`, or confirm it already belongs to `owner`."""
        self._conn.execute(
            "INSERT OR IGNORE INTO conversations (id, title, created_at, updated_at, owner_id) "
            "VALUES (?, ?, ?, ?, ?)", (conversation_id, make_title(title), now, now, owner))
        row = self._conn.execute("SELECT owner_id FROM conversations WHERE id = ?",
                                 (conversation_id,)).fetchone()
        if row[0] != owner:
            raise StoreError("That conversation belongs to another session.")

    # ---------------------------------------------------------- conversations
    def ensure(self, conversation_id: str, title: str = "New conversation", *,
               owner: str = LOCAL_OWNER) -> None:
        """Create the conversation row if it does not exist yet."""
        self._check(conversation_id)
        with self._lock, self._conn:
            self._claim(conversation_id, owner, title, self._clock())

    def get(self, conversation_id: str, *, owner: str = LOCAL_OWNER) -> Conversation | None:
        if not is_conversation_id(conversation_id):
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(m.id) "
                "FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id "
                "WHERE c.id = ? AND c.owner_id = ? GROUP BY c.id",
                (conversation_id, owner)).fetchone()
        return Conversation(*row) if row else None

    def list(self, limit: int = 50, *, owner: str = LOCAL_OWNER) -> list[Conversation]:
        """Most recently active first; conversations with no messages are not listed."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(m.id) AS n "
                "FROM conversations c JOIN messages m ON m.conversation_id = c.id "
                "WHERE c.owner_id = ? GROUP BY c.id ORDER BY c.updated_at DESC LIMIT ?",
                (owner, int(limit))).fetchall()
        return [Conversation(*r) for r in rows]

    def rename(self, conversation_id: str, title: str, *, owner: str = LOCAL_OWNER) -> None:
        self._check(conversation_id)
        with self._lock, self._conn:
            self._conn.execute("UPDATE conversations SET title = ? WHERE id = ? AND owner_id = ?",
                               (make_title(title), conversation_id, owner))

    def delete(self, conversation_id: str, *, owner: str = LOCAL_OWNER) -> bool:
        """Delete a conversation with its messages and images (cascade)."""
        self._check(conversation_id)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "DELETE FROM conversations WHERE id = ? AND owner_id = ?",
                (conversation_id, owner))
            return cursor.rowcount == 1

    def purge_owners(self, *, owner_prefix: str, older_than: float) -> list[str]:
        """Delete stale conversations of temporary (per-session) owners; returns their ids."""
        with self._lock, self._conn:
            ids = [r[0] for r in self._conn.execute(
                "SELECT id FROM conversations WHERE substr(owner_id, 1, ?) = ? "
                "AND updated_at < ?", (len(owner_prefix), owner_prefix, older_than))]
            self._conn.executemany("DELETE FROM conversations WHERE id = ?",
                                   [(i,) for i in ids])
        return ids

    # --------------------------------------------------------------- messages
    def add_message(self, conversation_id: str, turn_id: str, role: str, content: str,
                    payload: dict[str, Any] | None = None, *,
                    owner: str = LOCAL_OWNER) -> bool:
        """Save one message exactly once per (conversation, turn, role). Returns True if new."""
        self._check(conversation_id)
        if role not in ("user", "assistant"):
            raise StoreError("Invalid role.")
        text = redact(content or "")
        blob = None
        if payload:
            blob = json.dumps(payload, default=str, ensure_ascii=False)
            if len(blob.encode("utf-8")) > MAX_PAYLOAD_BYTES:
                slim = {k: v for k, v in payload.items() if k not in ("artifacts", "evidence")}
                slim["note"] = "Tables, charts and evidence were too large to keep in history."
                blob = json.dumps(slim, default=str, ensure_ascii=False)
            blob = redact(blob)
        now = self._clock()
        with self._lock, self._conn:
            self._claim(conversation_id, owner, content if role == "user" else "", now)
            if role == "user":
                self._conn.execute(  # a placeholder title becomes the first question
                    "UPDATE conversations SET title = ? WHERE id = ? AND title = ?",
                    (make_title(content), conversation_id, "New conversation"))
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO messages (conversation_id, turn_id, role, content, "
                "payload, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (conversation_id, turn_id, role, text, blob, now))
            self._conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?",
                               (now, conversation_id))
            return cursor.rowcount == 1

    def messages(self, conversation_id: str, *,
                 owner: str = LOCAL_OWNER) -> list[dict[str, Any]]:
        """Display entries in order; a user turn with no reply is marked `interrupted`."""
        if not is_conversation_id(conversation_id):
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT m.turn_id, m.role, m.content, m.payload FROM messages m "
                "JOIN conversations c ON c.id = m.conversation_id "
                "WHERE m.conversation_id = ? AND c.owner_id = ? ORDER BY m.id",
                (conversation_id, owner)).fetchall()
        answered = {turn for turn, role, _, _ in rows if role == "assistant"}
        entries = []
        for turn_id, role, content, payload in rows:
            entry = json.loads(payload) if payload else {}
            entry |= {"role": role, "content": content, "turn_id": turn_id}
            if role == "user" and turn_id not in answered:
                entry["interrupted"] = True
            entries.append(entry)
        return entries

    # ----------------------------------------------------------------- images
    def add_image(self, conversation_id: str, *, name: str, mime: str, width: int,
                  height: int, sha256: str, data: bytes, max_images: int,
                  owner: str = LOCAL_OWNER) -> StoredImage:
        self._check(conversation_id)
        self.ensure(conversation_id, owner=owner)  # titled by the first question asked
        with self._lock, self._conn:
            existing = self._conn.execute(
                "SELECT id FROM images WHERE conversation_id = ? AND sha256 = ?",
                (conversation_id, sha256)).fetchone()
            if existing:
                return self.image(conversation_id, existing[0])
            count = self._conn.execute("SELECT COUNT(*) FROM images WHERE conversation_id = ?",
                                       (conversation_id,)).fetchone()[0]
            if count >= max_images:
                raise StoreError(f"The limit is {max_images} images per conversation.")
            image_id = uuid.uuid4().hex[:12]
            self._conn.execute(
                "INSERT INTO images (id, conversation_id, name, mime, width, height, sha256, "
                "data, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (image_id, conversation_id, name, mime, width, height, sha256, data,
                 self._clock()))
        return self.image(conversation_id, image_id)

    def images(self, conversation_id: str) -> list[StoredImage]:
        if not is_conversation_id(conversation_id):
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, conversation_id, name, mime, width, height, sha256, LENGTH(data) "
                "FROM images WHERE conversation_id = ? ORDER BY created_at",
                (conversation_id,)).fetchall()
        return [StoredImage(*r) for r in rows]

    def image(self, conversation_id: str, image_id: str) -> StoredImage | None:
        if not is_conversation_id(conversation_id):
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT id, conversation_id, name, mime, width, height, sha256, LENGTH(data) "
                "FROM images WHERE conversation_id = ? AND id = ?",
                (conversation_id, image_id)).fetchone()
        return StoredImage(*row) if row else None

    def image_bytes(self, conversation_id: str, image_id: str) -> bytes | None:
        """Only returns an image that belongs to the given conversation."""
        if not is_conversation_id(conversation_id):
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM images WHERE conversation_id = ? AND id = ?",
                (conversation_id, image_id)).fetchone()
        return bytes(row[0]) if row else None

    def remove_image(self, conversation_id: str, image_id: str) -> bool:
        self._check(conversation_id)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "DELETE FROM images WHERE conversation_id = ? AND id = ?",
                (conversation_id, image_id))
            return cursor.rowcount == 1
