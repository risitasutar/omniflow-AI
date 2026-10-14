"""Memory store: schema, CRUD, duplicates, owners, deletion scope, failures, restarts."""

import sqlite3
import threading

import pytest

from omniflow.conversations.store import ConversationStore, new_conversation_id
from omniflow.memory.policy import MemoryRejected
from omniflow.memory.store import (
    GLOBAL_SCOPE,
    SCHEMA_VERSION,
    MemoryStore,
    MemoryStoreError,
    MemoryUnavailable,
    project_scope,
)
from tests.conftest import FAKE_KEY

ME, OTHER = "local", "s-other"


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "mem" / "memory.db")


def test_initialises_in_a_clean_directory_with_a_versioned_schema(tmp_path):
    path = tmp_path / "fresh" / "nested" / "memory.db"
    MemoryStore(path).close()
    with sqlite3.connect(path) as raw:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        tables = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"memories", "memory_settings"} <= tables
    MemoryStore(path).close()  # reopening is a no-op migration


def test_newer_schema_is_refused_without_changes(tmp_path):
    path = tmp_path / "memory.db"
    MemoryStore(path).close()
    with sqlite3.connect(path) as raw:
        raw.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    with pytest.raises(MemoryUnavailable, match="newer version"):
        MemoryStore(path)


def test_unwritable_or_damaged_storage_is_reported_not_hidden(tmp_path):
    folder = tmp_path / "is-a-directory"
    folder.mkdir()
    with pytest.raises(MemoryUnavailable, match="could not be opened"):
        MemoryStore(folder)
    damaged = tmp_path / "damaged.db"
    damaged.write_bytes(b"this is not a sqlite database" * 100)
    with pytest.raises(MemoryUnavailable):
        MemoryStore(damaged)


def test_failed_migration_leaves_no_half_built_schema(tmp_path, monkeypatch):
    from omniflow.memory import store as module

    monkeypatch.setitem(module._MIGRATIONS, 1, module._MIGRATIONS[1] + "\nTHIS IS NOT SQL;")
    path = tmp_path / "memory.db"
    with pytest.raises(MemoryUnavailable):
        MemoryStore(path)
    with sqlite3.connect(path) as raw:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == 0
        assert not raw.execute("SELECT name FROM sqlite_master WHERE name='memories'").fetchall()


def test_crud_and_metadata(store):
    saved = store.add(ME, "I prefer concise answers", category="preference")
    assert saved.created and saved.memory.source == "explicit" and saved.memory.active
    memory = saved.memory
    assert store.get(ME, memory.id).content == "I prefer concise answers"
    updated = store.update(ME, memory.id, "I prefer very concise answers", category="preference")
    assert updated.content == "I prefer very concise answers"
    assert updated.updated_at >= memory.updated_at
    assert store.set_active(ME, memory.id, False) is True
    assert not store.get(ME, memory.id).active
    assert store.list(ME, active=True) == [] and len(store.list(ME, active=False)) == 1
    assert store.delete(ME, memory.id) is True and store.get(ME, memory.id) is None
    assert store.delete(ME, memory.id) is False


def test_exact_and_near_duplicates_are_not_stored_twice(store):
    first = store.add(ME, "I prefer concise answers.", category="preference")
    for variant in ("i prefer CONCISE answers", "  I prefer concise answers!! ",
                    "Please remember I prefer concise answers"):
        again = store.add(ME, variant, category="preference")
        assert not again.created and again.memory.id == first.memory.id
    assert len(store.list(ME)) == 1
    assert store.add(ME, "I prefer detailed answers with examples", category="preference").created


def test_a_disabled_duplicate_is_re_enabled_instead_of_copied(store):
    memory = store.add(ME, "We deploy on Fridays", category="project").memory
    store.set_active(ME, memory.id, False)
    again = store.add(ME, "we deploy on fridays", category="project")
    assert not again.created and again.memory.active and len(store.list(ME)) == 1


def test_editing_into_a_duplicate_is_refused(store):
    store.add(ME, "Use metric units", category="preference")
    other = store.add(ME, "Answer in English", category="preference").memory
    with pytest.raises(MemoryStoreError, match="already says"):
        store.update(ME, other.id, "use metric units")


def test_secrets_are_refused_even_on_edit(store):
    with pytest.raises(MemoryRejected):
        store.add(ME, f"my gemini key is {FAKE_KEY}", category="explicit")
    memory = store.add(ME, "Prod database is Postgres", category="project").memory
    with pytest.raises(MemoryRejected):
        store.update(ME, memory.id, "Prod database password is hunter2!")
    assert store.get(ME, memory.id).content == "Prod database is Postgres"
    with sqlite3.connect(store.path) as raw:
        dump = " ".join(str(r) for r in raw.execute("SELECT * FROM memories"))
    assert FAKE_KEY not in dump and "hunter2" not in dump


def test_owners_are_isolated_for_every_operation(store):
    mine = store.add(ME, "My team uses Python", category="project").memory
    theirs = store.add(OTHER, "Their team uses Go", category="project").memory
    assert [m.content for m in store.list(ME)] == ["My team uses Python"]
    assert store.get(ME, theirs.id) is None
    assert store.delete(ME, theirs.id) is False
    assert store.set_active(ME, theirs.id, False) is False
    with pytest.raises(MemoryStoreError):
        store.update(ME, theirs.id, "hijacked")
    assert store.clear(ME) == 1
    assert store.get(OTHER, theirs.id) is not None and store.get(ME, mine.id) is None
    # The same text may exist for two owners without clashing.
    assert store.add(OTHER, "My team uses Python", category="project").created


def test_scopes_keep_projects_apart(store):
    alpha, beta = project_scope("Alpha"), project_scope("Beta")
    store.add(ME, "Use SQLite", category="project", scope=alpha)
    assert store.add(ME, "Use SQLite", category="project", scope=beta).created
    assert sorted(store.scopes(ME)) == sorted([alpha, beta])
    assert [m.scope for m in store.list(ME, scope=alpha)] == [alpha]
    assert project_scope("  ") == GLOBAL_SCOPE


def test_settings_enabled_and_active_project(store):
    assert store.enabled(ME) is True  # on by default
    store.set_enabled(ME, False)
    assert store.enabled(ME) is False and store.enabled(OTHER) is True
    store.set_active_project(ME, "  OmniFlow   AI ")
    assert store.active_project(ME) == "OmniFlow AI" and store.enabled(ME) is False
    store.set_active_project(ME, "")
    assert store.active_project(ME) is None


def test_memories_survive_a_restart(tmp_path):
    path = tmp_path / "memory.db"
    first = MemoryStore(path)
    first.add(ME, "I prefer tables for comparisons", category="preference")
    first.set_enabled(ME, False)
    first.close()
    second = MemoryStore(path)
    assert [m.content for m in second.list(ME)] == ["I prefer tables for comparisons"]
    assert second.enabled(ME) is False


def test_expired_and_stale_session_memories_are_purged(tmp_path):
    clock = [1000.0]
    store = MemoryStore(tmp_path / "memory.db", clock=lambda: clock[0])
    store.add(ME, "keep me", category="explicit")
    store.add(ME, "temporary note", category="explicit", expires_at=1500.0)
    store.add("s-old", "old session memory", category="explicit")
    clock[0] = 2000.0
    assert [m.content for m in store.list(ME)] == ["keep me"]  # expired hidden at once
    assert store.purge(owner_prefix="s-", older_than=1999.0) == 2
    assert store.list("s-old") == [] and [m.content for m in store.list(ME)] == ["keep me"]


def test_clearing_memories_never_touches_conversations(tmp_path):
    conversations = ConversationStore(tmp_path / "conv.db")
    cid = new_conversation_id()
    conversations.add_message(cid, "t1", "user", "keep this chat")
    memories = MemoryStore(tmp_path / "memory.db")
    memories.add(ME, "a memory", category="explicit")
    memories.clear(ME)
    assert conversations.get(cid) is not None and len(conversations.messages(cid)) == 1


def test_sql_metacharacters_are_stored_literally(store):
    text = "Robert'); DROP TABLE memories; --"
    memory = store.add(ME, text, category="explicit").memory
    assert store.get(ME, memory.id).content == text
    assert store.list(ME, category="explicit' OR '1'='1") == []
    assert store.get(ME, "x' OR '1'='1") is None


def test_unknown_category_is_rejected(store):
    with pytest.raises(MemoryStoreError):
        store.add(ME, "fine text", category="system")


def test_concurrent_sessions_save_one_copy(tmp_path):
    path = tmp_path / "memory.db"
    stores = [MemoryStore(path) for _ in range(4)]
    results, errors = [], []

    def save(s):
        try:
            results.append(s.add(ME, "We release every Monday", category="project"))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=save, args=(s,)) for s in stores for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(stores[0].list(ME)) == 1
    assert sum(r.created for r in results) == 1


def test_write_failure_is_reported(store, monkeypatch):
    store.close()  # any further write fails like a lost or locked database
    with pytest.raises(MemoryStoreError, match="not saved"):
        store.set_enabled(ME, True)
