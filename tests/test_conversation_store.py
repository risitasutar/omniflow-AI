"""Conversation store: persistence across restarts, exactly-once saves, isolation, deletion."""

import sqlite3

import pytest

from omniflow.conversations.store import (
    MAX_PAYLOAD_BYTES,
    SCHEMA_VERSION,
    ConversationStore,
    StoreError,
    is_conversation_id,
    make_title,
    new_conversation_id,
)
from tests.conftest import FAKE_KEY


@pytest.fixture
def path(tmp_path):
    return tmp_path / "conv.db"


def _image(store, cid, sha="a" * 64, data=b"img", limit=5):
    return store.add_image(cid, name="chart.png", mime="image/png", width=10, height=8,
                           sha256=sha, data=data, max_images=limit)


def test_ids_are_server_generated_hex_and_validated():
    cid = new_conversation_id()
    assert is_conversation_id(cid) and len(cid) == 32
    for bad in (None, "", "abc", "../../etc", "x" * 32, cid.upper(), cid + "0"):
        assert not is_conversation_id(bad)


def test_conversation_survives_a_restart(path):
    store = ConversationStore(path)
    cid = new_conversation_id()
    store.add_message(cid, "t1", "user", "What is RAG?")
    store.add_message(cid, "t1", "assistant", "Retrieval-augmented generation [1].",
                      {"route": "tool", "evidence": [{"id": 1, "title": "Paper"}]})
    store.close()

    reopened = ConversationStore(path)  # a new process would do exactly this
    [conversation] = reopened.list()
    assert conversation.id == cid and conversation.title == "What is RAG?"
    assert conversation.message_count == 2
    messages = reopened.messages(cid)
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["route"] == "tool" and messages[1]["evidence"][0]["title"] == "Paper"
    assert not any(m.get("interrupted") for m in messages)


def test_each_message_is_saved_exactly_once(path):
    store = ConversationStore(path)
    cid = new_conversation_id()
    assert store.add_message(cid, "t1", "user", "hi") is True
    assert store.add_message(cid, "t1", "user", "hi") is False  # rerun / double submit
    assert store.add_message(cid, "t1", "assistant", "partial", {"interrupted": True})
    assert store.add_message(cid, "t1", "assistant", "final") is False  # first write wins
    assert [m["content"] for m in store.messages(cid)] == ["hi", "partial"]


def test_user_turn_without_reply_is_marked_interrupted(path):
    store = ConversationStore(path)
    cid = new_conversation_id()
    store.add_message(cid, "t1", "user", "first")
    store.add_message(cid, "t1", "assistant", "answer")
    store.add_message(cid, "t2", "user", "second")  # the app stopped before replying
    messages = store.messages(cid)
    assert messages[0].get("interrupted") is None
    assert messages[2]["interrupted"] is True


def test_conversations_are_isolated(path):
    store = ConversationStore(path)
    a, b = new_conversation_id(), new_conversation_id()
    store.add_message(a, "t1", "user", "alpha question")
    store.add_message(b, "t1", "user", "beta question")
    assert [m["content"] for m in store.messages(a)] == ["alpha question"]
    assert [m["content"] for m in store.messages(b)] == ["beta question"]
    image = _image(store, a)
    assert store.image_bytes(b, image.id) is None  # another conversation's image id
    assert store.images(b) == []
    assert store.messages("not-an-id") == [] and store.get("../x") is None


def test_list_is_most_recent_first_and_hides_empty_conversations(path):
    clock = iter(range(100, 200))
    store = ConversationStore(path, clock=lambda: next(clock))
    a, b, empty = new_conversation_id(), new_conversation_id(), new_conversation_id()
    store.add_message(a, "t1", "user", "older")
    store.add_message(b, "t1", "user", "newer")
    store.ensure(empty)
    assert [c.id for c in store.list()] == [b, a]
    store.add_message(a, "t2", "user", "again")
    assert [c.id for c in store.list()] == [a, b]
    assert len(store.list(limit=1)) == 1


def test_rename_and_titles(path):
    store = ConversationStore(path)
    cid = new_conversation_id()
    store.add_message(cid, "t1", "user", "  Compare   GDP of India\nand Brazil  ")
    assert store.get(cid).title == "Compare GDP of India and Brazil"
    store.rename(cid, "GDP comparison")
    store.add_message(cid, "t2", "user", "another question")  # does not overwrite a title
    assert store.get(cid).title == "GDP comparison"
    assert len(make_title("x" * 500)) == 80
    assert make_title("   ") == "New conversation"


def test_delete_removes_messages_and_images(path):
    store = ConversationStore(path)
    cid = new_conversation_id()
    store.add_message(cid, "t1", "user", "hello")
    _image(store, cid)
    store.delete(cid)
    assert store.get(cid) is None and store.list() == []
    with sqlite3.connect(path) as raw:
        assert raw.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        assert raw.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 0


def test_secrets_are_redacted_before_saving(path):
    store = ConversationStore(path)
    cid = new_conversation_id()
    store.add_message(cid, "t1", "user", f"my key is {FAKE_KEY}")
    store.add_message(cid, "t1", "assistant", "ok", {"note": f"token={FAKE_KEY}"})
    with sqlite3.connect(path) as raw:
        dump = "\n".join(str(r) for r in raw.execute("SELECT * FROM messages"))
        dump += "\n".join(str(r) for r in raw.execute("SELECT * FROM conversations"))
    assert FAKE_KEY not in dump and "[REDACTED]" in dump


def test_oversized_payload_keeps_the_answer_and_drops_tables(path):
    store = ConversationStore(path)
    cid = new_conversation_id()
    huge = {"route": "tool", "artifacts": [{"rows": "x" * (MAX_PAYLOAD_BYTES + 10)}]}
    store.add_message(cid, "t1", "assistant", "the answer", huge)
    [entry] = store.messages(cid)
    assert entry["content"] == "the answer" and entry["route"] == "tool"
    assert "artifacts" not in entry and "too large" in entry["note"]


def test_schema_is_versioned_and_newer_databases_are_refused(path):
    ConversationStore(path).close()
    with sqlite3.connect(path) as raw:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        raw.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    with pytest.raises(StoreError, match="newer version"):
        ConversationStore(path)


def test_sql_injection_in_values_is_stored_as_text(path):
    store = ConversationStore(path)
    cid = new_conversation_id()
    store.add_message(cid, "t1", "user", "'); DROP TABLE messages; --")
    store.rename(cid, "x'; DELETE FROM conversations; --")
    assert store.messages(cid)[0]["content"] == "'); DROP TABLE messages; --"
    assert store.get(cid).title.startswith("x'; DELETE")


def test_invalid_ids_and_roles_are_rejected(path):
    store = ConversationStore(path)
    with pytest.raises(StoreError):
        store.add_message("../evil", "t1", "user", "hi")
    with pytest.raises(StoreError):
        store.add_message(new_conversation_id(), "t1", "system", "hi")


def test_images_are_deduplicated_limited_and_removable(path):
    store = ConversationStore(path)
    cid = new_conversation_id()
    first = _image(store, cid, sha="a" * 64, limit=2)
    assert _image(store, cid, sha="a" * 64, limit=2).id == first.id  # same content again
    second = _image(store, cid, sha="b" * 64, limit=2)
    with pytest.raises(StoreError, match="limit is 2"):
        _image(store, cid, sha="c" * 64, limit=2)
    assert [i.id for i in store.images(cid)] == [first.id, second.id]
    assert store.image_bytes(cid, first.id) == b"img"
    assert store.remove_image(cid, first.id) is True
    assert store.remove_image(cid, first.id) is False
    assert [i.id for i in store.images(cid)] == [second.id]
