"""Memory in the Streamlit UI (AppTest): chat commands, confirmations, retrieval, the
Memory page, the global switch, per-session isolation and storage failures."""

from pathlib import Path

import pytest
import streamlit as st
from langchain_core.messages import AIMessage, SystemMessage
from streamlit.testing.v1 import AppTest

from omniflow.agent import prompts
from omniflow.conversations.store import ConversationStore
from omniflow.core import llm
from omniflow.core.config import get_settings
from omniflow.memory.store import MemoryStore
from omniflow.ui import memory_chat
from tests.conftest import FAKE_KEY
from tests.fakes import ScriptedChatModel

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(autouse=True)
def clear_streamlit_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def _run(target) -> AppTest:
    at = target.run(timeout=60)
    assert not at.exception, at.exception
    return at


def _use_model(monkeypatch, *replies):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    model = ScriptedChatModel(replies=list(replies))
    monkeypatch.setattr(llm, "create_chat_runnable", lambda settings: model)
    return model


def _memories(owner="local"):
    return MemoryStore(get_settings().memory_db_path).list(owner)


def _text(at) -> str:
    groups = (at.markdown, at.caption, at.error, at.warning, at.info, at.success)
    return "\n".join(str(e.value) for group in groups for e in group)


def _click(at, label):
    return _run(next(b for b in at.button if b.label == label).click())


def _system_prompt(model, call=-1) -> str:
    return next(m.content for m in model.calls[call] if isinstance(m, SystemMessage))


def _memory_page():
    from omniflow.ui import memory_page

    memory_page.render()


# ---------------------------------------------------------------- chat commands
def test_remember_command_saves_without_a_model_call(monkeypatch):
    model = _use_model(monkeypatch)
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("Remember that I prefer concise answers."))
    assert model.calls == []
    assert [m.content for m in _memories()] == ["I prefer concise answers"]
    assert _memories()[0].category == "preference"
    text = _text(at)
    assert "Saved as a preference" in text and "Memory command" in text
    # Saying it again does not create a second copy.
    at = _run(at.chat_input[0].set_value("remember that i prefer CONCISE answers"))
    assert len(_memories()) == 1 and "already remember" in _text(at)


def test_secrets_are_refused_and_kept_out_of_chat_history(monkeypatch):
    _use_model(monkeypatch)
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("Remember that my bank password is Tr0ub4dor&3"))
    assert _memories() == []
    assert "never stored" in _text(at)
    saved = ConversationStore(get_settings().conversations_db_path).messages(
        at.session_state["thread_id"])
    assert "Tr0ub4dor" not in str(saved) and saved[0]["content"] == memory_chat.WITHHELD_MESSAGE


def test_personal_details_need_a_confirm_click(monkeypatch):
    _use_model(monkeypatch)
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("Remember that my email is asha@example.com"))
    assert _memories() == []  # nothing saved before the click
    at = _click(at, "Save it")
    assert [m.source for m in _memories()] == ["confirmed"]
    assert "Saved as" in _text(at)


def test_declining_a_confirmation_saves_nothing(monkeypatch):
    _use_model(monkeypatch)
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("Remember that my email is asha@example.com"))
    at = _click(at, "Don't save")
    assert _memories() == []


def test_project_memory_requires_an_active_project(monkeypatch):
    _use_model(monkeypatch)
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("Remember this for my project: we use SQLite"))
    assert _memories() == [] and "active project" in _text(at)
    MemoryStore(get_settings().memory_db_path).set_active_project("local", "OmniFlow")
    at = _run(at.chat_input[0].set_value("Remember this for my project: we use SQLite"))
    [memory] = _memories()
    assert memory.scope == "project:OmniFlow" and memory.category == "project"


def test_preference_statements_are_suggested_not_saved(monkeypatch):
    _use_model(monkeypatch, AIMessage("- Sure."))
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("I prefer answers in bullet points"))
    assert _memories() == []
    assert "Remember this as a preference" in _text(at)
    at = _click(at, "Remember it")
    assert [m.content for m in _memories()] == ["I prefer answers in bullet points"]


def test_forget_asks_which_memory_then_deletes_only_that_one(monkeypatch):
    _use_model(monkeypatch)
    store = MemoryStore(get_settings().memory_db_path)
    keep = store.add("local", "Our API runs on port 8080", category="project").memory
    drop = store.add("local", "I like tables for comparisons", category="preference").memory
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("Forget that I like tables"))
    assert len(_memories()) == 2  # nothing deleted yet
    at = _run(next(b for b in at.button if b.key == f"memory-forget-{drop.id}").click())
    assert [m.id for m in _memories()] == [keep.id]
    assert "Forgotten" in _text(at)


def test_show_lists_saved_memories(monkeypatch):
    _use_model(monkeypatch)
    MemoryStore(get_settings().memory_db_path).add("local", "I prefer metric units",
                                                   category="preference")
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("What do you remember about me?"))
    assert "I prefer metric units" in _text(at)


# ------------------------------------------------------------------- retrieval
def test_relevant_memories_reach_the_model_and_are_shown(monkeypatch):
    model = _use_model(monkeypatch, AIMessage("- PostgreSQL tuning tips"))
    store = MemoryStore(get_settings().memory_db_path)
    store.add("local", "I prefer bullet points", category="preference")
    store.add("local", "Our database is PostgreSQL 16", category="explicit")
    store.add("local", "My cat is called Miso", category="explicit")
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("How should I tune our database?"))
    system = _system_prompt(model)
    assert "bullet points" in system and "PostgreSQL 16" in system and "Miso" not in system
    assert "Saved memory available to this answer" in _text(at)


def test_turning_memory_off_stops_retrieval_and_saving(monkeypatch):
    model = _use_model(monkeypatch, AIMessage("ok"))
    store = MemoryStore(get_settings().memory_db_path)
    store.add("local", "I prefer bullet points", category="preference")
    store.set_enabled("local", False)
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("Remember that I like dark mode"))
    assert "turned off" in _text(at) and len(_memories()) == 1
    at = _run(at.chat_input[0].set_value("hello"))
    assert prompts.MEMORY_OPEN not in _system_prompt(model)


def test_installation_switch_disables_memory(monkeypatch):
    monkeypatch.setenv("OMNIFLOW_MEMORY_ENABLED", "false")
    model = _use_model(monkeypatch, AIMessage("ok"))
    MemoryStore(get_settings().memory_db_path).add("local", "I prefer bullet points",
                                                   category="preference")
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("hello"))
    assert prompts.MEMORY_OPEN not in _system_prompt(model)
    assert any("switched off" in c.value for c in at.sidebar.caption)


# ------------------------------------------------------------------ Memory page
def test_memory_page_add_edit_disable_delete(monkeypatch):
    _use_model(monkeypatch)
    at = _run(AppTest.from_function(_memory_page))
    assert at.title[0].value == "🧠 Memory"
    at.text_area(key="memory_new_text").set_value("I prefer short summaries")
    at.selectbox(key="memory_new_category").set_value("preference")
    at = _click(at, "Save memory")
    [memory] = _memories()
    assert memory.source == "manual" and any("saved" in s.value for s in at.success)

    at.text_area(key=f"memory_edit_{memory.id}").set_value("I prefer very short summaries")
    at = _click(at, "Save changes")
    assert _memories()[0].content == "I prefer very short summaries"

    at = _click(at, "Disable")
    assert not _memories()[0].active
    at.selectbox(key="memory_filter_status").set_value("Active")
    at = _run(at)
    assert "No memories match" in _text(at)
    at = _run(at.selectbox(key="memory_filter_status").set_value("All"))
    at = _click(at, "Re-enable")
    assert _memories()[0].active

    at = _click(at, "Delete…")
    assert _memories()  # a confirmation is required
    at = _click(at, "Yes, delete")
    assert _memories() == []


def test_memory_page_refuses_secrets_and_reports_errors_without_success(monkeypatch):
    _use_model(monkeypatch)
    at = _run(AppTest.from_function(_memory_page))
    at.text_area(key="memory_new_text").set_value(f"key: {FAKE_KEY}")
    at = _click(at, "Save memory")
    assert _memories() == [] and not at.success and at.error
    assert FAKE_KEY not in _text(at)


def test_clear_all_requires_typing_the_confirmation_word(monkeypatch):
    _use_model(monkeypatch)
    store = MemoryStore(get_settings().memory_db_path)
    store.add("local", "one", category="explicit")
    store.add("local", "two things", category="explicit")
    at = _run(AppTest.from_function(_memory_page))
    clear = next(b for b in at.button if b.label == "Clear all memories")
    assert clear.disabled
    at.text_input(key="memory_clear_word").set_value("delete")
    at = _run(at)
    assert next(b for b in at.button if b.label == "Clear all memories").disabled
    at.text_input(key="memory_clear_word").set_value("DELETE")
    at = _run(at)
    at = _click(at, "Clear all memories")
    assert _memories() == [] and "Deleted all 2" in _text(at)


def test_memory_page_toggle_and_project(monkeypatch):
    _use_model(monkeypatch)
    at = _run(AppTest.from_function(_memory_page))
    at = _run(at.toggle(key="memory_enabled_toggle").set_value(False))
    store = MemoryStore(get_settings().memory_db_path)
    assert store.enabled("local") is False
    at.text_input(key="memory_project_input").set_value("OmniFlow")
    at = _click(at, "Set active project")
    assert store.active_project("local") == "OmniFlow"
    assert "survive app restarts" in _text(at)


def test_memories_survive_an_app_restart(monkeypatch):
    _use_model(monkeypatch)
    at = _run(AppTest.from_file(APP))
    _run(at.chat_input[0].set_value("Remember that I prefer metric units"))
    st.cache_resource.clear()  # a new process
    at = _run(AppTest.from_function(_memory_page))
    assert "I prefer metric units" in _text(at)


# ------------------------------------------------- shared deployments and failures
def test_per_session_mode_isolates_memories_and_conversations(monkeypatch):
    monkeypatch.setenv("OMNIFLOW_ACCESS_MODE", "per_session")
    _use_model(monkeypatch, AIMessage("hi Asha"))
    alice = _run(AppTest.from_file(APP))
    alice = _run(alice.chat_input[0].set_value("Remember that my favourite language is Rust"))
    alice = _run(alice.chat_input[0].set_value("Hello, I am Asha"))
    alice_owner = alice.session_state["owner_id"]
    assert alice_owner.startswith("s-")

    bob = _run(AppTest.from_function(_memory_page))
    assert "Rust" not in _text(bob)
    assert bob.session_state["owner_id"] != alice_owner
    assert "Temporary on this deployment" in _text(bob)
    bob_app = _run(AppTest.from_file(APP))
    sidebar = " ".join(b.label for b in bob_app.sidebar.button)
    assert "Hello, I am Asha" not in sidebar
    # Bob cannot open Alice's conversation even with its id in the URL.
    sneaky = AppTest.from_file(APP)
    sneaky.query_params["c"] = alice.session_state["thread_id"]
    sneaky = _run(sneaky)
    assert sneaky.session_state["thread_id"] != alice.session_state["thread_id"]
    assert "Hello, I am Asha" not in _text(sneaky)


def test_unwritable_memory_storage_is_reported_and_chat_keeps_working(monkeypatch, tmp_path):
    blocked = tmp_path / "not-a-file"
    blocked.mkdir()
    monkeypatch.setenv("OMNIFLOW_MEMORY_DB", str(blocked))
    _use_model(monkeypatch, AIMessage("normal answer"))
    at = _run(AppTest.from_file(APP))
    assert any("unavailable" in c.value for c in at.sidebar.caption)
    at = _run(at.chat_input[0].set_value("Remember that I like tea"))
    assert "could not be opened" in _text(at)
    at = _run(at.chat_input[0].set_value("hello"))
    assert "normal answer" in _text(at)
    page = _run(AppTest.from_function(_memory_page))
    assert any("could not be opened" in e.value for e in page.error)
