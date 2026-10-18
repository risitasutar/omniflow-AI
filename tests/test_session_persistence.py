"""Shared deployments (per_session): conversations survive a browser refresh (AppTest).

A browser refresh starts a brand-new Streamlit session with empty session_state, arriving
with the same URL. Each test reproduces that by starting a new AppTest with the query
parameters the previous session left in its URL.
"""

import re
from pathlib import Path

import pytest
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from streamlit.testing.v1 import AppTest

from omniflow.core import llm
from tests.conftest import FAKE_KEY
from tests.fakes import ScriptedChatModel

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(autouse=True)
def clear_streamlit_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


@pytest.fixture
def shared_deployment(monkeypatch):
    monkeypatch.setenv("OMNIFLOW_ACCESS_MODE", "per_session")


def _run(target) -> AppTest:
    at = target.run(timeout=60)
    assert not at.exception, at.exception
    return at


def _use_model(monkeypatch, *replies) -> ScriptedChatModel:
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    model = ScriptedChatModel(replies=list(replies))
    monkeypatch.setattr(llm, "create_chat_runnable", lambda settings: model)
    return model


def _refresh(previous: AppTest) -> AppTest:
    """A new browser session that arrives with the previous session's URL."""
    fresh = AppTest.from_file(APP)
    for key, value in previous.query_params.items():
        fresh.query_params[key] = value
    return _run(fresh)


def _text(at) -> str:
    groups = (at.markdown, at.caption, at.error, at.warning, at.info)
    return "\n".join(str(e.value) for group in groups for e in group)


def _sidebar_titles(at) -> str:
    return " ".join(b.label for b in at.sidebar.button)


def test_the_owner_key_is_put_in_the_url(monkeypatch, shared_deployment):
    _use_model(monkeypatch)
    at = _run(AppTest.from_file(APP))
    key = at.query_params.get("o")
    key = key[0] if isinstance(key, list) else key
    assert re.fullmatch(r"[0-9a-f]{32}", key)
    assert at.session_state["owner_id"] == f"s-{key}"


def test_a_refresh_restores_the_open_conversation_and_the_list(monkeypatch, shared_deployment):
    _use_model(monkeypatch, AIMessage("Hello Asha"), AIMessage("Second chat answer"))
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("Hi, I am Asha"))
    first_thread = at.session_state["thread_id"]
    at = _run(next(b for b in at.sidebar.button if "New chat" in b.label).click())
    at = _run(at.chat_input[0].set_value("Another question"))
    second_thread = at.session_state["thread_id"]
    assert second_thread != first_thread  # a new chat never overwrites an existing one

    refreshed = _refresh(at)
    assert refreshed.session_state["owner_id"] == at.session_state["owner_id"]
    assert refreshed.session_state["thread_id"] == second_thread  # the open chat reopens
    assert "Second chat answer" in _text(refreshed)
    titles = _sidebar_titles(refreshed)
    assert "Hi, I am Asha" in titles and "Another question" in titles  # both still listed


def test_the_url_keys_survive_switching_pages(monkeypatch, shared_deployment):
    _use_model(monkeypatch, AIMessage("ok"))
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("Keep me"))
    owner, thread = at.session_state["owner_id"], at.session_state["thread_id"]
    nav = next(s for s in at.sidebar.selectbox if s.label == "Go to")
    at = _run(nav.set_value(1))  # Documents page
    assert at.title[0].value.startswith("📄")
    refreshed = _refresh(at)  # refresh while on another page
    assert refreshed.session_state["owner_id"] == owner
    assert refreshed.session_state["thread_id"] == thread


def test_without_the_key_a_visitor_sees_nothing(monkeypatch, shared_deployment):
    _use_model(monkeypatch, AIMessage("private answer"))
    alice = _run(AppTest.from_file(APP))
    alice = _run(alice.chat_input[0].set_value("Alice's private question"))
    stranger = AppTest.from_file(APP)
    stranger.query_params["c"] = alice.session_state["thread_id"]  # conversation id only
    stranger = _run(stranger)
    assert stranger.session_state["owner_id"] != alice.session_state["owner_id"]
    assert "Alice's private question" not in _sidebar_titles(stranger)
    assert "private answer" not in _text(stranger)


@pytest.mark.parametrize("bad_key", ["../../etc", "ABCDEF0123456789ABCDEF0123456789", "1234",
                                     "s-0123456789abcdef0123456789abcdef"])
def test_malformed_keys_are_ignored(monkeypatch, shared_deployment, bad_key):
    _use_model(monkeypatch)
    at = AppTest.from_file(APP)
    at.query_params["o"] = bad_key
    at = _run(at)
    owner = at.session_state["owner_id"]
    assert re.fullmatch(r"s-[0-9a-f]{32}", owner) and owner != f"s-{bad_key}"


def test_conversations_and_agent_memory_survive_a_restart(monkeypatch, shared_deployment):
    model = _use_model(monkeypatch, AIMessage("Nice to meet you, Asha"),
                       AIMessage("Your name is Asha"))
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("My name is Asha"))
    st.cache_resource.clear()  # a new process: stores, checkpointer and graph rebuilt
    refreshed = _refresh(at)
    assert "Nice to meet you, Asha" in _text(refreshed)
    refreshed = _run(refreshed.chat_input[0].set_value("What is my name?"))
    # The agent's checkpoint (SQLite) was reloaded: the earlier turn is in the prompt.
    second_prompt = model.calls[1]
    assert any(isinstance(m, HumanMessage) and m.text == "My name is Asha"
               for m in second_prompt)
    assert "Your name is Asha" in _text(refreshed)
    # Messages are not duplicated: 2 user turns, 2 answers.
    assert [m["role"] for m in refreshed.session_state["messages"]] == [
        "user", "assistant", "user", "assistant"]


def test_single_user_installs_keep_their_behaviour(monkeypatch):
    monkeypatch.setenv("OMNIFLOW_ACCESS_MODE", "single_user")
    _use_model(monkeypatch, AIMessage("local answer"))
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("Local question"))
    assert "o" not in at.query_params  # no owner key needed locally
    refreshed = _refresh(at)
    assert refreshed.session_state["thread_id"] == at.session_state["thread_id"]
