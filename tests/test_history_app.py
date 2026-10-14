"""Saved conversations in the UI: New chat, reopen, restart, rename, delete, images (AppTest)."""

import io
from pathlib import Path

import pytest
import streamlit as st
from langchain_core.messages import AIMessage
from PIL import Image
from streamlit.testing.v1 import AppTest

import omniflow.tools as tools_pkg
from omniflow.agent import runner
from omniflow.conversations.store import ConversationStore, new_conversation_id
from omniflow.core import llm
from omniflow.core.config import get_settings
from omniflow.images.processing import prepare_image
from omniflow.tools.registry import Permission, SideEffect
from omniflow.ui import chat
from tests.conftest import FAKE_KEY
from tests.fakes import ScriptedChatModel, make_tool, tool_call

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


def _restart():
    """Drop every cached resource (store, checkpointer, graph): what a new process sees."""
    st.cache_resource.clear()


def _store() -> ConversationStore:
    return ConversationStore(get_settings().conversations_db_path)


def _button(at, label, sidebar=False):
    buttons = at.sidebar.button if sidebar else at.button
    return next(b for b in buttons if b.label == label)


def _text(at) -> str:
    groups = (at.markdown, at.caption, at.error, at.warning)
    return "\n".join(str(e.value) for group in groups for e in group)


def test_new_chat_keeps_the_previous_conversation_and_it_can_be_reopened(monkeypatch):
    _use_model(monkeypatch, AIMessage("first answer"), AIMessage("second answer"))
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("Hello there"))
    first = at.session_state["thread_id"]
    assert at.query_params["c"] == first

    at = _run(_button(at, "➕ New chat", sidebar=True).click())
    assert at.session_state["messages"] == []
    assert at.session_state["thread_id"] != first
    at = _run(at.chat_input[0].set_value("Another topic"))

    at = _run(_button(at, "Hello there", sidebar=True).click())
    assert at.session_state["thread_id"] == first
    assert [m["content"] for m in at.session_state["messages"]] == ["Hello there",
                                                                    "first answer"]
    assert [c.title for c in _store().list()] == ["Another topic", "Hello there"]


def test_conversation_and_agent_memory_survive_a_restart(monkeypatch):
    model = _use_model(monkeypatch, AIMessage("Noted, Asha."))
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("My name is Asha"))
    conversation_id = at.session_state["thread_id"]

    _restart()
    model.replies.append(AIMessage("Your name is Asha."))
    at = _run(AppTest.from_file(APP))
    assert at.session_state["messages"] == []  # a fresh session starts a new chat
    at = _run(_button(at, "My name is Asha", sidebar=True).click())
    assert at.session_state["thread_id"] == conversation_id
    at = _run(at.chat_input[0].set_value("What is my name?"))
    remembered = " ".join(m.text for m in model.calls[-1])
    assert "My name is Asha" in remembered  # the checkpoint was reloaded from disk
    assert "Your name is Asha." in _text(at)
    assert len(_store().messages(conversation_id)) == 4


def test_browser_refresh_reopens_the_conversation_from_the_url(monkeypatch):
    _use_model(monkeypatch, AIMessage("an answer"))
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("Refresh me"))
    conversation_id = at.session_state["thread_id"]

    _restart()
    fresh = AppTest.from_file(APP)
    fresh.query_params["c"] = conversation_id
    fresh = _run(fresh)
    assert fresh.session_state["thread_id"] == conversation_id
    assert "an answer" in _text(fresh)


def test_reruns_never_duplicate_saved_messages(monkeypatch):
    _use_model(monkeypatch, AIMessage("only once"))
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("Save me once"))
    at = _run(at)
    at = _run(at)
    conversation_id = at.session_state["thread_id"]
    assert [m["role"] for m in _store().messages(conversation_id)] == ["user", "assistant"]


def test_rename_and_delete_with_confirmation(monkeypatch):
    _use_model(monkeypatch, AIMessage("answer"))
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("Original title"))
    conversation_id = at.session_state["thread_id"]

    at = _run(at.sidebar.text_input(key=f"rename-{conversation_id}").set_value("Renamed"))
    at = _run(_button(at, "Rename", sidebar=True).click())
    assert _store().get(conversation_id).title == "Renamed"

    at = _run(_button(at, "Delete…", sidebar=True).click())
    assert any("permanently" in w.value for w in at.sidebar.warning)
    assert _store().get(conversation_id) is not None  # nothing deleted before confirming
    at = _run(_button(at, "Yes, delete", sidebar=True).click())

    assert _store().get(conversation_id) is None and _store().list() == []
    assert at.session_state["thread_id"] != conversation_id
    assert at.session_state["messages"] == []
    from omniflow.ui import state

    assert runner.turn_outcome(state.get_chatbot(), conversation_id).final_answer == ""


def test_cancelling_delete_keeps_the_conversation(monkeypatch):
    _use_model(monkeypatch, AIMessage("answer"))
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("Keep me"))
    at = _run(_button(at, "Delete…", sidebar=True).click())
    at = _run(_button(at, "Cancel", sidebar=True).click())
    assert [c.title for c in _store().list()] == ["Keep me"]


def test_a_turn_cut_off_by_a_restart_is_shown_as_unanswered(monkeypatch):
    _use_model(monkeypatch)
    store = _store()
    conversation_id = new_conversation_id()
    store.add_message(conversation_id, "t1", "user", "Question before the crash")
    at = AppTest.from_file(APP)
    at.query_params["c"] = conversation_id
    at = _run(at)
    assert chat.NO_REPLY_NOTE in _text(at)


def test_a_partial_reply_is_shown_as_interrupted(monkeypatch):
    _use_model(monkeypatch)
    store = _store()
    conversation_id = new_conversation_id()
    store.add_message(conversation_id, "t1", "user", "Long question")
    store.add_message(conversation_id, "t1", "assistant", "The first half of",
                      {"interrupted": True})
    at = AppTest.from_file(APP)
    at.query_params["c"] = conversation_id
    at = _run(at)
    text = _text(at)
    assert "The first half of" in text and chat.INTERRUPTED_NOTE in text
    assert chat.NO_REPLY_NOTE not in text


def test_provider_errors_are_saved_so_the_turn_is_not_lost(monkeypatch):
    from langchain_core.exceptions import ModelRateLimitError

    _use_model(monkeypatch, ModelRateLimitError("429"))
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("Will fail"))
    saved = _store().messages(at.session_state["thread_id"])
    assert saved[1]["error"] is True and not saved[0].get("interrupted")


def test_pending_approval_is_restored_after_a_restart(monkeypatch):
    executed = []
    original = tools_pkg.build_default_registry

    def registry_with_write_tool(settings, **kwargs):
        registry = original(settings, **kwargs)
        registry.register(make_tool(
            "send_note", lambda a: executed.append(a.text) or {"results": []},
            permission=Permission.EXTERNAL_WRITE, side_effect=SideEffect.IRREVERSIBLE))
        return registry

    monkeypatch.setattr(tools_pkg, "build_default_registry", registry_with_write_tool)
    _use_model(monkeypatch, tool_call("send_note", {"text": "hi"}))
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("send a note"))
    conversation_id = at.session_state["thread_id"]
    assert executed == [] and at.session_state["pending_approval"] is not None

    _restart()
    at = AppTest.from_file(APP)
    at.query_params["c"] = conversation_id
    at = _run(at)
    assert any("Approval required" in w.value for w in at.warning)
    assert chat.NO_REPLY_NOTE not in _text(at)
    at = _run(_button(at, "Approve this exact action").click())
    assert executed == ["hi"]
    roles = [m["role"] for m in _store().messages(conversation_id)]
    assert roles == ["user", "assistant"]


def test_data_page_questions_are_not_added_to_saved_conversations(monkeypatch):
    _use_model(monkeypatch, AIMessage("page answer"))
    from omniflow.ui import state

    state.get_page_chatbot()  # builds without touching the persistent checkpointer
    assert _store().list() == []


# ------------------------------------------------------------------ images
class _Upload:
    def __init__(self, name, data):
        self.name, self._data = name, data

    def getvalue(self):
        return self._data


def _png(color=(10, 120, 200)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (40, 30), color).save(out, format="PNG")
    return out.getvalue()


def test_attach_images_validates_each_file_and_reports_safely(monkeypatch):
    _use_model(monkeypatch)
    conversation_id = new_conversation_id()
    reports = chat.attach_images(conversation_id, [
        _Upload("chart.png", _png()),
        _Upload("notes.png", b"GIF89a not really a png"),
        _Upload("chart-again.png", _png()),  # same content: not stored twice
    ])
    assert reports[0].startswith("✅") and reports[1].startswith("❌")
    assert len(_store().images(conversation_id)) == 1


def test_image_question_uses_only_this_conversations_images(monkeypatch):
    model = _use_model(
        monkeypatch,
        tool_call("image_analyze", {"question": "What colour is it?"}),
        AIMessage("The image is a solid blue rectangle."),  # the vision call
        AIMessage("It appears to be a solid blue rectangle [1]."),
    )
    store = _store()
    conversation_id = new_conversation_id()
    prepared = prepare_image(_png(), "blue.png", max_bytes=10**7, max_pixels=10**7,
                             max_side=1536)
    image = store.add_image(conversation_id, name=prepared.name, mime=prepared.mime,
                            width=prepared.width, height=prepared.height,
                            sha256=prepared.sha256, data=prepared.data, max_images=6)
    store.add_message(conversation_id, "t0", "user", "I attached an image")
    store.add_message(conversation_id, "t0", "assistant", "Got it.")

    at = AppTest.from_file(APP)
    at.query_params["c"] = conversation_id
    at = _run(at)
    at = _run(at.chat_input[0].set_value("What colour is the image?"))

    planner_prompt = model.calls[0][0].content
    assert image.id in planner_prompt and "image_analyze" in model.bound_tools[0]
    vision_blocks = [b for b in model.calls[1][1].content if b["type"] == "image"]
    assert len(vision_blocks) == 1
    text = _text(at)
    assert "blue rectangle" in text and "may contain errors" in text

    # A different conversation has no images, so the tool is not even offered there.
    at = _run(_button(at, "➕ New chat", sidebar=True).click())
    model.replies.append(AIMessage("No image here."))
    at = _run(at.chat_input[0].set_value("What colour is the image?"))
    assert "image_analyze" not in model.bound_tools[-1]
