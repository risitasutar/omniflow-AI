"""UI redesign: brand header, dropdown navigation, advanced details and suggested prompts."""

from pathlib import Path

import pytest
import streamlit as st
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from omniflow.core import llm
from omniflow.tools.registry import AvailabilityStatus
from omniflow.ui import theme
from tests.conftest import FAKE_KEY
from tests.fakes import ScriptedChatModel

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["Assistant", "Documents", "Data Analysis", "SQL", "Planner", "Code", "Memory",
         "Integrations"]


@pytest.fixture(autouse=True)
def clear_streamlit_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def _run(at: AppTest) -> AppTest:
    at = at.run(timeout=60)
    assert not at.exception, at.exception
    return at


def _nav(at: AppTest):
    return next(s for s in at.sidebar.selectbox if s.label == "Go to")


def test_dropdown_lists_every_page_and_shows_the_current_one():
    at = _run(AppTest.from_file(APP))
    nav = _nav(at)
    assert [label.split("  ", 1)[1] for label in nav.options] == PAGES
    assert nav.value == 0 and at.title[0].value.endswith("OmniFlow AI")


@pytest.mark.parametrize("index, title_start", [(1, "📄"), (3, "🗄️"), (6, "🧠"),
                                                (7, "🔌")])
def test_choosing_a_page_in_the_dropdown_switches_to_it(index, title_start):
    at = _run(AppTest.from_file(APP))
    at = _run(_nav(at).set_value(index))
    assert at.title[0].value.startswith(title_start)
    assert _nav(at).value == index  # the dropdown now shows the page that is open
    at = _run(_nav(at).set_value(0))  # and back to the Assistant
    assert at.title[0].value.endswith("OmniFlow AI") and _nav(at).value == 0


def test_page_changes_made_without_the_dropdown_keep_it_in_sync():
    """Opening a saved conversation from another page switches to the Assistant through the
    existing go_assistant flow; the dropdown must follow. (AppTest can only switch to
    file-based pages, so this is the in-app equivalent of a deep link.)"""
    at = _run(AppTest.from_file(APP))
    at = _run(_nav(at).set_value(4))
    assert at.title[0].value.startswith("🗂️")
    at.session_state["go_assistant"] = True
    at = _run(at)
    assert at.title[0].value.endswith("OmniFlow AI") and _nav(at).value == 0


def test_brand_header_and_friendly_status_in_the_sidebar():
    at = _run(AppTest.from_file(APP))
    assert "Gemini not set up" in str(at.sidebar) or any(
        "not set" in e.value for e in at.sidebar.error)
    advanced = next(e for e in at.sidebar.expander if "Advanced details" in e.label)
    inside = " ".join(c.value for c in advanced.caption)
    assert "`web_search`" in inside and "Ready" in inside  # raw ids live here only
    outside = [c.value for c in at.sidebar.caption if c not in list(advanced.caption)]
    assert not any("external_read" in c or "not_ready" in c for c in outside)


def test_raw_states_are_translated():
    assert theme.status_label(AvailabilityStatus.NOT_READY) == "Waiting for your input"
    assert theme.status_label(AvailabilityStatus.MISSING_CREDENTIALS) == "Needs setup"
    assert theme.tool_label("currency_convert") == "Currency conversion"
    assert theme.tool_label("some_new_tool") == "Some new tool"


def test_suggested_prompt_is_sent_like_typed_text(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    model = ScriptedChatModel(replies=[AIMessage("Here is what I can do.")])
    monkeypatch.setattr(llm, "create_chat_runnable", lambda settings: model)
    at = _run(AppTest.from_file(APP))
    button = next(b for b in at.button if "What can you do?" in b.label)
    at = _run(button.click())
    # /help is answered deterministically, exactly as when typed.
    assert at.session_state["messages"][0]["content"] == "/help"
    assert not any("What can you do?" in b.label for b in at.button)  # suggestions hidden


def test_documents_page_keeps_ocr_notice_and_moves_model_details():
    at = _run(AppTest.from_file(APP))
    at = _run(_nav(at).set_value(1))
    labels = [e.label for e in at.expander]
    assert any("OCR" in label for label in labels)
    advanced = next(e for e in at.main.expander if "Advanced details" in e.label)
    assert any("Embedding model" in c.value for c in advanced.caption)


def test_integrations_page_lists_existing_services_with_plain_status():
    at = _run(AppTest.from_file(APP))
    at = _run(_nav(at).set_value(7))
    text = " ".join(m.value for m in at.markdown)
    for service in ("Tavily research", "Weather (Open-Meteo)", "Currency (Frankfurter)"):
        assert service in text
    assert "Needs setup" in text and "Ready" in text  # Tavily has no key in tests
    assert "only external integration" not in " ".join(c.value for c in at.caption)
