"""Integrations page (image generation only) and Assistant approval UI (AppTest, no network)."""

import re
from pathlib import Path

import pytest
import streamlit as st
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from omniflow.core import llm
from tests.conftest import FAKE_KEY
from tests.fakes import ScriptedChatModel, tool_call

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


def _page():
    from omniflow.ui import integrations_page

    integrations_page.render()


def _text(at) -> str:
    groups = (at.markdown, at.caption, at.error, at.warning, at.info, at.success)
    text = "\n".join(str(e.value) for group in groups for e in group)
    return re.sub(r"\\([\\`*_{}\[\]()#+!<>|~-])", r"\1", text)


def test_default_page_shows_disabled_image_generation_and_no_google():
    at = _run(AppTest.from_function(_page))
    text = _text(at)
    assert "Image generation · ⛔ Disabled" in text
    assert "no access to email or calendar accounts" in text
    assert "Google" not in text.replace("Google's", "") or "OAuth" not in text
    assert "GOOGLE_OAUTH" not in text and not at.toggle


def test_cloudflare_states(monkeypatch):
    monkeypatch.setenv("OMNIFLOW_IMAGE_PROVIDER", "cloudflare")
    at = _run(AppTest.from_function(_page))
    assert "Needs your setup" in _text(at) and "dash.cloudflare.com" in _text(at)
    st.cache_resource.clear()
    from omniflow.core import config

    config.get_settings.cache_clear()
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "a" * 32)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-FAKE-token-123456")
    at = _run(AppTest.from_function(_page))
    text = _text(at)
    assert "Configured" in text and "never substitutes a placeholder" in text
    assert "cf-FAKE-token-123456" not in text and "a" * 32 not in text


def test_mock_provider_is_labelled(monkeypatch):
    monkeypatch.setenv("OMNIFLOW_IMAGE_PROVIDER", "mock")
    at = _run(AppTest.from_function(_page))
    assert "MOCK (placeholder images)" in _text(at) and "not AI-generated" in _text(at)


def test_memory_proposal_from_the_agent_becomes_a_confirm_card(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    model = ScriptedChatModel(replies=[
        tool_call("memory_propose", {"content": "Weekly reviews are on Fridays",
                                     "category": "project"}),
        AIMessage("I can remember that once you confirm [1].")])
    monkeypatch.setattr(llm, "create_chat_runnable", lambda settings: model)
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("Please keep in mind weekly reviews are on Fridays"))
    at = _run(at)
    from omniflow.core.config import get_settings
    from omniflow.memory.store import MemoryStore

    assert MemoryStore(get_settings().memory_db_path).list("local") == []  # not yet saved
    at = _run(next(b for b in at.button if b.label == "Save it").click())
    assert [m.content for m in MemoryStore(get_settings().memory_db_path).list("local")] == [
        "Weekly reviews are on Fridays"]


def test_navigation_includes_integrations():
    at = _run(AppTest.from_file(APP))
    assert at.sidebar.caption
    assert "integrations_page.render" in Path(APP).read_text(encoding="utf-8")
