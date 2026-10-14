"""Regressions through the real Assistant page (app.py -> state.get_chatbot -> registry).

1. Image bug (2026-10-13): "generate an image of a cat smiling" produced a MOCK placeholder
   although OMNIFLOW_IMAGE_PROVIDER=cloudflare, because the leftover
   OMNIFLOW_INTEGRATIONS_MODE=mock overrode the provider in IntegrationsHub.image_info().
2. Gmail/Calendar were removed: such requests must get an accurate answer and no tools.
"""

import io
import re
from pathlib import Path

import pytest
import streamlit as st
from langchain_core.messages import AIMessage, SystemMessage
from PIL import Image
from streamlit.testing.v1 import AppTest

from omniflow.agent import prompts
from omniflow.core import llm
from omniflow.integrations import images
from omniflow.integrations.base import IntegrationError
from omniflow.tools.integration_tools import MOCK_LABEL, REAL_LABEL
from omniflow.tools.registry import PLANNED_CAPABILITIES
from tests.conftest import FAKE_KEY
from tests.fakes import ScriptedChatModel, tool_call

APP = str(Path(__file__).resolve().parents[1] / "app.py")
CAT = "generate an image of a cat smiling"


@pytest.fixture(autouse=True)
def clear_streamlit_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def _run(target) -> AppTest:
    at = target.run(timeout=60)
    assert not at.exception, at.exception
    return at


def _assistant(monkeypatch, *replies, **env):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    model = ScriptedChatModel(replies=list(replies))
    monkeypatch.setattr(llm, "create_chat_runnable", lambda settings: model)
    return model


def _cloudflare_env(monkeypatch):
    return {"OMNIFLOW_IMAGE_PROVIDER": "cloudflare", "OMNIFLOW_INTEGRATIONS_MODE": "mock",
            "CLOUDFLARE_ACCOUNT_ID": "a" * 32, "CLOUDFLARE_API_TOKEN": "cf-FAKE-token-123456"}


def _text(at) -> str:
    groups = (at.markdown, at.caption, at.error, at.warning, at.info, at.success)
    text = "\n".join(str(e.value) for group in groups for e in group)
    return re.sub(r"\\([\\`*_{}\[\]()#+!<>|~-])", r"\1", text)


def _saved_files():
    from omniflow.core.config import get_settings

    return [p for p in get_settings().generated_images_dir.glob("**/*") if p.is_file()]


def _approve(at):
    return _run(next(b for b in at.button if b.label == "Approve this exact action").click())


def test_cat_prompt_uses_cloudflare_not_the_mock(monkeypatch):
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "orange").save(buffer, format="JPEG")
    calls = []
    monkeypatch.setattr(images.CloudflareImages, "generate",
                        lambda self, prompt, steps: calls.append(prompt) or buffer.getvalue())
    model = _assistant(monkeypatch, tool_call("image_generate", {"prompt": "a cat smiling"}),
                       AIMessage("Here is your image [1]."), **_cloudflare_env(monkeypatch))
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value(CAT))
    preview = next(c.value for c in at.code if "Generate an image" in c.value)
    assert "Provider: cloudflare" in preview
    assert calls == [] and _saved_files() == []  # nothing before approval
    at = _approve(at)
    assert calls == ["a cat smiling"]
    [saved] = _saved_files()
    assert "cloudflare" in saved.name and saved.suffix == ".jpg"
    text = _text(at)
    assert REAL_LABEL in text and MOCK_LABEL not in text
    assert at.get("image")  # the generated image is displayed
    assert "image_generate" in model.bound_tools[0]


def test_cloudflare_failure_shows_the_error_and_no_mock(monkeypatch):
    def fail(self, prompt, steps):
        raise IntegrationError("auth_failed", "Cloudflare rejected the API token (HTTP 401). "
                               "Check that CLOUDFLARE_API_TOKEN is an API token. No image was "
                               "generated.")

    monkeypatch.setattr(images.CloudflareImages, "generate", fail)
    _assistant(monkeypatch, tool_call("image_generate", {"prompt": "a cat smiling"}),
               AIMessage("x"), **_cloudflare_env(monkeypatch))
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value(CAT))
    at = _approve(at)
    text = _text(at)
    assert "HTTP 401" in text and MOCK_LABEL not in text
    assert _saved_files() == []


def test_rejecting_the_image_generates_nothing(monkeypatch):
    calls = []
    monkeypatch.setattr(images.CloudflareImages, "generate",
                        lambda self, prompt, steps: calls.append(prompt))
    _assistant(monkeypatch, tool_call("image_generate", {"prompt": "a cat smiling"}),
               AIMessage("x"), **_cloudflare_env(monkeypatch))
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value(CAT))
    at = _run(next(b for b in at.button if b.label == "Reject").click())
    assert calls == [] and _saved_files() == [] and "no change was made" in _text(at)


def test_explicit_mock_provider_is_labelled_mock(monkeypatch):
    _assistant(monkeypatch, tool_call("image_generate", {"prompt": "a cat smiling"}),
               AIMessage("This is a MOCK placeholder [1]."), OMNIFLOW_IMAGE_PROVIDER="mock")
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value(CAT))
    at = _approve(at)
    assert MOCK_LABEL in _text(at) and REAL_LABEL not in _text(at)


def test_gmail_requests_get_an_accurate_answer_and_no_tools(monkeypatch):
    model = _assistant(monkeypatch, tool_call(prompts.UNSUPPORTED_TOOL, {
        "capability": "Gmail access", "explanation": "Email accounts aren't supported; paste "
        "the text or upload a document instead."}))
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("Turn the action items in my kickoff email into a "
                                         "plan."))
    assert not any(n.startswith(("gmail", "calendar")) for n in model.bound_tools[0])
    system = next(m.content for m in model.calls[0] if isinstance(m, SystemMessage))
    assert "Gmail, Google Calendar or any other email/calendar account access" in system
    assert "Phase 8" not in system
    assert "Gmail access" in _text(at)


def test_planned_list_is_accurate():
    listed = " ".join(PLANNED_CAPABILITIES).lower()
    assert "audio and video" in listed and "gmail" in listed
    assert "image generation" not in listed and "code assistant" not in listed


def test_unconfigured_image_generation_explains_setup(monkeypatch):
    model = _assistant(monkeypatch, AIMessage("Image generation needs setup first."),
                       OMNIFLOW_IMAGE_PROVIDER="cloudflare")
    at = _run(AppTest.from_file(APP))
    _run(at.chat_input[0].set_value(CAT))
    system = next(m.content for m in model.calls[0] if isinstance(m, SystemMessage))
    assert "image_generate" not in model.bound_tools[0]
    assert "Image generation: OMNIFLOW_IMAGE_PROVIDER=cloudflare, but" in system
