"""Timeout attribution, Cloudflare error surfacing and image delivery (no network).

Real diagnosis (2026-10-13): "generate an image of smiling cat" failed because the very
first Gemini call (the planner) raised httpx.ReadTimeout after ~62 s, so image_generate and
Cloudflare were never reached. A minimal Gemini call also failed after ~66 s: Gemini itself
was not responding. These tests pin down how such failures are attributed and shown.
"""

import io
import logging
from pathlib import Path

import httpx
import pytest
import streamlit as st
from langchain_core.messages import AIMessage
from PIL import Image
from streamlit.testing.v1 import AppTest

from omniflow.agent import runner
from omniflow.agent.graph import AgentLimits, build_agent_graph
from omniflow.core import errors, llm
from omniflow.core.config import Settings
from omniflow.integrations import images
from omniflow.integrations.base import IntegrationError
from omniflow.integrations.hub import IntegrationsHub
from omniflow.tools import build_default_registry
from omniflow.tools.integration_tools import make_integration_tools
from tests.conftest import FAKE_KEY
from tests.fakes import ScriptedChatModel, tool_call

APP = str(Path(__file__).resolve().parents[1] / "app.py")
CAT = "generate an image of smiling cat"
ACCOUNT = "b" * 32
TOKEN = "cf-FAKE-token-zyxwvutsrq"


@pytest.fixture(autouse=True)
def clear_streamlit_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def read_timeout() -> httpx.ReadTimeout:
    return httpx.ReadTimeout("The read operation timed out")


def jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (24, 24), "orange").save(buffer, format="JPEG")
    return buffer.getvalue()


# --------------------------------------------------------------- attribution
def test_tagged_planner_timeout_names_gemini_and_the_stage():
    exc = read_timeout()
    errors.tag_stage(exc, "planner", 62.4)
    info, stage = errors.describe_failure(exc)
    assert stage.stage == "planner" and info.category == "timeout"
    assert "Gemini took too long" in info.user_message
    assert "deciding which tools to use, after 62 s" in info.user_message


def test_untagged_timeouts_are_not_blamed_on_gemini():
    info, stage = errors.describe_failure(TimeoutError("something else"))
    assert stage is None and "not a Gemini call" in info.user_message
    assert "Gemini took too long" not in info.user_message


def test_other_errors_keep_their_meaning_with_the_stage():
    from langchain_core.exceptions import ModelRateLimitError

    exc = ModelRateLimitError("429")
    errors.tag_stage(exc, "synthesize", 3.2)
    info, _ = errors.describe_failure(exc)
    assert info.category == "rate_limited"
    assert "writing the answer from the tool results" in info.user_message


def _image_graph(monkeypatch, *replies, generate=None):
    monkeypatch.setenv("OMNIFLOW_IMAGE_PROVIDER", "cloudflare")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", ACCOUNT)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", TOKEN)
    calls = []
    monkeypatch.setattr(images.CloudflareImages, "generate",
                        generate or (lambda self, p, s: calls.append(p) or jpeg()))
    settings = Settings(_env_file=None)
    registry = build_default_registry(settings)
    for spec in make_integration_tools(IntegrationsHub(settings)):
        registry.register(spec)
    model = ScriptedChatModel(replies=list(replies))
    graph = build_agent_graph(model, registry, limits=AgentLimits(max_tool_rounds=3,
                                                                  max_steps=14))
    return graph, calls


def test_planner_timeout_propagates_with_type_kept_and_no_tool_runs(monkeypatch):
    graph, calls = _image_graph(monkeypatch, read_timeout())
    with pytest.raises(httpx.ReadTimeout) as err:
        list(runner.stream_turn(graph, "t", CAT, owner_id="local"))
    assert err.value.omniflow_stage.stage == "planner"
    assert calls == []  # Cloudflare is never contacted when the planner fails


def test_llm_failures_are_logged_with_stage_and_duration(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="omniflow")
    graph, _ = _image_graph(monkeypatch, read_timeout())
    with pytest.raises(httpx.ReadTimeout):
        list(runner.stream_turn(graph, "t", CAT, owner_id="local"))
    assert "event=llm_call_failed node=planner error=ReadTimeout" in caplog.text


# ---------------------------------------------------------------- Cloudflare
def test_cloudflare_calls_log_status_and_timing_without_secrets(caplog):
    caplog.set_level(logging.INFO, logger="omniflow")
    client = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(401, json={"errors": [{"code": 10000}]})))
    with pytest.raises(IntegrationError, match="HTTP 401"):
        images.CloudflareImages(ACCOUNT, TOKEN, client).generate("a cat", 4)
    assert "event=image_provider_request provider=cloudflare" in caplog.text
    assert "event=image_provider_response provider=cloudflare status=401" in caplog.text
    assert TOKEN not in caplog.text and ACCOUNT not in caplog.text and "a cat" not in caplog.text


def test_cloudflare_timeout_is_a_tool_error_not_a_gemini_timeout(monkeypatch):
    def slow(self, prompt, steps):
        raise IntegrationError("timeout", "Cloudflare didn't answer within 60 seconds; no "
                               "image was generated.")

    from omniflow.agent.approval import fingerprint

    graph, _ = _image_graph(monkeypatch, tool_call("image_generate", {"prompt": "smiling cat"}),
                            AIMessage("x"), generate=slow)
    list(runner.stream_turn(graph, "t", CAT, owner_id="local"))
    outcome = runner.turn_outcome(graph, "t")
    list(runner.stream_turn(graph, "t", None, owner_id="local", resume={
        "approved": True, "fingerprint": fingerprint(
            "image_generate", outcome.pending_approval["args"])}))
    outcome = runner.turn_outcome(graph, "t")
    assert outcome.tool_results[0]["error_category"] == "timeout"
    assert "Cloudflare didn't answer within 60 seconds" in outcome.final_answer
    assert "Gemini took too long" not in outcome.final_answer


# ------------------------------------------------------------------- UI path
def _assistant(monkeypatch, *replies, generate=None):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("OMNIFLOW_IMAGE_PROVIDER", "cloudflare")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", ACCOUNT)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", TOKEN)
    monkeypatch.setattr(images.CloudflareImages, "generate",
                        generate or (lambda self, p, s: jpeg()))
    model = ScriptedChatModel(replies=list(replies))
    monkeypatch.setattr(llm, "create_chat_runnable", lambda settings: model)
    return model


def _run(target) -> AppTest:
    at = target.run(timeout=60)
    assert not at.exception, at.exception
    return at


def _text(at) -> str:
    return "\n".join(str(e.value) for g in (at.markdown, at.caption, at.error, at.warning)
                     for e in g)


def test_ui_names_the_stage_when_the_planner_times_out(monkeypatch):
    _assistant(monkeypatch, read_timeout())
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value(CAT))
    message = next(e.value for e in at.error)
    assert "Gemini took too long" in message and "deciding which tools to use" in message


def test_image_is_still_delivered_if_gemini_times_out_after_generation(monkeypatch):
    _assistant(monkeypatch, tool_call("image_generate", {"prompt": "smiling cat"}),
               read_timeout())  # the synthesis call after the tool times out
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value(CAT))
    at = _run(next(b for b in at.button if b.label == "Approve this exact action").click())
    message = next(e.value for e in at.error)
    assert "writing the answer from the tool results" in message
    text = _text(at)
    assert "already finished before the failure" in text and "Cloudflare Workers AI" in text
    assert "MOCK" not in text and at.get("image")


def test_successful_generation_delivers_the_real_artifact(monkeypatch):
    _assistant(monkeypatch, tool_call("image_generate", {"prompt": "smiling cat"}),
               AIMessage("Here is your smiling cat [1]."))
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value(CAT))
    at = _run(next(b for b in at.button if b.label == "Approve this exact action").click())
    assert not at.error
    assert "Image generated with Cloudflare Workers AI" in _text(at) and at.get("image")
