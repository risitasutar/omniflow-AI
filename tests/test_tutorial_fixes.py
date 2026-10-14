"""Behavioural checks for the fixes applied to the original CampusX tutorial backends.

Skipped automatically when the optional tutorial dependencies
(requirements-tutorials.txt) are not installed.
"""

import importlib
import sys

import pytest
import requests

pytest.importorskip("langchain_openai")
pytest.importorskip("langchain_community")
pytest.importorskip("ddgs")


@pytest.fixture
def import_tutorial(monkeypatch, tmp_path):
    """Import a tutorial backend safely: no real .env, no repo-local chatbot.db."""
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.chdir(tmp_path)  # the backends create chatbot.db in the CWD
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    loaded = []

    def _import(name):
        sys.modules.pop(name, None)
        module = importlib.import_module(name)
        loaded.append(name)
        return module

    yield _import
    for name in loaded:
        module = sys.modules.pop(name, None)
        conn = getattr(module, "conn", None)
        if conn is not None:
            conn.close()


class FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"Global Quote": {"01. symbol": "AAPL"}}


def test_stock_tool_reads_key_from_env_and_sets_timeout(import_tutorial, monkeypatch):
    backend = import_tutorial("langgraph_tool_backend")
    captured = {}

    def fake_get(url, **kwargs):
        captured.update(url=url, **kwargs)
        return FakeResponse()

    monkeypatch.setattr(backend.requests, "get", fake_get)
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "test-av-key")

    result = backend.get_stock_price.invoke({"symbol": "AAPL"})
    assert result == {"Global Quote": {"01. symbol": "AAPL"}}
    assert captured["params"] == {"function": "GLOBAL_QUOTE", "symbol": "AAPL",
                                  "apikey": "test-av-key"}
    assert captured["timeout"] == 10
    assert "apikey" not in captured["url"]


def test_stock_tool_without_key_returns_error_and_makes_no_request(import_tutorial, monkeypatch):
    backend = import_tutorial("langgraph_tool_backend")
    monkeypatch.setattr(backend.requests, "get", lambda *a, **k: pytest.fail("no request"))
    result = backend.get_stock_price.invoke({"symbol": "AAPL"})
    assert "ALPHAVANTAGE_API_KEY is not set" in result["error"]


def test_stock_tool_error_does_not_leak_key(import_tutorial, monkeypatch):
    backend = import_tutorial("langgraph_tool_backend")

    def failing_get(url, **kwargs):
        raise requests.ConnectionError(f"failed {url}?apikey={kwargs['params']['apikey']}")

    monkeypatch.setattr(backend.requests, "get", failing_get)
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "test-av-key")
    result = backend.get_stock_price.invoke({"symbol": "AAPL"})
    assert "test-av-key" not in str(result)


def test_rag_backend_imports_on_langchain_1x(import_tutorial):
    """Regression: `langchain.text_splitter` no longer exists in LangChain 1.x."""
    backend = import_tutorial("langraph_rag_backend")
    assert backend.RecursiveCharacterTextSplitter.__module__.startswith("langchain_text_splitters")
