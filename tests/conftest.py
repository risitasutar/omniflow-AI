"""Shared fixtures.

Every test runs with an isolated configuration: the developer's real `.env` is never
read and Gemini/OmniFlow environment variables are cleared, so tests never use a real
API key or make live calls (except tests explicitly marked `live`).
"""

from __future__ import annotations

import os

import pytest

from omniflow.core import config

_PREFIXES = ("GEMINI_", "GOOGLE_API_KEY", "OMNIFLOW_")


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    for name in list(os.environ):
        # Keep the explicit opt-in switch for live tests; clear everything else.
        if name.startswith(_PREFIXES) and name != "OMNIFLOW_RUN_LIVE_TESTS":
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OMNIFLOW_ENV_FILE", str(tmp_path / "does-not-exist.env"))
    # The sample SQLite database is generated in a temp dir, never in the repository.
    monkeypatch.setenv("OMNIFLOW_SQL_SAMPLE_PATH", str(tmp_path / "sample" / "sample.db"))
    # Saved conversations and agent checkpoints also go to the temp dir.
    monkeypatch.setenv("OMNIFLOW_CONVERSATIONS_DB", str(tmp_path / "history" / "conv.db"))
    monkeypatch.setenv("OMNIFLOW_CHECKPOINTS_DB", str(tmp_path / "history" / "ckpt.db"))
    monkeypatch.setenv("OMNIFLOW_MEMORY_DB", str(tmp_path / "history" / "memory.db"))
    monkeypatch.setenv("OMNIFLOW_CODE_WORKSPACE_DIR", str(tmp_path / "code_workspaces"))
    monkeypatch.setenv("OMNIFLOW_PLANNER_DB", str(tmp_path / "history" / "planner.db"))
    monkeypatch.setenv("OMNIFLOW_GENERATED_IMAGES_DIR", str(tmp_path / "generated_images"))
    for name in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN", "TAVILY_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


@pytest.fixture(autouse=True)
def no_live_web_search(monkeypatch):
    """Fail loudly if any test would reach the real search provider."""
    from omniflow.tools import web_search

    def _blocked(*_args, **_kwargs):
        raise AssertionError("live web search is disabled in tests; mock _ddgs_text")

    monkeypatch.setattr(web_search, "_ddgs_text", _blocked)


@pytest.fixture(autouse=True)
def no_live_tavily(monkeypatch):
    """Fail loudly if any test would reach the real Tavily API."""
    from omniflow.tools import tavily_search

    def _blocked(*_args, **_kwargs):
        raise AssertionError("live Tavily search is disabled in tests; mock _tavily_post")

    monkeypatch.setattr(tavily_search, "_tavily_post", _blocked)


@pytest.fixture(autouse=True)
def no_live_weather(monkeypatch):
    """Fail loudly if any test would reach the real Open-Meteo APIs."""
    from omniflow.tools import weather

    def _blocked(*_args, **_kwargs):
        raise AssertionError("live Open-Meteo is disabled in tests; mock _geocode_get and "
                             "_forecast_get")

    monkeypatch.setattr(weather, "_geocode_get", _blocked)
    monkeypatch.setattr(weather, "_forecast_get", _blocked)


@pytest.fixture(autouse=True)
def no_live_currency(monkeypatch):
    """Fail loudly if any test would reach the real Frankfurter API."""
    from omniflow.tools import currency

    def _blocked(*_args, **_kwargs):
        raise AssertionError("live Frankfurter is disabled in tests; mock _rate_get")

    monkeypatch.setattr(currency, "_rate_get", _blocked)


@pytest.fixture(autouse=True)
def no_real_embedding_model(monkeypatch):
    """Tests use FakeEmbedder; loading or downloading the real model fails loudly."""
    from omniflow.documents import embeddings

    def _blocked(*_args, **_kwargs):
        raise AssertionError("real embedding model is disabled in tests; use FakeEmbedder")

    monkeypatch.setattr(embeddings, "create_embedder", _blocked)


FAKE_KEY = "AIzaFAKE-test-key-0000000000000000000"


@pytest.fixture(autouse=True)
def no_live_research_providers(monkeypatch):
    """Fail loudly if any test would reach DuckDuckGo News, a feed, OpenAlex or arXiv."""
    from omniflow.research import academic, news

    def _blocked(*_args, **_kwargs):
        raise AssertionError("live research providers are disabled in tests; mock them")

    for module, name in [(news, "_ddgs_news"), (news, "_fetch_feed"),
                         (academic, "_openalex_get"), (academic, "_arxiv_get")]:
        monkeypatch.setattr(module, name, _blocked)
    news._CACHE._entries.clear()
