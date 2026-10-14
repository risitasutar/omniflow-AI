"""DuckDuckGo web search tool - all network access is mocked."""

import pytest
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
from pydantic import ValidationError

from omniflow.core.config import Settings
from omniflow.tools import web_search
from omniflow.tools.registry import AvailabilityStatus, Permission, SideEffect, ToolError
from omniflow.tools.web_search import WebSearchArgs, make_web_search_tool, normalise_results

# Captured at import, before the autouse network guard replaces it.
REAL_DDGS_TEXT = web_search._ddgs_text

RAW = [
    {"title": "Python", "href": "https://www.python.org/", "body": "Official site."},
    {"title": "Dup", "href": "https://www.python.org", "body": "Same URL."},
    {"title": "JS", "href": "javascript:alert(1)", "body": "bad"},
    {"title": "FTP", "href": "ftp://x.org/f", "body": "bad"},
    {"title": "Spaces", "href": "https://x.org/a b", "body": "bad"},
    {"title": "Docs\x00\x07", "href": "https://docs.python.org/3/", "body": "y" * 900},
]


def _tool(**overrides):
    return make_web_search_tool(Settings(_env_file=None, **overrides))


def test_metadata_is_read_only_and_untrusted():
    tool = _tool()
    assert tool.permission is Permission.EXTERNAL_READ
    assert tool.side_effect is SideEffect.NONE
    assert tool.untrusted_output is True
    assert tool.timeout_seconds > 10  # wall-clock cap above the HTTP timeout


@pytest.mark.parametrize("query", ["", " ", "x", "q" * 301])
def test_query_validation(query):
    with pytest.raises(ValidationError):
        WebSearchArgs(query=query)


def test_query_is_cleaned():
    assert WebSearchArgs(query="  hello\x00   world \n").query == "hello world"


def test_normalise_filters_dedupes_truncates_and_numbers():
    results = normalise_results(RAW, 10)
    assert [r["url"] for r in results] == ["https://www.python.org/", "https://docs.python.org/3/"]
    assert [r["id"] for r in results] == [1, 2]
    assert results[1]["title"] == "Docs"
    assert len(results[1]["snippet"]) <= 500


def test_successful_search_returns_structured_snippets(monkeypatch):
    seen = {}

    def fake(query, n, timeout):
        seen.update(query=query, n=n, timeout=timeout)
        return RAW

    monkeypatch.setattr(web_search, "_ddgs_text", fake)
    out = _tool(web_search_max_results=3).func(WebSearchArgs(query="python", max_results=10))
    assert seen == {"query": "python", "n": 3, "timeout": 10.0}  # capped by settings
    assert out["query"] == "python"
    assert "snippets" in out["content_type"]
    assert len(out["results"]) == 2


def test_ddgs_is_pinned_to_duckduckgo_backend(monkeypatch):
    import ddgs

    captured = {}

    class FakeDDGS:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        def text(self, query, **kwargs):
            captured.update(kwargs)
            return []

    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
    REAL_DDGS_TEXT("q", 3, 10)  # the real wrapper, talking to the fake client
    assert captured["backend"] == "duckduckgo"
    assert captured["max_results"] == 3
    assert captured["safesearch"] == "moderate"
    assert captured["timeout"] == 10


@pytest.mark.parametrize(
    "exc,category,retryable",
    [
        (RatelimitException("429"), "rate_limited", True),
        (TimeoutException("slow"), "timeout", True),
        (DDGSException("engine broke"), "provider_error", True),
    ],
)
def test_provider_errors_are_classified(monkeypatch, exc, category, retryable):
    def boom(*_a):
        raise exc

    monkeypatch.setattr(web_search, "_ddgs_text", boom)
    with pytest.raises(ToolError) as err:
        _tool().func(WebSearchArgs(query="python"))
    assert err.value.category == category
    assert err.value.retryable is retryable


def test_no_results_is_an_empty_success(monkeypatch):
    def none(*_a):
        raise DDGSException("No results found.")

    monkeypatch.setattr(web_search, "_ddgs_text", none)
    assert _tool().func(WebSearchArgs(query="zzqx"))["results"] == []


def test_availability(monkeypatch):
    assert _tool().availability().status is AvailabilityStatus.AVAILABLE
    assert _tool(web_search_enabled=False).availability().status is AvailabilityStatus.DISABLED
    monkeypatch.setattr(web_search.importlib.util, "find_spec", lambda name: None)
    assert _tool().availability().status is AvailabilityStatus.MISSING_DEPENDENCY
