"""Optional live check of the DuckDuckGo search tool (keyless; no Gemini call).

    $env:OMNIFLOW_RUN_LIVE_TESTS = "1"; .\\.venv\\Scripts\\python.exe -m pytest -m live
"""

import os

import pytest

from omniflow.core.config import Settings
from omniflow.tools import web_search
from omniflow.tools.web_search import WebSearchArgs, make_web_search_tool

pytestmark = pytest.mark.live
REAL_DDGS_TEXT = web_search._ddgs_text  # captured before the autouse network guard


def test_real_duckduckgo_search(monkeypatch):
    if os.environ.get("OMNIFLOW_RUN_LIVE_TESTS") != "1":
        pytest.skip("set OMNIFLOW_RUN_LIVE_TESTS=1 to run live web search tests")
    monkeypatch.setattr(web_search, "_ddgs_text", REAL_DDGS_TEXT)
    out = make_web_search_tool(Settings(_env_file=None)).func(
        WebSearchArgs(query="Python programming language", max_results=3)
    )
    assert out["results"], "expected at least one result"
    assert all(r["url"].startswith(("https://", "http://")) for r in out["results"])
