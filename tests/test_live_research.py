"""Optional live checks of the Phase 5 providers (no Gemini call; real network).

Skipped unless you opt in:
    $env:OMNIFLOW_RUN_LIVE_TESTS = "1"; .\\.venv\\Scripts\\python.exe -m pytest -m live
Uses about 1 OpenAlex search (10 of ~1,000 keyless daily credits), 1 arXiv request,
1-2 DuckDuckGo News requests and one download of each official feed.
"""

import os

import pytest

from omniflow.core.config import Settings
from omniflow.research import academic, news
from omniflow.tools.research_tools import AcademicSearchArgs, NewsSearchArgs, make_research_tools

pytestmark = pytest.mark.live
# Captured at import, before the autouse network guards replace them.
REAL = {"ddg": news._ddgs_news, "feed": news._fetch_feed,
        "openalex": academic._openalex_get, "arxiv": academic._arxiv_get}


@pytest.fixture
def live(monkeypatch):
    if os.environ.get("OMNIFLOW_RUN_LIVE_TESTS") != "1":
        pytest.skip("set OMNIFLOW_RUN_LIVE_TESTS=1 to run live research tests")
    monkeypatch.setattr(news, "_ddgs_news", REAL["ddg"])
    monkeypatch.setattr(news, "_fetch_feed", REAL["feed"])
    monkeypatch.setattr(academic, "_openalex_get", REAL["openalex"])
    monkeypatch.setattr(academic, "_arxiv_get", REAL["arxiv"])
    return make_research_tools(Settings(_env_file=None))


def test_live_official_feeds_return_dated_items(live):
    news_tool, _ = live
    out = news_tool.func(NewsSearchArgs(query="health", sources="official", max_age_days=60))
    assert not [p for p in out["provider_problems"] if p["category"] != "no_results"], \
        out["provider_problems"]
    assert all(r["url"].startswith("https://") for r in out["results"])


def test_live_academic_search_returns_metadata(live):
    _, academic_tool = live
    out = academic_tool.func(AcademicSearchArgs(query="retrieval augmented generation",
                                                max_results=5))
    assert out["results"], out["provider_problems"]
    paper = out["results"][0]
    assert paper["title"] and paper["url"] and paper["source_provider"] in ("OpenAlex", "arXiv")
