"""Optional live check: Gemini explains computed data stories and forecasts.

Skipped unless you opt in:
    $env:OMNIFLOW_RUN_LIVE_TESTS = "1"
    .\\.venv\\Scripts\\python.exe -m pytest tests/test_live_data_story.py
Uses about 4-6 free-tier Gemini requests on a synthetic dataset. Output varies by run.
"""

import os
import re
import uuid

import pytest

from omniflow.agent import runner
from omniflow.agent.graph import build_agent_graph
from omniflow.core import llm
from omniflow.core.config import PROJECT_ROOT, Settings
from omniflow.core.sessions import SessionStore
from omniflow.data.service import DataService
from omniflow.tools import build_default_registry
from tests.test_data_story import sales_csv

pytestmark = pytest.mark.live


@pytest.fixture
def live():
    if os.environ.get("OMNIFLOW_RUN_LIVE_TESTS") != "1":
        pytest.skip("set OMNIFLOW_RUN_LIVE_TESTS=1 to run live data-story tests")
    settings = Settings(_env_file=PROJECT_ROOT / ".env")
    if not settings.gemini_configured:
        pytest.skip("GEMINI_API_KEY not set in .env")
    data = DataService(settings, SessionStore(dict, max_sessions=5, ttl_seconds=3600))
    sid = uuid.uuid4().hex
    assert data.ingest(sid, "sales.csv", sales_csv()).status == "loaded"
    registry = build_default_registry(settings, data=data)
    return build_agent_graph(llm.create_chat_runnable(settings), registry), sid


def _ask(graph, sid, text, thread):
    list(runner.stream_turn(graph, thread, text, data_session_id=sid, answer_mode="data"))
    return runner.turn_outcome(graph, thread)


def test_live_story_is_grounded_and_not_causal(live):
    graph, sid = live
    outcome = _ask(graph, sid, "Summarize the most important findings in sales.csv and "
                               "recommend actions.", "s")
    tools = {c["tool"] for c in outcome.planned_calls}
    assert tools & {"dataset_story", "dataset_trends", "dataset_profile", "dataset_analyze"}
    answer = outcome.final_answer
    assert "North" in answer
    assert not re.search(r"(?i)\b(caused by|because of the (ads|marketing))\b", answer)
    print("verification:", outcome.verification["status"], outcome.verification["notes"])


def test_live_forecast_asks_or_uses_explicit_columns(live):
    graph, sid = live
    outcome = _ask(graph, sid, "Forecast revenue for the next 3 months using the month "
                               "column.", "f")
    calls = [c for c in outcome.planned_calls if c["tool"] == "dataset_forecast"]
    assert calls, outcome.final_answer
    assert calls[0]["args"]["value_column"] == "revenue"
    assert "naive" in outcome.final_answer.lower()
