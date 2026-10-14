"""Optional live check: Gemini proposes plans through the planner tool (nothing is saved).

Skipped unless you opt in:
    $env:OMNIFLOW_RUN_LIVE_TESTS = "1"
    .\\.venv\\Scripts\\python.exe -m pytest tests/test_live_planner.py
Uses about 4-6 free-tier Gemini requests. Model output varies between runs.
"""

import os
from datetime import date, timedelta

import pytest

from omniflow.agent import runner
from omniflow.agent.graph import build_agent_graph
from omniflow.core import llm
from omniflow.core.config import PROJECT_ROOT, Settings
from omniflow.planner.service import PlannerService
from omniflow.planner.store import PlannerStore
from omniflow.tools import build_default_registry
from omniflow.tools.planner_tools import make_planner_tools

pytestmark = pytest.mark.live


@pytest.fixture
def live(tmp_path):
    if os.environ.get("OMNIFLOW_RUN_LIVE_TESTS") != "1":
        pytest.skip("set OMNIFLOW_RUN_LIVE_TESTS=1 to run live planner tests")
    settings = Settings(_env_file=PROJECT_ROOT / ".env")
    if not settings.gemini_configured:
        pytest.skip("GEMINI_API_KEY not set in .env")
    service = PlannerService(PlannerStore(tmp_path / "planner.db"))
    registry = build_default_registry(settings)
    for spec in make_planner_tools(service):
        registry.register(spec)
    return build_agent_graph(llm.create_chat_runnable(settings), registry), service


def _plan(graph, text, thread):
    list(runner.stream_turn(graph, thread, text, answer_mode="planner", owner_id="me"))
    return runner.turn_outcome(graph, thread)


def test_live_plan_without_a_deadline_invents_none(live):
    graph, service = live
    outcome = _plan(graph, "Create a plan to finish a small Streamlit application. I haven't "
                           "set any deadline.", "a")
    [proposal] = service.proposals("me")
    assert len(proposal.plan.tasks) >= 3
    assert not any(t.date_source == "user" for t in proposal.plan.tasks)
    assert service.store.list_projects("me") == []
    assert outcome.final_answer


def test_live_user_deadline_is_respected(live):
    graph, service = live
    deadline = date.today() + timedelta(days=21)
    _plan(graph, f"Help me finish my data analytics portfolio. My final deadline is "
                 f"{deadline.isoformat()}.", "b")
    [proposal] = service.proposals("me")
    dated = [date.fromisoformat(t.due_date) for t in proposal.plan.tasks if t.due_date]
    assert all(d <= deadline for d in dated), dated
