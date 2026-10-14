"""Planner page (AppTest): proposals, explicit saves, previews, confirmations, failures."""

import re
from pathlib import Path

import pytest
import streamlit as st
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from omniflow.core import llm
from tests.conftest import FAKE_KEY
from tests.fakes import ScriptedChatModel, tool_call
from tests.test_planner_agent import PLAN_ARGS
from tests.test_planner_core import plan

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
    from omniflow.ui import planner_page

    planner_page.render()


def _service():
    from omniflow.ui import state

    service, problem = state.planner_service_or_none()
    assert service is not None, problem
    return service


def _text(at) -> str:
    groups = (at.markdown, at.caption, at.error, at.warning, at.info, at.success)
    text = "\n".join(str(e.value) for group in groups for e in group)
    return re.sub(r"\\([\\`*_{}\[\]()#+!<>|~-])", r"\1", text)  # undo Markdown escaping


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def _saved_project():
    service = _service()
    project_id, _ = service.save_proposal("local", "p", plan(), name="Portfolio",
                                          accept_assumed_dates=False, token="seed")
    return service, project_id


def _open(project_id) -> AppTest:
    at = AppTest.from_function(_page)
    at.session_state["planner_project_id"] = project_id
    return _run(at)


def test_empty_page_offers_planning_and_manual_projects():
    at = _run(AppTest.from_function(_page))
    assert at.title[0].value == "🗂️ Planner"
    assert "doesn't send reminders" in _text(at)
    assert any(b.label == "Generate plan" for b in at.button)


def test_generating_a_plan_shows_a_proposal_that_is_not_saved(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    model = ScriptedChatModel(replies=[tool_call("planner_propose_plan", PLAN_ARGS),
                                       AIMessage("I proposed a plan [1]; review and save it.")])
    monkeypatch.setattr(llm, "create_chat_runnable", lambda settings: model)
    at = _run(AppTest.from_function(_page))
    at.text_area[0].set_value("Create a plan to finish a Streamlit application")
    at = _run(_button(at, "Generate plan").click())
    assert "No final deadline was given" in model.calls[0][-1].content
    at = _run(at)  # the next rerun shows the proposal card
    assert "Proposed plan: Streamlit app" in _text(at)
    assert _service().store.list_projects("local") == []
    assert _button(at, "Save project").disabled  # needs the review tick


def test_saving_a_reviewed_proposal_needs_the_tick_and_happens_once():
    service = _service()
    proposal = service.keep_proposal("local", plan())
    at = _run(AppTest.from_function(_page))
    tick = next(c for c in at.checkbox if c.label.startswith("I reviewed this plan"))
    at = _run(tick.check())
    at = _run(_button(at, "Save project").click())
    assert "Saved project with 4 task(s)" in _text(at)
    [project] = service.store.list_projects("local")
    assert project["tasks"] == 4
    # Saving the same proposal again (rerun or double submit) creates nothing new.
    service.save_proposal("local", proposal.id, plan(), name="Portfolio",
                          accept_assumed_dates=False, token=f"save-{proposal.id}")
    assert len(service.store.list_projects("local")) == 1


def test_project_view_shows_computed_progress_and_next_task():
    _, project_id = _saved_project()
    at = _open(project_id)
    text = _text(at)
    assert "Portfolio" in text and "lowest-level tasks" in text
    assert any(m.label == "Complete" and m.value == "0%" for m in at.metric)
    at = _run(_button(at, "What should I work on next?").click())
    text = _text(at)
    assert "Pick datasets" in text and "High priority (+30)" in text


def test_editing_a_task_previews_then_applies_after_a_click():
    service, project_id = _saved_project()
    at = _open(project_id)
    pick_id = next(t.id for t in service.store.load("local", project_id).tasks.values()
                   if t.title == "Pick datasets")
    at = _run(at.selectbox(key="planner_edit_task").set_value(pick_id))
    status = next(s for s in at.selectbox if s.label == "Status")
    status.set_value("Completed")
    at = _run(_button(at, "Review changes").click())
    assert "status Not Started -> Completed" in _text(at)
    assert service.store.load("local", project_id).tasks[pick_id].status == "Not Started"
    at = _run(_button(at, "Apply changes").click())
    assert "Saved:" in _text(at)
    assert service.store.load("local", project_id).tasks[pick_id].status == "Completed"
    assert not [b for b in at.button if b.label == "Apply changes"]  # nothing pending now


def test_deleting_a_task_needs_an_extra_confirmation():
    service, project_id = _saved_project()
    at = _open(project_id)
    dash_id = next(t.id for t in service.store.load("local", project_id).tasks.values()
                   if t.title == "Build dashboard")
    at = _run(at.selectbox(key="planner_edit_task").set_value(dash_id))
    at = _run(_button(at, "Delete this task…").click())
    assert "Delete task 'Build dashboard' and 1 subtask(s)" in _text(at)
    assert _button(at, "Apply changes").disabled
    at = _run(_button(at, "Cancel").click())
    assert dash_id in service.store.load("local", project_id).tasks


def test_assistant_suggested_changes_wait_for_the_user():
    service, project_id = _saved_project()
    from omniflow.planner.changes import Op

    task = next(t for t in service.store.load("local", project_id).tasks.values()
                if t.title == "Polish README")
    service.preview("local", project_id, [Op(op="update_task", task_id=task.id,
                                             fields={"priority": "High"})], source="assistant")
    at = _open(project_id)
    assert "Suggested by OmniFlow" in _text(at)
    assert service.store.load("local", project_id).tasks[task.id].priority == "Low"
    at = _run(_button(at, "Apply changes").click())
    assert service.store.load("local", project_id).tasks[task.id].priority == "High"


def test_archive_and_delete_with_typed_confirmation():
    service, project_id = _saved_project()
    at = _open(project_id)
    at = _run(_button(at, "Archive").click())
    assert service.store.list_projects("local") == []
    at = _run(at)
    assert any(b.label == "Reopen project" for b in at.button)
    delete = _button(at, "Delete project permanently")
    assert delete.disabled
    at.text_input(key="planner_delete_name").set_value("Portfolio")
    at = _run(at)
    at = _run(_button(at, "Delete project permanently").click())
    assert service.store.load("local", project_id) is None


def test_timeline_and_board_render_with_partial_dates():
    _, project_id = _saved_project()
    at = _open(project_id)
    text = _text(at)
    assert "Faded bars are suggested dates" in text
    assert "To do (" in text


def test_unavailable_storage_is_reported(monkeypatch, tmp_path):
    folder = tmp_path / "not-a-file"
    folder.mkdir()
    monkeypatch.setenv("OMNIFLOW_PLANNER_DB", str(folder))
    at = _run(AppTest.from_function(_page))
    assert any("could not be opened" in e.value for e in at.error)


def test_planner_turned_off(monkeypatch):
    monkeypatch.setenv("OMNIFLOW_PLANNER_ENABLED", "false")
    at = _run(AppTest.from_function(_page))
    assert any("turned off" in e.value for e in at.error)


def test_projects_are_private_per_session(monkeypatch):
    monkeypatch.setenv("OMNIFLOW_ACCESS_MODE", "per_session")
    service = _service()
    service.store.create_project("s-someone-else", name="Their project", token="x")
    at = _run(AppTest.from_function(_page))
    assert "Their project" not in str(at.selectbox[0].options)
    assert "private to this browser session" in _text(at)


def test_app_navigation_includes_the_planner():
    at = _run(AppTest.from_file(APP))
    assert at.sidebar.caption  # the app ran
    assert "planner_page.render" in Path(APP).read_text(encoding="utf-8")
