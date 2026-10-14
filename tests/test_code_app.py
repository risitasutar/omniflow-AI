"""Code page (AppTest): project overview, explain, checks with approval, patches, security."""

from pathlib import Path

import pytest
import streamlit as st
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from omniflow.code import evaluate as fx
from omniflow.core import llm
from tests.conftest import FAKE_KEY
from tests.fakes import ScriptedChatModel, tool_call

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


def _code_page():
    from omniflow.ui import code_page

    code_page.render()


def _text(at) -> str:
    groups = (at.markdown, at.caption, at.error, at.warning, at.info, at.success, at.code)
    return "\n".join(str(e.value) for group in groups for e in group)


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def _open(kind="upload", tmp_path=None):
    from omniflow.ui import state

    service = state.get_code_service()
    if kind == "upload":
        return service.load_zip(fx.zip_bytes(), "shop-main.zip", "local")
    root = fx.write_project(tmp_path / "shop")
    return service.load_local(str(root), "local")


def _page_with(ws) -> AppTest:
    at = AppTest.from_function(_code_page)
    at.session_state["code_workspace_id"] = ws.id
    return _run(at)


def test_empty_page_offers_upload_and_local_folder():
    at = _run(AppTest.from_function(_code_page))
    assert at.title[0].value == "💻 Code"
    assert at.radio(key="code_source").options == ["Upload a ZIP", "Open a local folder"]
    assert "credential files" in _text(at)


def test_shared_deployments_hide_local_folders(monkeypatch):
    monkeypatch.setenv("OMNIFLOW_ACCESS_MODE", "per_session")
    at = _run(AppTest.from_function(_code_page))
    assert at.radio(key="code_source").options == ["Upload a ZIP"]
    assert "always off on shared or hosted" in _text(at)


def test_overview_shows_files_skips_and_entry_points():
    at = _page_with(_open())
    text = _text(at)
    assert "shop-main.zip" in text and "untrusted - static analysis only" in text
    assert "app/main.py" in text and "secret or credential file" in text
    assert fx.ENV_SECRET not in text and fx.FAKE_GOOGLE_KEY not in text


def test_explain_runs_a_focused_cited_turn(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    model = ScriptedChatModel(replies=[tool_call("code_search", {"query": "unit price"}),
                                       AIMessage("Prices are computed in unit_price [1].")])
    monkeypatch.setattr(llm, "create_chat_runnable", lambda settings: model)
    at = _page_with(_open())
    at.text_area[0].set_value("Where is the unit price computed?")
    at = _run(_button(at, "Explain").click())
    text = _text(at)
    assert "Prices are computed in unit_price" in text and "app/orders.py" in text
    assert all(n.startswith("code_") for n in model.bound_tools[0]
               if n not in ("ask_user_for_clarification", "report_unsupported_request"))


def test_checks_show_the_command_and_need_approval(tmp_path):
    ws = _open("local", tmp_path)
    at = _page_with(ws)
    at = _run(at.radio(key="code_action").set_value("Run checks"))
    at = _run(at.selectbox(key="code_check").set_value("syntax"))
    at = _run(_button(at, "Prepare").click())
    at = _run(_button(at, "Run").click())
    assert "No syntax errors" in _text(at)


def test_pytest_is_not_available_for_uploads():
    at = _page_with(_open())
    at = _run(at.radio(key="code_action").set_value("Run checks"))
    at = _run(at.selectbox(key="code_check").set_value("pytest"))
    at = _run(_button(at, "Prepare").click())
    assert "only run in local folders you trust" in _text(at)
    assert not [b for b in at.button if b.label == "Run"]


def test_subprocess_check_requires_the_approval_tick(tmp_path):
    from omniflow.code import checks

    if not checks.ruff_available():
        pytest.skip("ruff not installed")
    at = _page_with(_open("local", tmp_path))
    at = _run(at.radio(key="code_action").set_value("Run checks"))
    at = _run(at.selectbox(key="code_check").set_value("ruff"))
    at = _run(_button(at, "Prepare").click())
    assert "python -m ruff check" in _text(at)
    assert _button(at, "Run").disabled
    at.checkbox[0].check()
    at = _run(at)
    at = _run(_button(at, "Run").click())
    assert "exit code" in _text(at)


def test_pasted_diff_review_apply_and_undo_on_an_upload():
    ws = _open()
    original = (ws.root / "app/orders.py").read_text()
    from omniflow.code import patches

    diff = patches.from_edits(ws, [{"path": "app/orders.py",
                                    "find": '    return order["total"] / order["quantity"]',
                                    "replace": '    return order["total"] / max(1, '
                                               'order["quantity"])'}], "guard").diff_text()
    at = _page_with(ws)
    at = _run(at.radio(key="code_action").set_value("Review patch"))
    diff_box = next(t for t in at.text_area if t.label == "Unified diff")
    diff_box.set_value(diff)
    at = _run(_button(at, "Check this diff").click())
    assert "Python syntax" in _text(at) and "max(1," in _text(at)
    assert (ws.root / "app/orders.py").read_text() == original  # review only
    assert _button(at, "Apply patch").disabled  # needs the approval tick
    next(c for c in at.checkbox if c.label.startswith("I reviewed this diff")).check()
    at = _run(at)
    at = _run(_button(at, "Apply patch").click())
    assert "max(1," in (ws.root / "app/orders.py").read_text()
    at = _run(_button(at, "Undo").click())
    assert (ws.root / "app/orders.py").read_text() == original


def test_local_patch_apply_is_off_on_shared_deployments(monkeypatch, tmp_path):
    monkeypatch.setenv("OMNIFLOW_CODE_PATCH_APPLY_ENABLED", "false")
    ws = _open("local", tmp_path)
    from omniflow.code import patches
    from omniflow.ui import state

    state.get_code_service().store_patch(patches.from_edits(ws, [{
        "path": "app/orders.py", "find": '    return order["total"] / order["quantity"]',
        "replace": "    return 0"}], "x"))
    at = _page_with(ws)
    at = _run(at.radio(key="code_action").set_value("Review patch"))
    assert "turned off here" in _text(at)
    assert not [b for b in at.button if b.label == "Apply patch"]


def test_security_review_lists_findings_without_secrets():
    at = _page_with(_open())
    at = _run(at.radio(key="code_action").set_value("Security review"))
    at = _run(_button(at, "Run security review").click())
    table = at.dataframe[0].value
    assert "PY003" in set(table["Rule"]) and "SEC001" in set(table["Rule"])
    assert fx.FAKE_GOOGLE_KEY not in table.to_string()


def test_close_project_deletes_the_extracted_copy():
    ws = _open()
    at = _page_with(ws)
    assert ws.root.exists()
    at = _run(_button(at, "Close project").click())
    assert not ws.root.exists()
    assert at.radio(key="code_source")


def test_another_session_cannot_open_the_project(monkeypatch):
    monkeypatch.setenv("OMNIFLOW_ACCESS_MODE", "per_session")
    from omniflow.ui import state

    ws = state.get_code_service().load_zip(fx.zip_bytes(), "s.zip", "s-someone-else")
    at = _page_with(ws)
    assert "expired or was closed" in _text(at)


def test_app_navigation_includes_the_code_page():
    at = _run(AppTest.from_file(APP))
    assert at.sidebar.caption  # the app ran
    assert "code_page.render" in Path(APP).read_text(encoding="utf-8")
