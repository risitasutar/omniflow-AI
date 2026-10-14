"""Streamlit UI tests via AppTest (runs app.py in-process, no browser, no network)."""

from pathlib import Path

import pytest
import streamlit as st
from langchain_core.exceptions import ModelRateLimitError
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

import omniflow.tools as tools_pkg
from omniflow.core import llm
from omniflow.tools import web_search
from omniflow.tools.registry import Permission, SideEffect
from tests.conftest import FAKE_KEY
from tests.fakes import ScriptedChatModel, make_tool, tool_call

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(autouse=True)
def clear_streamlit_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def _run(target) -> AppTest:
    """Run an AppTest (or a widget after an interaction) and fail on any app exception."""
    at = target.run(timeout=60)
    assert not at.exception, at.exception
    return at


def _all_text(at: AppTest) -> str:
    groups = (at.markdown, at.caption, at.error, at.info, at.warning)
    return "\n".join(str(e.value) for group in groups for e in group)


def _use_model(monkeypatch, *replies):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    model = ScriptedChatModel(replies=list(replies))
    monkeypatch.setattr(llm, "create_chat_runnable", lambda settings: model)
    return model


def test_missing_key_shows_setup_help_and_disables_chat():
    at = _run(AppTest.from_file(APP))
    assert at.title[0].value.endswith("OmniFlow AI")
    assert any("Set up Gemini" in i.value for i in at.info)
    assert any("not set" in e.value for e in at.sidebar.error)
    assert at.chat_input[0].disabled
    assert "web_search" in " ".join(c.value for c in at.sidebar.caption)


def test_chat_reply_shows_no_tools_route(monkeypatch):
    _use_model(monkeypatch, AIMessage("Hi from the fake model"))
    at = _run(AppTest.from_file(APP))
    assert not at.chat_input[0].disabled
    at = _run(at.chat_input[0].set_value("Hello"))

    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    text = _all_text(at)
    assert "Hi from the fake model" in text
    assert "no tools used" in text
    assert FAKE_KEY not in text


def test_search_turn_shows_executed_tool_evidence_and_verification(monkeypatch):
    monkeypatch.setattr(web_search, "_ddgs_text", lambda q, n, t: [
        {"title": "Python 3.14 released", "href": "https://www.python.org/news/",
         "body": "Release notes."},
    ])
    _use_model(
        monkeypatch,
        tool_call("web_search", {"query": "latest python"}),
        AIMessage("Python 3.14 is out [1]."),
    )
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("latest python release?"))
    text = _all_text(at)
    assert "Python 3.14 is out [1]." in text
    assert "Executed `web_search`" in text
    assert "Grounded" in text
    assert "https://www.python.org/news/" in text
    assert any("Evidence" in e.label for e in at.expander)


def test_provider_error_is_shown_safely(monkeypatch):
    _use_model(monkeypatch, ModelRateLimitError(f"429 quota exceeded for key {FAKE_KEY}"))
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("Hello"))

    errors = " ".join(e.value for e in at.error)
    assert "rate limit" in errors
    assert FAKE_KEY not in _all_text(at)


def test_whitespace_only_input_is_rejected(monkeypatch):
    model = _use_model(monkeypatch)
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("   "))
    assert any("enter a message" in w.value for w in at.warning)
    assert len(at.chat_message) == 0
    assert model.calls == []


def test_new_chat_clears_conversation(monkeypatch):
    _use_model(monkeypatch, AIMessage("first answer"))
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("Hello"))
    old_thread = at.session_state["thread_id"]

    at = _run(at.sidebar.button[0].click())
    assert at.session_state["messages"] == []
    assert at.session_state["thread_id"] != old_thread


def _with_write_tool(monkeypatch, executed):
    original = tools_pkg.build_default_registry

    def registry_with_write_tool(settings, **kwargs):
        registry = original(settings, **kwargs)
        registry.register(make_tool(
            "send_note", lambda a: executed.append(a.text) or {"results": []},
            permission=Permission.EXTERNAL_WRITE, side_effect=SideEffect.IRREVERSIBLE,
        ))
        return registry

    monkeypatch.setattr(tools_pkg, "build_default_registry", registry_with_write_tool)


def test_approval_card_blocks_until_approved(monkeypatch):
    executed = []
    _with_write_tool(monkeypatch, executed)
    _use_model(monkeypatch, tool_call("send_note", {"text": "hi bob"}))
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("send bob a note"))

    assert executed == []
    assert any("Approval required" in w.value for w in at.warning)
    assert at.chat_input[0].disabled  # no new requests while a decision is pending
    approve = next(b for b in at.button if b.label.startswith("Approve"))
    at = _run(approve.click())
    assert executed == ["hi bob"]
    assert at.session_state["pending_approval"] is None
    assert not at.chat_input[0].disabled


def test_rejecting_approval_never_executes(monkeypatch):
    executed = []
    _with_write_tool(monkeypatch, executed)
    _use_model(monkeypatch, tool_call("send_note", {"text": "hi"}))
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("send a note"))
    reject = next(b for b in at.button if b.label == "Reject")
    at = _run(reject.click())
    assert executed == []
    assert "not approved" in _all_text(at)


# ------------------------------------------------------------ Documents page
def _documents_page():
    from omniflow.ui import documents

    documents.render()


def _with_fake_embedder(monkeypatch):
    from omniflow.documents import embeddings
    from tests.fakes import FakeEmbedder

    monkeypatch.setattr(embeddings, "create_embedder", lambda *a, **k: FakeEmbedder())


def test_documents_page_renders_empty_state_and_ocr_help():
    at = _run(AppTest.from_function(_documents_page))
    assert at.title[0].value.startswith("📄")
    text = _all_text(at)
    assert "No documents yet" in text
    assert "runs locally" in text
    assert any("OCR" in e.label for e in at.expander) or "Local OCR available" in text


def test_documents_page_lists_removes_and_previews_retrieval(monkeypatch):
    import uuid

    from omniflow.ui import state
    from tests.pdf_fixtures import make_pdf

    monkeypatch.setenv("OMNIFLOW_RAG_MIN_SCORE", "0.2")
    _with_fake_embedder(monkeypatch)
    sid = uuid.uuid4().hex
    service = state.get_document_service()
    service.ingest(sid, "handbook.pdf", make_pdf(["Holiday policy gives twenty days of leave."]))
    service.ingest(sid, "other.pdf", make_pdf(["Cafeteria menu for Tuesday and Wednesday."]))

    at = AppTest.from_function(_documents_page)
    at.session_state["doc_session_id"] = sid
    at = _run(at)
    text = _all_text(at)
    assert "handbook.pdf" in text and "other.pdf" in text
    assert any("(2/10)" in h.value for h in at.subheader)

    at.text_input(key="retrieval_preview").set_value("holiday leave days")
    at = _run(at)
    assert "handbook.pdf · page 1" in _all_text(at)

    remove = next(b for b in at.button if b.key and b.key.startswith("remove_")
                  and service.documents(sid)[0].doc_id in b.key)
    at = _run(remove.click())
    assert len(service.documents(sid)) == 1


def test_documents_page_is_session_isolated(monkeypatch):
    import uuid

    from omniflow.ui import state
    from tests.pdf_fixtures import make_pdf

    _with_fake_embedder(monkeypatch)
    service = state.get_document_service()
    service.ingest(uuid.uuid4().hex, "someone-else.pdf",
                   make_pdf(["Private text belonging to another session."]))
    at = _run(AppTest.from_function(_documents_page))  # a fresh session of its own
    assert "someone-else.pdf" not in _all_text(at)
    assert "No documents yet" in _all_text(at)


def test_assistant_offers_documents_only_mode_when_pdfs_are_loaded(monkeypatch):
    import uuid

    from omniflow.ui import state
    from tests.pdf_fixtures import make_pdf

    monkeypatch.setenv("OMNIFLOW_RAG_MIN_SCORE", "0.2")
    _with_fake_embedder(monkeypatch)
    model = _use_model(monkeypatch, AIMessage("Twenty days [1]."))
    sid = uuid.uuid4().hex
    state.get_document_service().ingest(
        sid, "handbook.pdf", make_pdf(["Holiday policy gives twenty days of leave."])
    )
    at = AppTest.from_file(APP)
    at.session_state["doc_session_id"] = sid
    at = _run(at)
    at = _run(at.toggle(key="docs_only").set_value(True))
    at = _run(at.chat_input[0].set_value("holiday leave days"))
    text = _all_text(at)
    assert "handbook.pdf, page 1" in text
    assert "Grounded" in text
    assert len(model.calls) == 1  # documents-only mode: synthesis only, no planner call


# --------------------------------------------------------- Data + SQL pages
DATA_PAGE = "from omniflow.ui import data_page\ndata_page.render()"
SQL_PAGE = "from omniflow.ui import sql_page\nsql_page.render()"


def _seed_dataset():
    import uuid

    from omniflow.ui import state
    from tests.data_fixtures import SMALL_CSV

    sid = uuid.uuid4().hex
    assert state.get_data_service().ingest(sid, "small.csv", SMALL_CSV).status == "loaded"
    return sid


def test_data_page_empty_state():
    at = _run(AppTest.from_string(DATA_PAGE))
    assert at.title[0].value.startswith("📊")
    assert any("No datasets yet" in i.value for i in at.info)


def test_data_page_profile_preview_and_quick_chart():
    sid = _seed_dataset()
    at = AppTest.from_string(DATA_PAGE)
    at.session_state["data_session_id"] = sid
    at = _run(at)
    assert [m.value for m in at.metric][:2] == ["6", "6"]
    assert len(at.dataframe) >= 2  # profile + preview tables
    at.selectbox(key="qc_type").set_value("bar")
    at = _run(at)
    at.selectbox(key="qc_x").set_value("product")
    at = _run(at.button(key="qc_draw").click())
    assert not at.error


def test_data_page_question_uses_data_tools(monkeypatch):
    model = _use_model(  # sets the key before any settings are read and cached
        monkeypatch,
        tool_call("dataset_analyze", {"dataset": "small.csv", "operation": "group_aggregate",
                                      "group_by": ["region"], "value_column": "units"}),
        AIMessage("North sold 11 units [1]."),
    )
    sid = _seed_dataset()
    at = AppTest.from_string(DATA_PAGE)
    at.session_state["data_session_id"] = sid
    at = _run(at)
    at = _run(at.chat_input(key="data_question").set_value("Which region sold the most units?"))
    text = _all_text(at)
    assert "North sold 11 units" in text and "Grounded" in text
    assert "web_search" not in model.bound_tools[0]


def _submit(at, form_key):
    button = next(b for b in at.button if b.form_id == form_key)
    return button.click()


def test_sql_page_shows_sample_label_schema_and_runs_manual_select():
    at = _run(AppTest.from_string(SQL_PAGE))
    assert any("SAMPLE DATA" in i.value for i in at.info)
    assert any("Schema browser" in e.label for e in at.expander)
    at.text_area[0].set_value("SELECT category, COUNT(*) AS n FROM products GROUP BY category")
    at = _run(_submit(at, "sql_manual_form"))
    assert any("row(s)" in s.value for s in at.success)


def test_sql_page_blocks_destructive_manual_sql():
    at = _run(AppTest.from_string(SQL_PAGE))
    at.text_area[0].set_value("DROP TABLE orders")
    at = _run(_submit(at, "sql_manual_form"))
    assert any("Blocked before execution" in e.value for e in at.error)


def test_sql_page_question_shows_generated_sql(monkeypatch):
    _use_model(
        monkeypatch,
        tool_call("sql_query", {"sql": "SELECT COUNT(*) AS n FROM orders"}),
        AIMessage("There are 600 orders [1]."),
    )
    at = _run(AppTest.from_string(SQL_PAGE))
    at = _run(at.chat_input(key="sql_question").set_value("How many orders?"))
    assert "There are 600 orders" in _all_text(at)
    assert any("SELECT COUNT(*) AS n FROM orders" in c.value for c in at.code)


# ------------------------------------------- Phase 4 fixes: pinned input + SQL import
BOTTOM = 3  # AppTest tree: 0 main, 1 sidebar, 2 event, 3 bottom (pinned) container


def _input_is_pinned(at, key: str) -> bool:
    bottom = at._tree.children[BOTTOM].chat_input
    main = at._tree.children[0].chat_input
    return any(c.key == key and not c.disabled for c in bottom) and not list(main)


def test_data_page_supports_many_consecutive_questions_including_failures(monkeypatch):
    from langchain_core.exceptions import ModelRateLimitError

    group = {"dataset": "small.csv", "operation": "group_aggregate", "group_by": ["region"],
             "value_column": "units"}
    _use_model(
        monkeypatch,
        tool_call("dataset_analyze", group), AIMessage("First: North sold 11 [1]."),
        tool_call("dataset_analyze", group | {"group_by": ["no_such_column"]}),  # tool fails
        ModelRateLimitError("429"),  # provider error
        tool_call("dataset_chart", {"dataset": "small.csv", "chart_type": "bar",
                                    "x": "product", "y": "units"}),
        AIMessage("Fourth: chart [1]."),
        tool_call("dataset_analyze", group), AIMessage("Fifth: North sold 11 [1]."),
    )
    sid = _seed_dataset()
    at = AppTest.from_string(DATA_PAGE)
    at.session_state["data_session_id"] = sid
    at = _run(at)
    assert _input_is_pinned(at, "data_question")
    for n in range(1, 6):
        at = _run(at.chat_input(key="data_question").set_value(f"question {n}"))
        assert _input_is_pinned(at, "data_question"), f"input lost after question {n}"
        assert len(at.session_state["data_results"]) == 2 * n
        assert at.chat_message[-1].name == "assistant"  # newest answer is last on the page
        assert [m.value for m in at.metric][:2] == ["6", "6"]  # dataset still loaded
    roles = [e["role"] for e in at.session_state["data_results"]]
    assert roles == ["user", "assistant"] * 5
    errors = [e for e in at.session_state["data_results"] if e.get("error")]
    assert len(errors) == 1  # the rate-limit turn
    assert "Unknown column" in at.session_state["data_results"][3]["content"]
    assert "Fifth" in at.session_state["data_results"][-1]["content"]


def test_data_page_clear_conversation_keeps_dataset(monkeypatch):
    _use_model(monkeypatch, tool_call("dataset_profile", {"dataset": "small.csv"}),
               AIMessage("Six rows [1]."))
    sid = _seed_dataset()
    at = AppTest.from_string(DATA_PAGE)
    at.session_state["data_session_id"] = sid
    at = _run(at)
    at = _run(at.chat_input(key="data_question").set_value("profile it"))
    clear = next(b for b in at.button if b.label == "Clear conversation")
    at = _run(clear.click())
    assert at.session_state["data_results"] == []
    assert [m.value for m in at.metric][:2] == ["6", "6"]
    assert _input_is_pinned(at, "data_question")


def _import_loaded_dataset(at):
    at.radio(key="sql_import_source").set_value("A dataset loaded on the Data Analysis page")
    at = _run(at)
    button = next(b for b in at.button if b.label.startswith("📥 Import"))
    return _run(button.click())


def test_sql_page_imports_a_loaded_dataset_and_lists_its_schema():
    sid = _seed_dataset()
    at = AppTest.from_string(SQL_PAGE)
    at.session_state["data_session_id"] = sid
    at = _run(at)
    at = _import_loaded_dataset(at)
    assert any("Imported" in s.value and "small" in s.value for s in at.success)
    assert at.selectbox(key="sql_db").value == "imported_data"
    assert any("Your imported data" in i.value for i in at.info)
    assert any(md.value.startswith("**small**") for md in at.markdown)  # schema browser
    from omniflow.ui import state

    db = state.get_sql_service().imported_database(sid)
    assert db.allowed_tables == {"small"}


def test_sql_page_manual_queries_on_imported_table_stay_read_only():
    sid = _seed_dataset()
    at = AppTest.from_string(SQL_PAGE)
    at.session_state["data_session_id"] = sid
    at = _import_loaded_dataset(_run(at))
    at.text_area[0].set_value("SELECT region, SUM(units) AS u FROM small GROUP BY region")
    at = _run(_submit(at, "sql_manual_form"))
    assert any("3 row(s)" in s.value for s in at.success)
    at.text_area[0].set_value("DELETE FROM small")
    at = _run(_submit(at, "sql_manual_form"))
    assert any("Blocked before execution" in e.value for e in at.error)


def test_sql_page_agent_queries_only_the_selected_imported_database(monkeypatch):
    model = _use_model(
        monkeypatch,
        tool_call("sql_query", {"sql": "SELECT COUNT(*) AS n FROM small"}),
        AIMessage("There are 6 rows [1]."),
    )
    sid = _seed_dataset()
    at = AppTest.from_string(SQL_PAGE)
    at.session_state["data_session_id"] = sid
    at = _import_loaded_dataset(_run(at))
    at = _run(at.chat_input(key="sql_question").set_value("How many rows?"))
    assert "There are 6 rows" in _all_text(at)
    assert _input_is_pinned(at, "sql_question")
    system = model.calls[0][0].content
    assert '"imported_data"' in system and '"orders"' not in system  # only the selection
    assert set(model.bound_tools[0]) <= {"sql_schema", "sql_query", "ask_user_for_clarification",
                                         "report_unsupported_request"}


def test_sql_page_delete_imports_returns_to_sample():
    sid = _seed_dataset()
    at = AppTest.from_string(SQL_PAGE)
    at.session_state["data_session_id"] = sid
    at = _import_loaded_dataset(_run(at))
    delete = next(b for b in at.button if b.label.startswith("🗑️ Delete imported"))
    at = _run(delete.click())
    assert at.selectbox(key="sql_db").value == "sample_shop"
    assert any("SAMPLE DATA" in i.value for i in at.info)


# ------------------------------------------ Arrow display of mixed-type columns
def _pivot_xlsx() -> bytes:
    """Mimics an Excel pivot-table export: 'Column Labels' mixes numbers and text."""
    import io

    import pandas as pd

    rows = [["Sum of Sales", "Column Labels", "Grand Total"],
            ["Row Labels", "Tier 1", None],
            ["Dairy", 1200, 1200],
            ["Snacks", 980, 980],
            ["Fruits", "(blank)", 0],
            ["Grand Total", 2180, 2180]]
    buffer = io.BytesIO()
    pd.DataFrame(rows).to_excel(buffer, header=False, index=False, engine="openpyxl")
    return buffer.getvalue()


def test_display_frame_converts_only_mixed_columns_and_leaves_data_untouched():
    import pandas as pd
    import pyarrow as pa

    from omniflow.ui.artifacts import display_frame

    frame = pd.DataFrame({"mixed": [1200, "Tier 1", None, 980],
                          "numbers": [1.5, 2.0, None, 4.0], "text": ["a", "b", None, "d"]})
    shown = display_frame(frame)
    pa.Table.from_pandas(shown)  # would raise ArrowTypeError without the fix
    values = shown["mixed"].tolist()
    assert values[0] == "1200" and values[1] == "Tier 1" and values[3] == "980"
    assert pd.isna(values[2])  # missing stays missing (shown as an empty cell)
    assert shown["numbers"].dtype == frame["numbers"].dtype  # homogeneous columns unchanged
    assert frame["mixed"].tolist()[0] == 1200  # the original is not modified


def test_pivot_style_workbook_renders_without_arrow_errors(caplog):
    import logging
    import uuid

    from omniflow.ui import state

    sid = uuid.uuid4().hex
    assert state.get_data_service().ingest(sid, "pivot.xlsx", _pivot_xlsx()).status == "loaded"
    stored = state.get_data_service().datasets(sid)[0].frame["Column Labels"]
    assert {type(v) for v in stored.dropna()} == {int, str}  # analysis data keeps real types

    with caplog.at_level(logging.WARNING):
        at = AppTest.from_string(DATA_PAGE)
        at.session_state["data_session_id"] = sid
        at = _run(at)
        sql = AppTest.from_string(SQL_PAGE)
        sql.session_state["data_session_id"] = sid
        sql = _run(sql)
        sql.radio(key="sql_import_source").set_value("A dataset loaded on the Data Analysis page")
        sql = _run(sql)
    assert "ArrowTypeError" not in caplog.text
    assert "Serialization of dataframe to Arrow table was unsuccessful" not in caplog.text


# --------------------------------------------- Phase 5: news + papers in the Assistant
def test_assistant_shows_news_and_paper_evidence(monkeypatch):
    import json
    from datetime import UTC, datetime, timedelta

    from omniflow.research import academic, news

    published = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    monkeypatch.setattr(news, "OFFICIAL_FEEDS", ())
    monkeypatch.setattr(news, "_ddgs_news", lambda q, n, t, timeout: [  # SYNTHETIC TEST DATA
        {"title": "AI lab ships model", "url": "https://news.example.org/x", "body": "Short.",
         "source": "Example Wire", "date": published}])
    monkeypatch.setattr(academic, "_openalex_get", lambda p, t: (json.dumps({"results": [{
        "id": "W1", "doi": "10.5555/test.1", "display_name": "A synthetic study",
        "publication_year": 2025, "authorships": [{"author": {"display_name": "Jane Doe"}}],
        "primary_location": {"source": {"display_name": "Test Journal"}},
        "abstract_inverted_index": None}]}).encode(), {}))
    monkeypatch.setattr(academic, "_arxiv_get", lambda p, t: (
        b'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'))
    plan = AIMessage(content="", tool_calls=[
        {"name": "news_search", "args": {"query": "AI"}, "id": "n"},
        {"name": "academic_search", "args": {"query": "AI"}, "id": "a"}])
    _use_model(monkeypatch, plan, AIMessage("News [1] and research [2]."))
    at = _run(AppTest.from_file(APP))
    at = _run(at.chat_input[0].set_value("AI news and papers"))
    labels = " ".join(e.label for e in at.expander)
    assert "news item(s)" in labels and "paper(s) - metadata and abstracts" in labels
    text = _all_text(at)
    assert "Example Wire" in text and "published" in text and "retrieved" in text
    assert "doi:10.5555/test.1" in text
    assert "No abstract available - metadata only" in text
    assert "Grounded" in text
