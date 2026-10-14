"""SQL page: pick a database, import your own CSV/XLSX, browse schemas, ask or run SELECTs.

Importing is an explicit button press handled here - it is never available to the model.
"""

from __future__ import annotations

import hashlib
import uuid

import pandas as pd
import streamlit as st

from omniflow.agent.verification import md_escape
from omniflow.core.config import get_settings
from omniflow.data.ingest import DataError, load_upload
from omniflow.sql.database import SqlExecutionError
from omniflow.sql.guard import SqlRejected
from omniflow.sql.importer import normalize_identifier, plan_columns
from omniflow.sql.service import IMPORTED_NAME
from omniflow.ui import chat, state
from omniflow.ui.artifacts import display_frame, render_table


def _clear_results() -> None:
    st.session_state["sql_results"] = []
    st.session_state.pop("sql_thread_id", None)


def _delete_imports() -> None:
    state.get_sql_service().clear_imports(st.session_state["data_session_id"])
    st.session_state["sql_import_report"] = None
    st.session_state.pop("sql_db", None)
    _clear_results()


def _label(db) -> str:
    if db.is_sample:
        return f"{db.name} - SAMPLE DATA (synthetic demo)"
    if db.is_imported:
        return f"{db.name} - your imported data (this session)"
    return f"{db.name} - configured database"


def _parsed_upload(upload) -> list:
    """Parse an upload once per file content (cached in session state, never on disk)."""
    data = upload.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    cache = st.session_state.setdefault("sql_upload_cache", {})
    if digest not in cache:
        cache.clear()  # keep at most one parsed upload in memory
        try:
            cache[digest] = ("ok", load_upload(upload.name, data,
                                               state.get_data_service().limits))
        except DataError as exc:
            cache[digest] = ("error", exc.safe_message)
    return cache[digest]


def _import_section() -> None:
    settings = get_settings()
    service = state.get_sql_service()
    session_id = st.session_state["data_session_id"]
    datasets = state.get_data_service().datasets(session_id)

    st.caption("Copy a CSV/XLSX into a private SQLite database for **this browser session "
               "only**, then ask questions about it here. Importing is a manual step; the AI "
               "can only read the imported tables.")
    sources = ["Upload a file here"]
    if datasets:
        sources.insert(0, "A dataset loaded on the Data Analysis page")
    source = st.radio("Source", sources, horizontal=True, key="sql_import_source")

    frame, label, warnings = None, "", []
    if source.startswith("A dataset"):
        names = [d.name for d in datasets]
        choice = st.selectbox("Dataset", names, key="sql_import_dataset")
        picked = datasets[names.index(choice)]
        frame, label, warnings = picked.frame, picked.name, picked.warnings
    else:
        upload = st.file_uploader(
            f"CSV or XLSX (up to {settings.data_max_file_mb:g} MB, "
            f"{settings.data_max_rows:,} rows, {settings.data_max_columns} columns)",
            type=["csv", "xlsx"], key=f"sql_upload_{st.session_state.get('sql_upload_key', 0)}",
        )
        if upload is not None:
            status, parsed = _parsed_upload(upload)
            if status == "error":
                st.error(f"❌ {md_escape(parsed)}")
                return
            tables = {t.name: t for t in parsed}
            sheet = st.selectbox("Sheet", list(tables), key="sql_import_sheet") \
                if len(tables) > 1 else next(iter(tables))
            frame, label, warnings = tables[sheet].frame, tables[sheet].name, tables[sheet].warnings
    if frame is None:
        return

    st.markdown(f"**Preview** - {len(frame):,} rows x {frame.shape[1]} columns")
    for warning in warnings:
        st.caption(f"⚠️ {md_escape(warning)}")
    st.dataframe(display_frame(frame.head(10)), width="stretch")
    requested = st.text_input("Table name", value=normalize_identifier(
        label.rsplit(".", 1)[0] if label.lower().endswith((".csv", ".xlsx")) else label,
        "imported_table"), max_chars=60, key="sql_import_name")
    st.caption(f"Will be created as `{normalize_identifier(requested, 'imported_table')}` "
               "(letters, digits and underscores only; a suffix is added if the name exists).")
    with st.expander("Column names and types in SQL"):
        st.dataframe(pd.DataFrame([{"original column": c.original, "SQL column": c.sql_name,
                                    "type": c.sql_type} for c in plan_columns(frame)]),
                     width="stretch", hide_index=True)
    if st.button("📥 Import into SQL (this session)", type="primary"):
        try:
            with st.spinner("Importing…"):
                report = service.import_frame(session_id, frame, requested, label)
        except SqlRejected as exc:
            st.error(f"❌ {md_escape(str(exc))}")
            return
        st.session_state["sql_import_report"] = {"table": report.table, "rows": report.rows,
                                                 "source": report.source}
        st.session_state["sql_db"] = f"{IMPORTED_NAME}"
        st.session_state["sql_upload_key"] = st.session_state.get("sql_upload_key", 0) + 1
        st.session_state.pop("sql_upload_cache", None)
        st.rerun()


def render() -> None:
    state.ensure_session()
    settings = get_settings()
    service = state.get_sql_service()
    session_id = st.session_state["data_session_id"]

    st.title("🗄️ SQL (read-only)")
    st.caption("Ask questions about a SQLite database. Every query is validated (one "
               "read-only SELECT, allowlisted tables) and runs on a read-only connection.")

    report = st.session_state.get("sql_import_report")
    if report:
        st.success(f"✅ Imported **{md_escape(report['source'])}** as table "
                   f"`{report['table']}` with {report['rows']:,} rows into `{IMPORTED_NAME}`.")

    with st.expander("📥 Import CSV/XLSX into SQL", expanded=not report):
        _import_section()

    for db in service.databases(include_unavailable=True):
        if not db.available:
            st.warning(f"Configured database `{db.name}` is not available: {db.problem}.")
    databases = service.databases(session_id=session_id)
    if not databases:
        st.info("No database yet. Import a CSV/XLSX above, enable the sample database "
                "(`OMNIFLOW_SQL_SAMPLE_ENABLED=true`), or configure one in `.env`.")
        return

    names = [d.name for d in databases]
    if st.session_state.get("sql_db") not in names:
        st.session_state["sql_db"] = names[-1] if IMPORTED_NAME in names else names[0]
    choice = st.selectbox("Database to query", names, key="sql_db",
                          format_func=lambda n: _label(service.get(n, session_id)),
                          on_change=_clear_results)
    db = service.get(choice, session_id)
    if db.is_sample:
        st.info("🧪 **SAMPLE DATA** - a synthetic online shop (fictional customers, products "
                "and orders). It is not your data.")
    elif db.is_imported:
        st.info("📄 **Your imported data** - private to this browser session, removed when you "
                f"delete it, after {settings.rag_session_ttl_minutes} minutes idle, or on "
                "restart.")
        st.button("🗑️ Delete imported tables", on_click=_delete_imports)
    else:
        st.caption(md_escape(db.description))

    try:
        schema = service.schema(db.name, session_id)
    except SqlExecutionError as exc:
        st.error(md_escape(str(exc)))
        return
    with st.expander(f"Schema browser - {len(schema)} table(s)", expanded=db.is_imported):
        for table in schema:
            st.markdown(f"**{md_escape(table['table'])}** · {table['rows']:,} rows")
            st.dataframe(pd.DataFrame([
                {"column": c["name"], "type": c["type"], "primary key": c["primary_key"],
                 "access": "restricted" if c["blocked"] else "readable"}
                for c in table["columns"]
            ]), width="stretch", hide_index=True)
        st.caption("Only permitted tables are listed. Restricted columns read as empty.")

    with st.expander("⌨️ Run your own SELECT (no AI call)"):
        with st.form("sql_manual_form"):
            sql = st.text_area("SQL", height=120, max_chars=settings.sql_max_query_chars,
                               placeholder="e.g. SELECT * FROM <table> LIMIT 10")
            run = st.form_submit_button("Validate and run")
        if run and sql.strip():
            try:
                result = service.query(db.name, sql, session_id)
            except SqlRejected as exc:
                st.error(f"🛑 Blocked before execution: {md_escape(str(exc))}")
            except SqlExecutionError as exc:
                st.error(f"❌ {md_escape(str(exc))}")
            else:
                status = f"✅ {result['row_count']} row(s) in {result['duration_ms']} ms"
                if result["truncated"]:
                    status += f" (stopped at the {result['max_rows']}-row limit)"
                st.success(status)
                st.code(result["sql"], language="sql")
                table = {"columns": result["columns"], "rows": result["rows"],
                         "total_rows": result["row_count"], "truncated": result["truncated"]}
                if result["rows"]:
                    render_table(table, key="sql_manual_table")
                else:
                    st.info("The query returned no rows.")

    st.subheader(f"💬 Ask about `{db.name}`")
    if not settings.gemini_configured:
        st.caption("Plain-English questions need the Gemini key; manual SELECTs don't.")

    def answer(question: str) -> None:
        st.session_state.setdefault("sql_thread_id", f"sql-{uuid.uuid4().hex}")
        chat.run_turn(st.session_state["sql_thread_id"], user_text=question,
                      data_session_id=session_id, answer_mode="sql",
                      history_key="sql_results", sql_database=db.name)

    chat.conversation(
        "sql_results", input_key="sql_question", enabled=settings.gemini_configured,
        placeholder=f"Ask about {db.name}, e.g. Which category has the highest total?",
        answer=answer,
    )
    if st.session_state["sql_results"]:
        st.button("Clear conversation", on_click=_clear_results)
