"""Data Analysis page: upload CSV/XLSX, profile, preview, charts and questions."""

from __future__ import annotations

import uuid

import pandas as pd
import streamlit as st

from omniflow.agent.verification import md_escape
from omniflow.core.config import get_settings
from omniflow.data.analysis import AnalysisError
from omniflow.data.charts import ChartArgs
from omniflow.ui import chat, data_story_ui, state
from omniflow.ui.artifacts import display_frame, render_chart


def _process(files) -> None:
    service = state.get_data_service()
    session_id = st.session_state["data_session_id"]
    reports = []
    with st.status(f"Loading {len(files)} file(s)…", expanded=True) as box:
        for upload in files:
            report = service.ingest(session_id, upload.name, upload.getvalue())
            box.write(f"{md_escape(report.filename)}: {report.status}")
            reports.append((report.filename, report.status, report.message))
        box.update(label="Done", state="complete", expanded=False)
    st.session_state["data_reports"] = reports
    st.session_state["data_uploader_key"] += 1


def _clear_all() -> None:
    state.get_data_service().clear(st.session_state["data_session_id"])
    state.new_data_session()
    st.session_state["data_reports"] = []
    st.session_state["data_results"] = []
    st.session_state.pop("data_thread_id", None)


def _clear_conversation() -> None:
    st.session_state["data_results"] = []
    st.session_state.pop("data_thread_id", None)


def _remove(dataset_id: str) -> None:
    state.get_data_service().remove(st.session_state["data_session_id"], dataset_id)
    st.session_state["data_reports"] = []


def _profile_table(profile: dict) -> pd.DataFrame:
    rows = []
    for c in profile["column_profiles"]:
        stats = c.get("stats", {})
        top = ", ".join(f"{t['value']} ({t['count']})" for t in c.get("top_values", [])[:3])
        rows.append({
            "column": c["name"], "role": c["role"], "missing": c["missing"],
            "missing %": c["missing_pct"], "unique": c["unique"],
            "mean": stats.get("mean"), "median": stats.get("50%"), "min": stats.get("min"),
            "max": stats.get("max"), "outliers (IQR)": c.get("outliers", {}).get("count"),
            "top values": top or (f"{c['range']['min']} → {c['range']['max']}"
                                  if "range" in c else ""),
        })
    return pd.DataFrame(rows)


def _quick_chart(dataset) -> None:
    columns = [str(c) for c in dataset.frame.columns]
    numeric = [c for c, r in dataset.roles.items() if r == "numeric"]
    kind = st.selectbox("Chart type", ["bar", "line", "histogram", "scatter", "heatmap"],
                        key="qc_type")
    x = y = ""
    agg = "sum"
    if kind == "histogram":
        x = st.selectbox("Numeric column", numeric or ["(none)"], key="qc_hist")
    elif kind == "scatter":
        x = st.selectbox("X (numeric)", numeric or ["(none)"], key="qc_sx")
        y = st.selectbox("Y (numeric)", numeric or ["(none)"], key="qc_sy")
    elif kind in ("bar", "line"):
        x = st.selectbox("X (category or date)", columns, key="qc_x")
        y = st.selectbox("Y (numeric; blank = count rows)", ["", *numeric], key="qc_y")
        agg = st.selectbox("Aggregation", ["sum", "mean", "median", "count", "min", "max"],
                           key="qc_agg")
    if st.button("Draw chart", key="qc_draw"):
        try:
            args = ChartArgs(dataset=dataset.name, chart_type=kind, x=x if x != "(none)" else "",
                             y=y if y != "(none)" else "", aggregation=agg)
            _, recipe = state.get_data_service().chart(st.session_state["data_session_id"], args)
            render_chart(recipe)
        except (AnalysisError, ValueError) as exc:
            st.error(md_escape(str(exc)))


def render() -> None:
    state.ensure_session()
    settings = get_settings()
    service = state.get_data_service()
    session_id = st.session_state["data_session_id"]

    st.title("📊 Data Analysis")
    st.caption("Upload CSV or Excel files. **Every number is computed locally by pandas**; "
               "Gemini only interprets your question and explains the computed results.")

    files = st.file_uploader(
        f"Add CSV / XLSX files (up to {settings.data_max_files} per session, "
        f"{settings.data_max_file_mb:g} MB, {settings.data_max_rows:,} rows and "
        f"{settings.data_max_columns} columns each)",
        type=["csv", "xlsx"], accept_multiple_files=True,
        key=f"data_uploader_{st.session_state['data_uploader_key']}",
    )
    if files and st.button(f"Load {len(files)} file(s)", type="primary"):
        _process(files)
        st.rerun()
    for name, status, message in st.session_state["data_reports"]:
        show = st.error if status == "rejected" else st.success if status == "loaded" else st.info
        show(f"**{md_escape(name)}** - {md_escape(message)}")

    datasets = service.datasets(session_id)
    if not datasets:
        st.info("No datasets yet. Upload a CSV or XLSX file to begin.")
        return

    names = [d.name for d in datasets]
    choice = st.selectbox("Dataset", names, key="data_selected")
    dataset = datasets[names.index(choice)] if choice in names else datasets[0]
    p = dataset.profile
    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{p['rows']:,}")
    b.metric("Columns", p["columns"])
    c.metric("Duplicate rows", f"{p['duplicate_rows']:,}")
    d.metric("Missing cells", f"{p['missing_pct']}%")
    for warning in dataset.warnings:
        st.caption(f"⚠️ {md_escape(warning)}")
    if p["identifier_columns"]:
        st.caption("Identifier columns (not analysed as numbers): "
                   + md_escape(", ".join(p["identifier_columns"])))

    left, right = st.columns(2)
    left.button("Remove this dataset", on_click=_remove, args=(dataset.dataset_id,))
    right.button("🗑️ Clear all datasets", on_click=_clear_all)

    tab_profile, tab_preview, tab_chart = st.tabs(["🧾 Profile", "👀 Preview", "📈 Quick chart"])
    with tab_profile:
        st.dataframe(_profile_table(p), width="stretch", hide_index=True)
        if p["correlations"]:
            st.markdown("**Strongest correlations**")
            st.dataframe(pd.DataFrame(p["correlations"]), width="stretch", hide_index=True)
        if p["correlation_note"]:
            st.caption(p["correlation_note"])
    with tab_preview:
        st.caption("First 50 rows. Cell contents are shown as data and never executed.")
        st.dataframe(display_frame(dataset.frame.head(50)), width="stretch")
    with tab_chart:
        st.caption("Build a chart directly - no AI call.")
        _quick_chart(dataset)

    st.caption(
        "Privacy and storage: files are read in memory and never written to disk. Datasets "
        "stay in this app's memory for your browser session only, and are removed when "
        f"cleared, after {settings.rag_session_ttl_minutes} minutes idle, or on restart. "
        "Only column names, types and computed results (never whole files) are sent to Gemini."
    )

    data_story_ui.render(dataset)

    st.subheader("💬 Ask about your data")
    if not settings.gemini_configured:
        st.caption("Questions need the Gemini key; the profile, preview and quick charts "
                   "work without it.")

    def answer(question: str) -> None:
        st.session_state.setdefault("data_thread_id", f"data-{uuid.uuid4().hex}")
        chat.run_turn(st.session_state["data_thread_id"], user_text=question,
                      data_session_id=session_id, answer_mode="data",
                      history_key="data_results")

    # Page level (not inside a tab): the input is pinned to the bottom and new answers
    # appear after the history, so the box never ends up mid-page or off-screen.
    chat.conversation(
        "data_results", input_key="data_question", enabled=settings.gemini_configured,
        placeholder="Ask about your data, e.g. Which products generate the highest revenue?",
        answer=answer,
    )
    if st.session_state["data_results"]:
        st.button("Clear conversation", on_click=_clear_conversation)
