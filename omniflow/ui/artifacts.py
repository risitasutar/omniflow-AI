"""Render computed results (tables, charts, SQL) and export them safely."""

from __future__ import annotations

import csv
import io
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from omniflow.agent.verification import md_escape

_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def safe_csv(table: dict[str, Any]) -> bytes:
    """CSV export that neutralises spreadsheet formula injection (=, +, -, @ ...)."""
    def cell(value):
        if isinstance(value, str) and value.startswith(_FORMULA_START):
            return "'" + value
        return "" if value is None else value

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([cell(c) for c in table["columns"]])
    for row in table["rows"]:
        writer.writerow([cell(v) for v in row])
    return buffer.getvalue().encode("utf-8-sig")


def display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """A copy that Streamlit can send to the browser via Apache Arrow.

    Arrow needs one type per column. Columns that mix value types (e.g. an Excel pivot
    "Column Labels" column holding both numbers and text) are shown as text in this
    display copy only; the stored dataset and every computed value are left untouched.
    Homogeneous columns, numeric or not, are passed through unchanged.
    """
    mixed = [
        c for c in frame.columns
        if frame[c].dtype == object
        and len({type(v) for v in frame[c].dropna()}) > 1
    ]
    if not mixed:
        return frame
    shown = frame.copy()
    for c in mixed:
        shown[c] = shown[c].map(lambda v: None if pd.isna(v) else str(v))
    return shown


def table_frame(table: dict[str, Any]) -> pd.DataFrame:
    columns = list(table["columns"])
    seen: dict[str, int] = {}
    for i, c in enumerate(columns):  # Streamlit needs unique column names
        seen[c] = seen.get(c, 0) + 1
        if seen[c] > 1:
            columns[i] = f"{c} ({seen[c]})"
    return display_frame(pd.DataFrame(table["rows"], columns=columns))


def forecast_chart(recipe: dict[str, Any]) -> alt.LayerChart:
    """History, holdout predictions, future forecast and the approximate 95% band, with a
    rule at the last observed period (the history/forecast boundary)."""
    order = [r["period"] for r in recipe["data"]]
    rows = []
    for r in recipe["data"]:
        for series_name, key in (("history", "actual"), ("holdout prediction",
                                                          "holdout_prediction"),
                                 ("forecast", "forecast")):
            if r.get(key) is not None:
                rows.append({"p": r["period"], "v": r[key], "s": series_name})
    band = [{"p": r["period"], "lo": r["lower_95"], "hi": r["upper_95"]}
            for r in recipe["data"] if r.get("lower_95") is not None]
    x = alt.X("p:O", sort=order, title=None)
    lines = alt.Chart(alt.Data(values=rows)).mark_line(point=True).encode(
        x=x, y=alt.Y("v:Q", title=str(recipe.get("y_field", ""))),
        color=alt.Color("s:N", title=None, scale=alt.Scale(
            domain=["history", "holdout prediction", "forecast"],
            range=["#1F2937", "#F59E0B", "#4F46E5"])),
        strokeDash=alt.StrokeDash("s:N", legend=None, scale=alt.Scale(
            domain=["history", "holdout prediction", "forecast"],
            range=[[1, 0], [4, 3], [6, 3]])),
        tooltip=[alt.Tooltip("p:O", title="period"), alt.Tooltip("s:N", title="series"),
                 alt.Tooltip("v:Q", title="value")])
    layers = []
    if band:
        layers.append(alt.Chart(alt.Data(values=band)).mark_area(opacity=0.18,
                                                                 color="#4F46E5").encode(
            x=x, y="lo:Q", y2="hi:Q",
            tooltip=[alt.Tooltip("p:O", title="period"),
                     alt.Tooltip("lo:Q", title="lower 95%"),
                     alt.Tooltip("hi:Q", title="upper 95%")]))
    layers.append(lines)
    if recipe.get("boundary"):
        layers.append(alt.Chart(alt.Data(values=[{"p": recipe["boundary"]}])).mark_rule(
            strokeDash=[3, 3], color="#6B7280").encode(x=x))
    return alt.layer(*layers).properties(title=recipe.get("title", ""), height=340)


def chart_object(recipe: dict[str, Any]) -> alt.Chart:
    kind = recipe["type"]
    if kind == "forecast":
        return forecast_chart(recipe)
    x, y = recipe.get("x_field"), recipe.get("y_field")
    x2, color = recipe.get("x2_field"), recipe.get("color_field")
    # Column names come from user files and may contain ':', '.' or brackets, which Altair
    # would parse as encoding syntax. Plot fixed keys and keep real names as titles only.
    rows = [{"x": r.get(x), "y": r.get(y),
             **({"x2": r.get(x2)} if x2 else {}),
             **({"c": r.get(color)} if color else {}),
             **({"v": r.get("value")} if kind == "heatmap" else {})}
            for r in recipe["data"]]
    base = alt.Chart(alt.Data(values=rows)).properties(title=recipe.get("title", ""), height=320)
    xt, yt = str(x), str(y)
    if kind == "heatmap":
        return base.mark_rect().encode(
            x=alt.X("x:N", title=None), y=alt.Y("y:N", title=None),
            color=alt.Color("v:Q", title="r",
                            scale=alt.Scale(domain=[-1, 1], scheme="blueorange")),
            tooltip=[alt.Tooltip("x:N"), alt.Tooltip("y:N"), alt.Tooltip("v:Q", title="r")],
        )
    if kind == "histogram":
        return base.mark_bar().encode(
            x=alt.X("x:Q", bin="binned", title=recipe.get("title", "")),
            x2="x2:Q", y=alt.Y("y:Q", title="count"),
            tooltip=[alt.Tooltip("x:Q", title="from"), alt.Tooltip("x2:Q", title="to"),
                     alt.Tooltip("y:Q", title="count")],
        )
    if kind == "scatter":
        encoding = {"x": alt.X("x:Q", title=xt), "y": alt.Y("y:Q", title=yt),
                    "tooltip": [alt.Tooltip("x:Q", title=xt), alt.Tooltip("y:Q", title=yt)]}
        if color:
            encoding["color"] = alt.Color("c:N", title=str(color))
        return base.mark_circle(opacity=0.6).encode(**encoding)
    if kind == "line":
        xtype = "Q" if isinstance(rows[0]["x"], (int, float)) else "O"
        return base.mark_line(point=True).encode(
            x=alt.X(f"x:{xtype}", sort=None, title=xt), y=alt.Y("y:Q", title=yt),
            tooltip=[alt.Tooltip(f"x:{xtype}", title=xt), alt.Tooltip("y:Q", title=yt)],
        )
    return base.mark_bar().encode(
        x=alt.X("x:N", sort="-y", title=xt), y=alt.Y("y:Q", title=yt),
        tooltip=[alt.Tooltip("x:N", title=xt), alt.Tooltip("y:Q", title=yt)],
    )


def render_chart(recipe: dict[str, Any]) -> None:
    if not recipe.get("data"):
        st.info("The chart has no data to show.")
        return
    st.altair_chart(chart_object(recipe), width="stretch")
    for note in recipe.get("notes", []):
        st.caption(md_escape(note))


def render_table(table: dict[str, Any], *, key: str) -> None:
    st.dataframe(table_frame(table), width="stretch", hide_index=True)
    shown = len(table["rows"])
    if table.get("truncated"):
        st.caption(f"Showing {shown} of {table.get('total_rows', shown)} rows.")
    st.download_button("Download CSV", safe_csv(table), file_name="omniflow_result.csv",
                       mime="text/csv", key=key)


def _sql_status(data: dict) -> str:
    limit = " (row limit reached)" if data.get("truncated") else ""
    return f"{data.get('row_count')} row(s){limit} in {data.get('duration_ms')} ms"


def artifacts_from_results(tool_results: list[dict], planned_calls: list[dict]) -> list[dict]:
    """Tables, charts and SQL produced by data / SQL tools in one turn (successes and
    failures), in a JSON-friendly form that can be stored in the chat history."""
    sql_by_call = {c["id"]: c["args"].get("sql") for c in planned_calls if c["tool"] == "sql_query"}
    artifacts = []
    for r in tool_results:
        data = r["data"]
        kind = data.get("source_type")
        if r["tool"] == "sql_query":
            artifacts.append({
                "kind": "sql", "ok": r["status"] == "ok",
                "submitted_sql": sql_by_call.get(r["call_id"]),
                "executed_sql": data.get("sql"),
                "database": data.get("database"), "sample_data": data.get("sample_data"),
                "table": data.get("table"), "error": r.get("error_message"),
                "status": None if r["status"] != "ok" else _sql_status(data),
            })
        elif r["status"] == "ok" and data.get("image_path"):
            artifacts.append({"kind": "image_file", "ok": True, "path": data["image_path"],
                              "title": data["results"][0]["title"],
                              "mock": bool(data.get("mock"))})
        elif kind == "data" and r["status"] == "ok":
            artifacts.append({
                "kind": "data", "ok": True, "title": data["results"][0]["title"],
                "description": data.get("description"), "table": data.get("table"),
                "chart": data.get("chart"), "notes": data.get("notes", []),
                "metrics": data.get("metrics"),
            })
    return artifacts


def render_artifacts(artifacts: list[dict], *, key: str) -> None:
    for i, a in enumerate(artifacts):
        if a["kind"] == "image_file":
            from pathlib import Path

            path = Path(a["path"])
            st.caption(md_escape(a["title"]) + (f" · saved as `{path.name}`"))
            if path.is_file():
                st.image(str(path), width=320)
            else:
                st.caption("The image file is no longer available.")
            continue
        if a["kind"] == "sql":
            label = "SAMPLE DATA · " if a.get("sample_data") else ""
            with st.expander(f"🗄️ SQL {label}- {'executed' if a['ok'] else 'not executed'}",
                             expanded=True):
                st.code(a.get("executed_sql") or a.get("submitted_sql") or "", language="sql")
                if a["ok"]:
                    st.caption(f"✅ Read-only query on `{a['database']}`: {a['status']}")
                    if a.get("table") and a["table"]["rows"]:
                        render_table(a["table"], key=f"{key}-sql-{i}")
                    else:
                        st.info("The query returned no rows.")
                else:
                    st.error(md_escape(a.get("error") or "The query failed."))
        else:
            with st.expander(f"📊 {md_escape(a['title'])} - computed by pandas", expanded=True):
                if a.get("description"):
                    st.caption(md_escape(a["description"]))
                if a.get("chart"):
                    render_chart(a["chart"])
                elif a.get("table"):
                    render_table(a["table"], key=f"{key}-data-{i}")
                for note in a.get("notes", []):
                    st.caption(f"• {md_escape(note)}")
