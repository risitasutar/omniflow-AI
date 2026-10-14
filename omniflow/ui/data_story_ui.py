"""Data page section: data story, trends, outliers, forecasts and recommendations.

All numbers are computed here, deterministically, when you click a button; nothing is
saved and the dataset is never modified. "Explain with Gemini" sends a precise request to
the page's chat, which re-runs the same tool and explains the computed result.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from omniflow.agent.verification import md_escape
from omniflow.core.config import get_settings
from omniflow.data.analysis import AnalysisArgs, AnalysisError
from omniflow.ui import state
from omniflow.ui.artifacts import render_chart, render_table

ACTIONS = ["Data story", "Trends", "Outliers", "Forecast", "Recommendations"]
CONFIDENCE = {"high-confidence": "🟢 high-confidence", "tentative": "🟡 tentative",
              "requires more data": "⚪ requires more data"}


def _columns(dataset, *roles) -> list[str]:
    return [c for c, r in dataset.roles.items() if r in roles]


def _date_candidates(dataset) -> list[str]:
    dated = _columns(dataset, "datetime")
    others = [c for c in dataset.frame.columns if c not in dated and dataset.roles.get(c) in
              ("text", "categorical")]
    return dated + others


def _explain(request: str) -> None:
    st.session_state["data_question_pending"] = request


def _store(dataset, kind: str, result) -> None:
    st.session_state.setdefault("data_story_results", {})[(dataset.dataset_id, kind)] = result


def _stored(dataset, kind: str):
    return st.session_state.get("data_story_results", {}).get((dataset.dataset_id, kind))


def _download(result: dict, name: str, key: str) -> None:
    st.download_button("Download results (JSON)", json.dumps(result, indent=2, default=str),
                       file_name=name, mime="application/json", key=key)


def _show_recommendations(recs: list[dict]) -> None:
    for r in recs:
        st.markdown(f"**{r['id']} · {CONFIDENCE.get(r['confidence'], r['confidence'])}** - "
                    f"{md_escape(r['action'])}  \n"
                    f"<small>Based on {', '.join(r['based_on']) or 'no specific finding'}"
                    f"{' · evidence: ' + md_escape(str(r['evidence'])) if r['evidence'] else ''}"
                    f"{' · assumes: ' + md_escape(r['assumptions']) if r['assumptions'] else ''}"
                    "</small>", unsafe_allow_html=True)


def _story(dataset, enabled: bool) -> None:
    numeric = _columns(dataset, "numeric")
    with st.form(f"story-{dataset.dataset_id}"):
        a, b = st.columns(2)
        measure = a.selectbox("Main measure", ["(let OmniFlow choose)"] + numeric)
        date = b.selectbox("Date column", ["(automatic)"] + _date_candidates(dataset))
        groups = st.multiselect("Compare by (up to 2)", _columns(dataset, "categorical",
                                                                 "boolean"), max_selections=2)
        aggregation = st.radio("Combine values with", ["sum", "mean"], horizontal=True,
                               help="sum for totals (revenue, units); mean for levels "
                                    "(prices, scores).")
        go = st.form_submit_button("Generate data story", type="primary")
    options = {"measure": None if measure.startswith("(") else measure,
               "date_column": None if date.startswith("(") else date,
               "group_columns": groups, "aggregation": aggregation}
    if go:
        try:
            _, result = state.get_data_service().story(st.session_state["data_session_id"],
                                                       dataset.name, **options)
        except AnalysisError as exc:
            st.error(str(exc))
            return
        _store(dataset, "story", (options, result))
    stored = _stored(dataset, "story")
    if not stored:
        return
    options, story = stored
    st.caption(f"Main measure: **{md_escape(str(story['measure']))}** "
               f"({md_escape(story['measure_reason'])}).")
    if story.get("kpis"):
        k = story["kpis"]
        cols = st.columns(4)
        for col, label in zip(cols, ("total", "mean", "median", "count"), strict=True):
            col.metric(label.capitalize(), f"{k[label]:,.4g}" if isinstance(k[label], float)
                       else f"{k[label]:,}")
    st.markdown("**Findings (computed)**")
    for f in story["findings"]:
        st.markdown(f"- **{f['id']}** {md_escape(f['statement'])}")
    if story.get("chart"):
        render_chart(story["chart"])
    for c in story.get("comparisons", []):
        with st.expander(f"Comparison by {md_escape(c['group_column'])} ({c['groups']} groups)"):
            render_table({"columns": ["group", "total", "mean", "rows", "share %"],
                          "rows": [[r["group"], r["total"], r["mean"], r["rows"],
                                    r["share_percent"]] for r in c["top"]],
                          "total_rows": len(c["top"])}, key=f"cmp-{c['group_column']}")
    if story["limitations"]:
        st.markdown("**Limitations**\n" + "\n".join(f"- {md_escape(x)}"
                                                    for x in story["limitations"]))
    st.markdown("**Recommendations (rule-based, from the findings)**")
    _show_recommendations(story["recommendations"])
    if story.get("follow_ups"):
        st.caption("Follow-up ideas: " + " · ".join(md_escape(x) for x in story["follow_ups"]))
    _download(story, "data_story.json", f"dl-story-{dataset.dataset_id}")
    if enabled:
        st.button("Explain for a non-technical audience (Gemini)", on_click=_explain, args=(
            f"Call dataset_story on dataset '{dataset.name}' with measure="
            f"'{options['measure'] or ''}', date_column='{options['date_column'] or ''}', "
            f"group_columns={options['group_columns']}, aggregation='{options['aggregation']}'"
            " and explain the findings and recommendations for a non-technical audience.",))


def _series_form(dataset, key: str, *, forecast: bool):
    settings = get_settings()
    numeric = _columns(dataset, "numeric")
    with st.form(f"{key}-{dataset.dataset_id}"):
        a, b = st.columns(2)
        date = a.selectbox("Date column", _date_candidates(dataset), index=None,
                           placeholder="Choose the date column")
        value = b.selectbox("Numeric column", numeric, index=None,
                            placeholder="Choose what to analyse")
        c, d = st.columns(2)
        aggregation = c.radio("Combine rows in a period with", ["sum", "mean"],
                              horizontal=True)
        frequency = d.selectbox("Period", ["auto", "day", "week", "month", "quarter", "year"])
        horizon, gap_policy = None, "refuse"
        if forecast:
            horizon = c.number_input("Forecast horizon (periods)", min_value=1,
                                     max_value=settings.data_forecast_max_horizon, value=6)
            gap_policy = d.selectbox(
                "Empty periods", ["refuse", "zero", "interpolate"],
                format_func={"refuse": "Don't forecast if there are gaps",
                             "zero": "Treat as zero (totals only)",
                             "interpolate": "Interpolate short gaps"}.get)
        label = "Evaluate methods and forecast" if forecast else "Analyze trend"
        go = st.form_submit_button(label, type="primary")
    if go and (not date or not value):
        st.warning("Choose both a date column and a numeric column - OmniFlow won't guess the "
                   "target.")
        return None
    return (go, {"date_column": date, "value_column": value, "aggregation": aggregation,
                 "frequency": frequency, **({"horizon": int(horizon), "gap_policy": gap_policy}
                                            if forecast else {})})


def _series_caption(info: dict) -> None:
    parts = [f"{info['rows_used']:,} of {info['rows_in_dataset']:,} rows used",
             f"{info['aggregation']} per {info['frequency']} ({info['frequency_source']})",
             f"{info['first_period']} to {info['last_period']}"]
    for key, label in (("unparseable_dates", "unparseable dates dropped"),
                       ("missing_values_dropped", "missing values dropped"),
                       ("duplicate_timestamps", "duplicate timestamps combined"),
                       ("empty_periods", "empty periods (gaps)")):
        if info.get(key):
            parts.append(f"{info[key]} {label}")
    st.caption(" · ".join(md_escape(str(p)) for p in parts))


def _trends(dataset, enabled: bool) -> None:
    submitted = _series_form(dataset, "trend", forecast=False)
    if submitted and submitted[0]:
        args = submitted[1]
        try:
            _, result = state.get_data_service().trends(
                st.session_state["data_session_id"], dataset.name, **args)
        except AnalysisError as exc:
            st.error(str(exc))
            return
        _store(dataset, "trends", (args, result))
    stored = _stored(dataset, "trends")
    if not stored:
        return
    args, t = stored
    _series_caption(t["series"])
    o = t["overall"]
    a, b, c = st.columns(3)
    a.metric("Direction", t["direction"].split(" (")[0])
    b.metric(f"{o['first_period']} → {o['last_period']}",
             f"{o['last_value']:,.4g}", f"{o['percent_change']}%"
             if o["percent_change"] is not None else "n/a (starts at 0)")
    test = t["trend_test"]
    c.metric("Mann-Kendall p", test["p_value"] if test.get("assessed") else "not tested")
    render_chart(t["chart"])
    if t["anomalies"]:
        st.markdown("**Unusual periods** (unusual vs neighbours - not confirmed errors)")
        render_table({"columns": ["period", "value", "neighbour median", "robust z", "kind"],
                      "rows": [[x["period"], x["value"], x["local_median"], x["robust_z"],
                                x["kind"]] for x in t["anomalies"]],
                      "total_rows": len(t["anomalies"])}, key=f"anom-{dataset.dataset_id}")
    season = t["seasonality"]
    if season.get("assessed"):
        verdict = ("possible seasonal pattern" if season["possible_seasonality"]
                   else "no clear pattern")
        st.caption(f"Seasonality: lag-{season['season_length']} autocorrelation "
                   f"{season['autocorrelation']} vs bound {season['bound']} - {verdict}")
    else:
        st.caption(f"Seasonality: {season.get('reason', '')}")
    for note in t["notes"]:
        st.caption(f"• {md_escape(note)}")
    with st.expander("Period-by-period changes and methods"):
        render_table({"columns": ["period", "value", "change", "% change"],
                      "rows": [[x["period"], x["value"], x["change"], x["percent_change"]]
                               for x in t["recent_changes"]],
                      "total_rows": len(t["recent_changes"])}, key=f"chg-{dataset.dataset_id}")
        for name, text in t["methods"].items():
            st.caption(f"**{name}**: {text}")
    _download(t, "trend_analysis.json", f"dl-trend-{dataset.dataset_id}")
    if enabled:
        st.button("Explain this trend (Gemini)", on_click=_explain, args=(
            f"Call dataset_trends on dataset '{dataset.name}' with date_column="
            f"'{args['date_column']}', value_column='{args['value_column']}', aggregation="
            f"'{args['aggregation']}', frequency='{args['frequency']}' and explain the trend, "
            "the largest changes and any unusual periods.",))


def _outliers(dataset) -> None:
    numeric = _columns(dataset, "numeric")
    if not numeric:
        st.info("This dataset has no numeric columns.")
        return
    with st.form(f"outliers-{dataset.dataset_id}"):
        a, b = st.columns(2)
        column = a.selectbox("Numeric column", numeric)
        method = b.selectbox("Method", ["iqr", "mad", "zscore"], format_func={
            "iqr": "1.5 x IQR (Tukey fences)", "mad": "Robust z (median/MAD) > 3.5",
            "zscore": "z-score > 3 (assumes roughly normal data)"}.get)
        go = st.form_submit_button("Detect outliers", type="primary")
    if go:
        try:
            _, out = state.get_data_service().analyze(
                st.session_state["data_session_id"],
                AnalysisArgs(dataset=dataset.name, operation="outliers", value_column=column,
                             method=method))
        except AnalysisError as exc:
            st.error(str(exc))
            return
        _store(dataset, "outliers", out)
    out = _stored(dataset, "outliers")
    if out:
        st.caption(md_escape(out["description"]) + f" · {out['metrics']['outlier_count']} of "
                   f"{out['metrics']['values_checked']} values flagged.")
        for note in out["notes"]:
            st.caption(f"• {md_escape(note)}")
        if out["table"]["rows"]:
            render_table(out["table"], key=f"out-{dataset.dataset_id}")


def _forecast(dataset, enabled: bool) -> None:
    st.caption("Methods are tested on past data (rolling-origin holdout) against a naive "
               "baseline before anything is forecast. Nothing is saved.")
    submitted = _series_form(dataset, "forecast", forecast=True)
    if submitted and submitted[0]:
        args = submitted[1]
        try:
            _, result = state.get_data_service().forecast(
                st.session_state["data_session_id"], dataset.name, **args)
        except AnalysisError as exc:
            st.error(str(exc))
            return
        _store(dataset, "forecast", (args, result))
    stored = _stored(dataset, "forecast")
    if not stored:
        return
    args, f = stored
    _series_caption(f["series"])
    st.success(f"**Selected: {f['selected']}** - {md_escape(f['selection_reason'])}")
    ev = f["evaluation"]
    st.caption(f"Evaluation: {ev['method']}; {ev['folds']} fold(s) of {ev['test_size_per_fold']}"
               f" period(s): {', '.join(ev['holdout_periods'])}.")
    rows = []
    for name, c in f["candidates"].items():
        if c.get("available"):
            rows.append([name, c["mae"], c["rmse"], c["mase"], c["mape_percent"],
                         c["skill_vs_naive_percent"], c["interval_coverage"],
                         c["holdout_points"]])
        else:
            rows.append([name, None, None, None, None, None, None, f"n/a: {c['reason']}"])
    st.markdown("**Model comparison on holdout data** (lower MAE/RMSE is better; skill = % "
                "MAE improvement over naive)")
    st.dataframe(pd.DataFrame(rows, columns=["method", "MAE", "RMSE", "MASE", "MAPE %",
                                             "skill vs naive %", "95% interval coverage",
                                             "holdout points"]).astype({"holdout points": str}),
                 hide_index=True, width="stretch")
    render_chart(f["chart"])
    render_table({"columns": ["period", "forecast", "lower 95%", "upper 95%"],
                  "rows": [[r["period"], r["forecast"], r["lower_95"], r["upper_95"]]
                           for r in f["forecast"]], "total_rows": len(f["forecast"])},
                 key=f"fc-{dataset.dataset_id}")
    for note in f["notes"]:
        st.caption(f"• {md_escape(note)}")
    with st.expander("Metric definitions"):
        st.caption("MAE: mean absolute error. RMSE: root mean squared error (penalises large "
                   "misses). MASE: MAE divided by the naive method's in-sample one-step MAE "
                   "(below 1 beats that naive scale). MAPE: mean absolute percentage error, "
                   "shown only when every actual value is positive. Coverage: share of "
                   "holdout actuals inside the approximate 95% interval.")
    _download(f, "forecast.json", f"dl-fc-{dataset.dataset_id}")
    if enabled:
        st.button("Explain this forecast (Gemini)", on_click=_explain, args=(
            f"Call dataset_forecast on dataset '{dataset.name}' with date_column="
            f"'{args['date_column']}', value_column='{args['value_column']}', horizon="
            f"{args['horizon']}, aggregation='{args['aggregation']}', frequency="
            f"'{args['frequency']}', gap_policy='{args['gap_policy']}' and explain the model "
            "comparison, the forecast and its limits.",))


def _recommendations(dataset, enabled: bool) -> None:
    story_state = _stored(dataset, "story")
    forecast_state = _stored(dataset, "forecast")
    st.caption("Built from the data story" + (" and your latest forecast" if forecast_state
                                              else "") + ". Each recommendation cites the "
               "findings behind it.")
    if st.button("Generate recommendations", type="primary"):
        options = story_state[0] if story_state else {}
        try:
            _, story = state.get_data_service().story(
                st.session_state["data_session_id"], dataset.name, **options,
                forecast=forecast_state[1] if forecast_state else None)
        except AnalysisError as exc:
            st.error(str(exc))
            return
        _store(dataset, "recommendations", story)
    story = _stored(dataset, "recommendations")
    if story:
        _show_recommendations(story["recommendations"])
        with st.expander("Findings behind them"):
            for f in story["findings"]:
                st.markdown(f"- **{f['id']}** {md_escape(f['statement'])}")
        _download(story, "recommendations.json", f"dl-rec-{dataset.dataset_id}")


def render(dataset) -> None:
    settings = get_settings()
    st.subheader("📖 Storytelling & forecasting")
    action = st.radio("Action", ACTIONS, horizontal=True, key="data_story_action",
                      label_visibility="collapsed")
    enabled = settings.gemini_configured
    if action == "Data story":
        _story(dataset, enabled)
    elif action == "Trends":
        _trends(dataset, enabled)
    elif action == "Outliers":
        _outliers(dataset)
    elif action == "Forecast":
        _forecast(dataset, enabled)
    else:
        _recommendations(dataset, enabled)
