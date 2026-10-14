"""Deterministic data story: findings, KPIs, comparisons, trends and recommendations.

Everything here is computed with pandas/NumPy from the session's dataset; Gemini only turns
the result into prose. Each finding has an id (F1, F2...) and the exact numbers behind it.
Each recommendation cites the finding ids that motivate it, its assumptions and a
confidence label:
- high-confidence: the finding is clear in the data and the action is low-risk (e.g. verify
  data, investigate a statistically detectable decline);
- tentative: plausible, but the data can't show why (no causal claims);
- requires more data: the dataset is too small or lacks a needed column.
Recommendations never promise outcomes (savings, revenue, demand).
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from omniflow.data import series as series_mod
from omniflow.data import trends
from omniflow.data.analysis import AnalysisError, resolve_column
from omniflow.data.ingest import json_safe

MAX_GROUP_COLUMNS = 2
MAX_GROUPS = 30
_MEASURE_HINTS = re.compile(r"(?i)revenue|sales|amount|total|profit|income|value|price|cost|"
                            r"spend|qty|quantity|units|volume|orders")


def choose_measure(frame: pd.DataFrame, roles: dict[str, str],
                   measure: str | None) -> tuple[str | None, str]:
    numeric = [c for c, r in roles.items() if r == "numeric"]
    if measure:
        column = resolve_column(frame, measure)
        if roles.get(column) == "identifier":
            raise AnalysisError(f"'{column}' looks like an identifier, not a measure.")
        if column not in numeric:
            raise AnalysisError(f"'{column}' is not a numeric column.")
        return column, "chosen by you"
    if not numeric:
        return None, "no numeric (non-identifier) column"
    if len(numeric) == 1:
        return numeric[0], "the only numeric column"
    hinted = [c for c in numeric if _MEASURE_HINTS.search(c)]
    if hinted:
        return hinted[0], f"its name suggests a key measure (of {len(numeric)} numeric columns)"
    return numeric[0], (f"the first of {len(numeric)} numeric columns - choose the measure "
                        "that matters to you")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.4g}" if abs(value) < 1e6 else f"{value:,.0f}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def build(dataset, *, measure: str | None = None, date_column: str | None = None,
          group_columns: list[str] | None = None, aggregation: str = "sum",
          forecast: dict | None = None) -> dict:
    frame, roles, profile = dataset.frame, dataset.roles, dataset.profile
    findings: list[dict] = []
    limitations: list[str] = []

    def finding(text: str, kind: str, **values) -> str:
        fid = f"F{len(findings) + 1}"
        findings.append({"id": fid, "kind": kind, "statement": text,
                         "values": {k: json_safe(v) if isinstance(v, (int, float, np.number))
                                    else v for k, v in values.items()}})
        return fid

    rows, cols = profile["rows"], profile["columns"]
    quality = {"rows": rows, "columns": cols, "duplicate_rows": profile["duplicate_rows"],
               "missing_cells_percent": profile["missing_pct"],
               "identifier_columns": profile["identifier_columns"],
               "warnings": dataset.warnings}
    quality_id = None
    if profile["missing_pct"] > 5 or profile["duplicate_rows"]:
        quality_id = finding(
            f"Data quality: {profile['missing_pct']}% of cells are missing and "
            f"{profile['duplicate_rows']} row(s) are exact duplicates.", "data_quality",
            missing_percent=profile["missing_pct"], duplicate_rows=profile["duplicate_rows"])
    if rows < 30:
        limitations.append(f"Only {rows} rows: patterns may not be reliable.")

    column, why = choose_measure(frame, roles, measure)
    out: dict[str, Any] = {"dataset": dataset.name, "quality": quality,
                           "measure": column, "measure_reason": why}
    if column is None:
        limitations.append("There is no numeric measure, so totals, trends and forecasts "
                           "aren't possible.")
    else:
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.dropna()
        kpis = {"count": int(valid.size), "missing": int(values.isna().sum()),
                "total": json_safe(valid.sum()), "mean": json_safe(valid.mean()),
                "median": json_safe(valid.median()), "min": json_safe(valid.min()),
                "max": json_safe(valid.max())}
        out["kpis"] = kpis
        finding(f"'{column}': total {_fmt(kpis['total'])}, mean {_fmt(kpis['mean'])}, median "
                f"{_fmt(kpis['median'])}, range {_fmt(kpis['min'])} to {_fmt(kpis['max'])} "
                f"over {kpis['count']} rows ({kpis['missing']} missing).", "kpi", **kpis)
        if kpis["missing"]:
            limitations.append(f"{kpis['missing']} row(s) have no '{column}' value; they're "
                               "excluded, not filled.")
        non_negative = bool((valid >= 0).all())

        # --- group comparisons
        groups = list(group_columns or [])[:MAX_GROUP_COLUMNS]
        if not groups:
            groups = [c for c, r in roles.items() if r in ("categorical", "boolean")
                      and 2 <= frame[c].nunique(dropna=True) <= MAX_GROUPS][:MAX_GROUP_COLUMNS]
        comparisons = []
        for g in groups:
            g = resolve_column(frame, g)
            if g == column:
                continue
            agg = (pd.DataFrame({"group": frame[g].astype("string").fillna("(missing)"),
                                 "value": values})
                   .dropna(subset=["value"]).groupby("group")["value"]
                   .agg(["sum", "mean", "count"]))
            if agg.empty or len(agg) > 200:
                limitations.append(f"'{g}' has too many groups to compare ({len(agg)}).")
                continue
            metric = "sum" if aggregation == "sum" else "mean"
            agg = agg.sort_values(metric, ascending=False)
            total = float(agg["sum"].sum())
            rows_out = []
            for name, row in agg.head(15).iterrows():
                share = (round(float(row["sum"]) / total * 100, 2)
                         if non_negative and total > 0 else None)
                rows_out.append({"group": str(name), "total": json_safe(row["sum"]),
                                 "mean": json_safe(row["mean"]), "rows": int(row["count"]),
                                 "share_percent": share})
            top, bottom = rows_out[0], agg.tail(1)
            comparison = {"group_column": g, "metric": metric, "groups": int(len(agg)),
                          "top": rows_out, "bottom": {"group": str(bottom.index[0]),
                                                      metric: json_safe(bottom[metric].iloc[0])}}
            fid = finding(
                f"By '{g}' ({metric} of '{column}'): highest is '{top['group']}' "
                f"({_fmt(top['total' if metric == 'sum' else 'mean'])}"
                + (f", {top['share_percent']}% of the total" if top["share_percent"] is not None
                   else "") + f"); lowest is '{comparison['bottom']['group']}' "
                f"({_fmt(comparison['bottom'][metric])}) across {len(agg)} groups.",
                "comparison", group_column=g, top_group=top["group"],
                top_value=top["total" if metric == "sum" else "mean"],
                top_share_percent=top["share_percent"])
            comparison["finding"] = fid
            if top["share_percent"] is not None:
                top3 = round(sum(r["share_percent"] for r in rows_out[:3]), 2)
                comparison["top3_share_percent"] = top3
            comparisons.append(comparison)
        out["comparisons"] = comparisons
        if not comparisons:
            limitations.append("No suitable category column (2-30 distinct values) to compare "
                               "groups.")

        # --- trend
        date_columns = [c for c, r in roles.items() if r == "datetime"]
        chosen_date = resolve_column(frame, date_column) if date_column else (
            date_columns[0] if len(date_columns) == 1 else None)
        if chosen_date is None:
            limitations.append("No single date column was found (or several exist), so trends "
                               "weren't analysed. Choose a date column to add them.")
        else:
            try:
                prepared = series_mod.prepare(frame, roles, date_column=chosen_date,
                                              value_column=column,
                                              aggregation="mean" if aggregation == "mean"
                                              else "sum")
                trend = trends.analyse(prepared)
            except (AnalysisError, ValueError) as exc:
                limitations.append(f"Trend not analysed: {exc}")
                trend = None
            if trend:
                overall = trend["overall"]
                fid = finding(
                    f"Over time ({prepared.aggregation} per {prepared.frequency}, "
                    f"{overall['first_period']} to {overall['last_period']}): '{column}' went "
                    f"from {_fmt(overall['first_value'])} to {_fmt(overall['last_value'])} "
                    + (f"({overall['percent_change']:+}%)" if overall['percent_change']
                       is not None else "(% change undefined: starts at 0)")
                    + f"; trend: {trend['direction']}"
                    + (f" (Mann-Kendall p = {trend['trend_test']['p_value']})"
                       if trend["trend_test"].get("assessed") else "") + ".",
                    "trend", direction=trend["direction"], **{
                        k: overall[k] for k in ("first_value", "last_value", "absolute_change",
                                                "percent_change")})
                trend["finding"] = fid
                if trend["anomalies"]:
                    trend["anomaly_finding"] = finding(
                        f"{len(trend['anomalies'])} unusual period(s) compared with nearby "
                        "periods: " + ", ".join(f"{a['period']} ({a['kind']}, value "
                                                f"{_fmt(a['value'])})"
                                                for a in trend["anomalies"][:5]) + ".",
                        "anomaly", periods=[a["period"] for a in trend["anomalies"]])
                if trend["largest_decreases"]:
                    d = trend["largest_decreases"][0]
                    finding(f"Largest drop: {d['previous_period']} to {d['period']}, "
                            f"{_fmt(d['change'])}"
                            + (f" ({d['percent_change']}%)" if d["percent_change"] is not None
                               else "") + ".", "change", **{k: d[k] for k in (
                                   "change", "percent_change")})
                out["trend"] = trend

        # --- outliers in the raw measure (rows, not periods)
        if valid.size >= 8:
            q1, q3 = np.percentile(valid, [25, 75])
            low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
            count = int(((valid < low) | (valid > high)).sum())
            out["outliers"] = {"method": "1.5 x IQR rule on rows", "count": count,
                               "lower_fence": json_safe(low), "upper_fence": json_safe(high)}
            if count:
                out["outliers"]["finding"] = finding(
                    f"{count} row(s) of '{column}' fall outside the 1.5 x IQR range "
                    f"[{_fmt(float(low))}, {_fmt(float(high))}] - statistical flags, not "
                    "confirmed errors.", "outliers", count=count)

    # --- correlations (association only)
    strong = [c for c in profile.get("correlations", []) if abs(c["pearson_r"]) >= 0.7]
    out["correlations"] = []
    for c in strong[:3]:
        fid = finding(f"'{c['a']}' and '{c['b']}' move together (Pearson r = {c['pearson_r']})"
                      " - an association, not evidence that one causes the other.",
                      "correlation", r=c["pearson_r"])
        out["correlations"].append(c | {"finding": fid})

    if forecast:
        out["forecast_summary"] = {k: forecast[k] for k in ("selected", "selection_reason",
                                                            "horizon")}
        out["forecast_summary"]["finding"] = finding(
            f"Forecast check: {forecast['selection_reason']}", "forecast",
            selected=forecast["selected"])

    out["findings"] = findings
    out["limitations"] = limitations
    out["recommendations"] = recommend(out, quality_id)
    out["follow_ups"] = follow_ups(out)
    out["methods"] = {
        "comparisons": "group totals/means with pandas; shares only when all values are >= 0",
        "trend": "see the trend analysis methods (Mann-Kendall, robust anomalies)",
        "outliers": "1.5 x IQR rule on individual rows",
        "correlation": "Pearson r between numeric columns; |r| >= 0.7 reported",
    }
    out["chart"] = _chart(out)
    return out


def recommend(story: dict, quality_id: str | None) -> list[dict]:
    recs: list[dict] = []

    def add(action, finding_ids, evidence, confidence, assumptions):
        recs.append({"id": f"R{len(recs) + 1}", "action": action, "based_on": finding_ids,
                     "evidence": evidence, "confidence": confidence,
                     "assumptions": assumptions})

    q = story["quality"]
    if quality_id:
        add("Review missing values and duplicate rows (and fix them at the source) before "
            "relying on totals or trends.", [quality_id],
            f"{q['missing_cells_percent']}% missing cells; {q['duplicate_rows']} duplicate rows",
            "high-confidence", "Duplicates may be legitimate repeated records; check before "
                               "removing them.")
    trend = story.get("trend")
    if trend:
        direction = trend["direction"]
        ids = [trend["finding"]]
        if direction == "decreasing":
            add(f"Investigate the sustained decline in '{story['measure']}' (start with the "
                "largest drops and the groups that changed most).", ids,
                f"Mann-Kendall p = {trend['trend_test']['p_value']}",
                "high-confidence", "The test assumes independent periods; it shows a decline "
                                   "exists, not why.")
        elif direction == "increasing":
            add(f"Identify what drove the rise in '{story['measure']}' before planning around "
                "it continuing.", ids, f"Mann-Kendall p = {trend['trend_test']['p_value']}",
                "tentative", "Past growth doesn't guarantee future growth; the data doesn't "
                             "show the cause.")
        if not trend["trend_test"].get("assessed") or not trend["seasonality"].get("assessed"):
            needed = trend["seasonality"].get("season_length")
            add("Collect more history before drawing trend or seasonal conclusions"
                + (f" (at least {2 * needed} periods for a seasonal check)" if needed else "")
                + ".", ids, f"{trend['overall']['non_empty_periods']} non-empty periods",
                "requires more data", "More periods of the same frequency are needed.")
        if trend.get("anomaly_finding"):
            add("Validate the unusual periods (data entry, one-off events) before making "
                "operational decisions based on them.", [trend["anomaly_finding"]],
                ", ".join(a["period"] for a in trend["anomalies"][:5]), "high-confidence",
                "A flag means unusual relative to nearby periods, not an error or fraud.")
    elif story.get("measure"):
        add("Add or choose a date column to analyse trends over time.", [], "no usable date "
            "column", "requires more data", "Trends need dated records.")
    for comparison in story.get("comparisons", []):
        top = comparison["top"][0]
        if top.get("share_percent") is not None and top["share_percent"] >= 50:
            add(f"Review how much '{story['measure']}' depends on '{top['group']}' "
                f"({comparison['group_column']}) and what would happen if it changed.",
                [comparison["finding"]], f"{top['share_percent']}% of the total",
                "tentative", "Concentration is a risk only if that group is uncertain; the "
                             "data can't tell.")
    if story.get("outliers", {}).get("finding"):
        add("Check the flagged outlier rows before using averages; consider reporting the "
            "median alongside the mean.", [story["outliers"]["finding"]],
            f"{story['outliers']['count']} row(s) outside the IQR fences", "high-confidence",
            "Outliers can be genuine large values.")
    for c in story.get("correlations", []):
        add(f"If it matters, test whether '{c['a']}' affects '{c['b']}' with a controlled "
            "comparison; the correlation alone doesn't show cause.", [c["finding"]],
            f"r = {c['pearson_r']}", "tentative", "Both may be driven by a third factor.")
    forecast = story.get("forecast_summary")
    if forecast:
        if forecast["selected"] == "naive":
            add("Plan with the simple baseline (last value) and its range; no method beat it "
                "on past data.", [forecast["finding"]], forecast["selection_reason"],
                "high-confidence", "Holdout results describe the past, not the future.")
        else:
            add(f"Use the '{forecast['selected']}' forecast as one input, compare it with the "
                "baseline, and re-check it as new data arrives.", [forecast["finding"]],
                forecast["selection_reason"], "tentative",
                "Forecasts are estimates; intervals are approximate.")
    if not recs:
        recs.append({"id": "R1", "action": "No specific action is supported by this data; "
                     "explore a specific question instead.", "based_on": [], "evidence": "",
                     "confidence": "requires more data", "assumptions": ""})
    return recs


def follow_ups(story: dict) -> list[str]:
    ideas = []
    if story.get("trend") and story.get("comparisons"):
        ideas.append(f"Trend of '{story['measure']}' for the top "
                     f"'{story['comparisons'][0]['group_column']}' groups separately.")
    if story.get("comparisons"):
        ideas.append(f"Bar chart of '{story['measure']}' by "
                     f"'{story['comparisons'][0]['group_column']}'.")
    if story.get("trend") and story["trend"]["overall"]["non_empty_periods"] >= 12:
        ideas.append(f"Forecast '{story['measure']}' and compare methods against the naive "
                     "baseline.")
    return ideas


def _chart(story: dict) -> dict | None:
    if story.get("comparisons"):
        c = story["comparisons"][0]
        key = "total" if c["metric"] == "sum" else "mean"
        return {"type": "bar", "title": f"{story['measure']} by {c['group_column']} "
                                        f"({c['metric']})",
                "x_field": c["group_column"], "y_field": story["measure"],
                "data": [{c["group_column"]: r["group"], story["measure"]: r[key]}
                         for r in c["top"]], "notes": []}
    if story.get("trend"):
        return story["trend"]["chart"]
    return None
